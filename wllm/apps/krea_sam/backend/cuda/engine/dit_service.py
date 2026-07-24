"""DiT denoise service for the krea_vae_dit_split model-graph pipeline.

Owns the DiT half of the Krea model graph: add_noise → fill_context →
denoise_step×N → append_context (the operators whose persistent state is
`clean_latent_context` + `dit_kv` + the noise generator). It receives
encoded input latents from the coordinator and returns the denoised
latents; the VAE encode/decode runs as a separate service on its own GPU,
so the two Krea sub-stages pipeline across chunks (model-graph stages
[vae_*] ‖ [DiT]).

With ``--sp 1`` it is a single process. With ``--sp N`` it is N processes
forming a torch.distributed world that runs the DiT with Ulysses sequence
parallelism (the Krea DiT shards the latent-frame sequence via the shared
``sequence_model_parallel_*`` ops), exactly like ``krea_service.py``. All
ranks run the denoise in SPMD; rank 0 is the *driver* that talks to the
coordinator and broadcasts each chunk's encoded latents to the other
ranks. Only rank 0 returns the denoised latents (every rank computes the
identical result because the DiT all-gathers, and the per-chunk noise is
sampled from the seeded ``_noise_generator`` that every rank re-seeds in
``init_session`` and advances in lockstep).

The denoise body is vendored verbatim from KreaSAMPipeline.step so the
math is identical to the reference.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch


from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.reference.pipeline import KreaSAMPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options


def _build_pipeline(cfg, device):
    pipe = KreaSAMPipeline(cfg=cfg, device=device)
    pipe.start_instance()
    set_global_seed(cfg.seed)
    pipe.init_session(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt or None)
    # warmup the DiT path with a dummy encoded latent chunk so the first live
    # chunk pays no compile cost. Run enough iters to grow the clean-context
    # cache across the full context window: _fill_clean_context_cache autotunes
    # per context length, so warming only 2 iters (context 0 and 1) left the
    # larger context sizes to compile on live chunks. All SP ranks run this
    # identically (the dummy is deterministic), keeping the collectives lockstep.
    warm = torch.zeros((1, cfg.vae_config.z_dim, int(cfg.chunk_size),
                        cfg.latent_height, cfg.latent_width), device=device, dtype=pipe.dtype)
    for _ in range(int(cfg.context_window_size) + 3):
        _denoise(pipe, warm)
    torch.cuda.synchronize(device)
    pipe.reset()
    set_global_seed(cfg.seed)
    pipe.init_session(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt or None)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--sp", type=int, default=1)
    args = ap.parse_args()

    set_torch_options()
    torch.set_grad_enabled(False)

    sp = args.sp
    if sp > 1:
        from wllm.serving.distributed.parallel_state import (
            maybe_init_distributed_environment_and_model_parallel, get_world_rank, get_world_group)
        from wllm.serving.distributed.communication_op import warmup_sequence_parallel_communication
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp)
        rank = get_world_rank()
        world = get_world_group()
        warmup_sequence_parallel_communication(torch.device("cuda:0"))
    else:
        rank = 0
        world = None

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ref_cfg = KreaSAMReferenceConfig.from_yaml(args.cfg, is_path=True)
    cfg = ref_cfg.to_runtime_config()

    if rank == 0:
        print(f"[dit] building pipeline (DiT stage, sp={sp}) ...", flush=True)
    pipe = _build_pipeline(cfg, device)
    if rank == 0:
        print("[dit] ready", flush=True)

    conn = None
    if rank == 0:
        from wllm.apps.krea_sam.backend.cuda.engine.ipc import connect_to_coordinator
        conn = connect_to_coordinator(args.address)
        conn.send({"ack": "ready"})

    def bcast_cmd(obj):
        if sp > 1:
            return world.broadcast_object(obj, src=0)
        return obj

    def start_session():
        set_global_seed(cfg.seed)
        pipe.init_session(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt or None)

    try:
        while True:
            if rank == 0:
                msg = conn.recv()
                cmd = msg.get("cmd")
            else:
                msg = None
                cmd = None
            cmd = bcast_cmd(cmd)

            if cmd == "stop":
                break
            elif cmd in ("start", "reset"):
                start_session()
                if rank == 0:
                    conn.send({"ack": cmd})
            elif cmd == "denoise":
                # rank 0 owns the encoded latents from the coordinator and
                # broadcasts them to the other ranks; all ranks then run the
                # identical denoise (the DiT shards the frame sequence inside).
                if rank == 0:
                    cid = msg["id"]
                    input_latents = torch.from_numpy(msg["latents"]).to(device=device, dtype=pipe.dtype)
                    shape = tuple(input_latents.shape)
                else:
                    cid, shape = None, None
                if sp > 1:
                    cid, shape = world.broadcast_object((cid, shape), src=0)
                    if rank != 0:
                        input_latents = torch.empty(shape, device=device, dtype=pipe.dtype)
                    world.broadcast(input_latents, src=0)

                denoised = _denoise(pipe, input_latents)
                torch.cuda.synchronize(device)
                if rank == 0:
                    conn.send({"id": cid, "out": denoised.float().cpu().numpy()})
    finally:
        if rank == 0 and conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if sp > 1:
            from wllm.serving.distributed.parallel_state import cleanup_dist_env_and_memory
            try:
                cleanup_dist_env_and_memory()
            except Exception:
                pass


@torch.inference_mode()
def _denoise(pipe, input_latents):
    """add_noise → fill_context → denoise → append (vendored from pipe.step)."""
    init_strength = float(pipe._denoise_timesteps[0].item()) / 1000.0
    noise = pipe._sample_noise(input_latents.shape)
    noisy = input_latents * (1.0 - init_strength) + noise * init_strength
    clean_context = pipe._current_context_latents()
    context_tokens = pipe._fill_clean_context_cache(clean_context)
    denoised = pipe._denoise_latents(noisy, context_tokens)
    pipe._append_clean_latents(denoised)
    return denoised


if __name__ == "__main__":
    main()
