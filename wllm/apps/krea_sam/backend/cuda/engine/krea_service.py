"""Krea v2v service (one or more processes, one GPU each).

Runs the Krea-Realtime v2v pipeline. With ``--sp 1`` it is a single
process. With ``--sp N`` it is N processes forming a torch.distributed
world that runs the DiT with Ulysses sequence parallelism (the Krea DiT
shards the latent-frame sequence via the shared
``sequence_model_parallel_*`` ops). All ranks run the pipeline in
SPMD; rank 0 is the *driver* that talks to the coordinator and
broadcasts each chunk's input to the other ranks. Only rank 0 returns
the decoded frames (every rank ends with the identical full result: the
DiT all-gathers its sequence-sharded output, and the WAN VAE — which
reads get_world_size() at construction — splits its decode into spatial
tiles across the SP ranks and all-gathers them via gather_tile, so the
VAE work is *distributed* across the ranks, not replicated).

Launched as independent subprocess(es) by the coordinator, each with its
own CUDA_VISIBLE_DEVICES (so each rank sees its GPU as cuda:0) and the
torchrun-style RANK/WORLD_SIZE/MASTER_* env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.reference.pipeline import KreaSAMPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options


def _build_pipeline(cfg, device):
    pipe = KreaSAMPipeline(cfg=cfg, device=device)
    pipe.start_instance()
    # match the reference worker warmup so first live chunk pays no compile cost
    set_global_seed(cfg.seed)
    pipe.init_session(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt or None)
    # Run several full chunks so EVERY per-chunk path is compiled before the
    # first live chunk, not just block 0: the encoder switches stream=False ->
    # stream=True after chunk 0, the DiT clean-context cache grows over the first
    # context_window_size chunks (each a new autotune shape), and the VAE decode
    # uses is_first True then False. Recompute the frame count per chunk since it
    # changes after block 0.
    for _ in range(int(cfg.context_window_size) + 3):
        warm = torch.zeros((int(pipe.input_frames_for_next_step()), 3, cfg.height, cfg.width),
                           device=device, dtype=pipe.dtype)
        pipe.step(warm)
    torch.cuda.synchronize(device)
    pipe.reset()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--sp", type=int, default=1)
    ap.add_argument("--stream-frames", action="store_true",
                    help="stream each latent's decoded frames to the coordinator as produced")
    args = ap.parse_args()

    set_torch_options()
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
        print(f"[krea] building pipeline (sp={sp}) ...", flush=True)
    pipe = _build_pipeline(cfg, device)
    if rank == 0:
        print("[krea] ready", flush=True)

    conn = None
    if rank == 0:
        from wllm.apps.krea_sam.backend.cuda.engine.ipc import connect_to_coordinator
        conn = connect_to_coordinator(args.address)
        conn.send({"ack": "ready"})

    def bcast_cmd(obj):
        if sp > 1:
            return world.broadcast_object(obj, src=0)
        return obj

    block_idx = [0]  # session chunk counter (for the streaming path)

    def start_session():
        set_global_seed(cfg.seed)
        pipe.init_session(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt or None)
        block_idx[0] = 0

    if args.stream_frames:
        from wllm.apps.krea_sam.backend.cuda.engine.streaming_step import streaming_step

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
            elif cmd == "chunk":
                # rank 0 normalizes + broadcasts the input tensor
                if rank == 0:
                    raw = msg["frames"]  # [T,H,W,3] uint8
                    cid = msg["id"]
                    ten = torch.from_numpy(raw).to(device=device, dtype=torch.uint8)
                    krea_input = (ten.permute(0, 3, 1, 2).to(dtype=pipe.dtype)
                                  .div_(127.5).sub_(1.0).contiguous())
                    shape = tuple(krea_input.shape)
                else:
                    cid, shape = None, None
                if sp > 1:
                    cid, shape = world.broadcast_object((cid, shape), src=0)
                    if rank != 0:
                        krea_input = torch.empty(shape, device=device, dtype=pipe.dtype)
                    world.broadcast(krea_input, src=0)

                if args.stream_frames:
                    seq = [0]

                    def _on_decoded(frames_np):
                        if rank == 0:
                            conn.send({"id": cid, "seq": seq[0], "frames": frames_np,
                                       "final": False})
                            seq[0] += 1

                    streaming_step(pipe, krea_input, block_idx[0], _on_decoded)
                    torch.cuda.synchronize(device)
                    block_idx[0] += 1
                    if rank == 0:
                        conn.send({"id": cid, "final": True})
                else:
                    krea_frames = pipe.step(krea_input)
                    torch.cuda.synchronize(device)
                    if rank == 0:
                        conn.send({"id": cid, "out": krea_frames})
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


if __name__ == "__main__":
    main()
