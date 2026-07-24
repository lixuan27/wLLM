"""pp_steps: cross-chunk pipeline of the 4 DiT denoising steps across GPUs.

IR basis: find_pipeline_stages on the model graph gives 6 stages, and
analyze_cross_chunk_dependencies shows every `dit_step_i || dit_step_j` (i!=j) is
independent across chunks (they touch disjoint private caches cache_i). So step k
of chunk N can run concurrently with step k+1 of chunk N-1: a classic pipeline.

This module is a torch.distributed "cluster" of `world_size` ranks (one DiT copy
+ its private cache per rank). Rank k runs denoising step k for every chunk;
the last rank also runs the VAE decode. Latents flow rank0->1->...->last (the
within-chunk latent chain) via NCCL p2p; cache_k stays on rank k across chunks
(the cross-chunk dependency). At steady state the pipeline emits one chunk per
max-stage time (~one DiT step) instead of 4 steps + VAE — the throughput lever.

Decoupled from the app worker by shared memory: rank0 reads per-chunk wav2vec
features from `audio_q` (worker writes them) and rank{last} writes decoded video
frames to `video_q` (worker reads them). Control (init/terminate/reset) comes via
a SharedControlBuffer. This keeps the dist world a PURE DiT pipeline (no vLLM/TTS
in it -> no torch.distributed env-var collision).

Each rank reuses the Phase-2-validated IR ops (DiTDenoiseStep, VAEDecode) for its
stage, so the distributed result equals the reference computation by
construction (verified bit-exact against the reference).

Ref-latents handling (the one cross-stage edge): on the session's chunk 0 the
final step (last rank) produces the updated ref_latents (chunk0's first latent);
chunk 1's prefill on every step needs it. The last rank sends it to rank0 (ring
next), and rank0 forwards it down the chain bundled with chunk 1's message. This
serializes chunks 0,1 (a one-time startup bubble); chunks >=2 pipeline freely.
"""
from __future__ import annotations

import os
import sys
import time
import faulthandler
import signal

import numpy as np
import torch

# SIGUSR1 -> dump all threads' Python tracebacks to stderr (the rank log), so we
# can diagnose a stall without py-spy.
faulthandler.register(signal.SIGUSR1)


from wllm.serving.rt_config import RTConfig
from wllm.apps.liveavatar.reference.pipeline import LiveAvatarPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel, get_world_group,
)
from wllm.serving.distributed.communication_op import global_barrier
from wllm.apps.liveavatar.backend.cuda.ir.ops import LAContext, DiTDenoiseStep, VAEDecode
from wllm.apps.liveavatar.backend.cuda.runtime_common import _detach_shm_resource_tracker

# cluster processes ATTACH to shm created by the client; never unlink it on exit
_detach_shm_resource_tracker()

# control opcodes (cluster ctrl buffer + ring ctrl scalar)
OP_NONE, OP_INIT, OP_DATA, OP_DATA_REF, OP_TERM, OP_RESET = 0, 1, 2, 3, 4, 5


class SimpleState:
    """Minimal dict-backed StateStore for reusing the IR ops outside the executor."""
    def __init__(self, d):
        self._d = d

    def get(self, name):
        return self._d[name]

    def set(self, name, value):
        self._d[name] = value


class SingleCachePipeline(LiveAvatarPipeline):
    """LiveAvatarPipeline that allocates ONLY one denoising-step cache (this
    rank's), instead of num_inference_steps caches. init_session still fills that
    cache's encoder (text cross-attn) KV the usual way. It also does NOT create
    the app output video buffer (the cluster client owns cfg.video_buffer_name;
    the last rank opens it separately) — avoiding a create() conflict."""
    def _allocate_step_caches(self) -> None:
        if self.dit_runner is None:
            return
        # One cache per denoising step THIS rank owns: steps_per_rank =
        # num_inference_steps // world_size (1 for the 4-GPU layout, 2 for the
        # 2-GPU layout). init_session's _sync_encoder_cache_to_all_steps then
        # populates the encoder (text) KV into all of this rank's caches.
        world = int(os.environ.get("PP_WORLD", int(self.cfg.num_inference_steps)))
        spr = int(self.cfg.num_inference_steps) // world
        base = self.dit_runner.kv_memory
        cache_cls = base.__class__
        self._step_kv_caches = [base] + [cache_cls(self.cfg, self.device)
                                         for _ in range(1, spr)]
        self._activate_step_cache(0)

    @torch.inference_mode()
    def start_instance(self):
        self._timesteps, self._sigmas = self._build_timestep_schedule()
        self._timesteps = self._timesteps.to(self.device)
        self._sigmas = self._sigmas.to(self.device)
        self._latents = self.prepare_latents(torch.float32)  # consumes RNG (rank0 noise alignment)
        self._num_timesteps = len(self._timesteps)
        self._allocate_step_state()
        self._video_buffer = None  # NOT owned here
        # No VAE decode warmup: the cluster never decodes (the non-dist consumer
        # does). The VAE is still loaded for init_session's encode/prime.
        self._session_ctx = {
            "prompt_embeds": None, "negative_prompt_embeds": None,
            "first_image_condition": None, "first_image_pixels": None,
            "current_frame_idx": 0, "latent_chunk_idx": 0,
        }

    def prepare_latents(self, dtype=None, generator=None, latents=None):
        # base.prepare_latents broadcasts self._latents across the world (an NCCL
        # COLLECTIVE). In the cluster that collective deadlocks when it follows
        # the ring p2p ops on the same NCCL comm. self._latents is UNUSED by
        # LiveAvatarPipeline.step (it draws its own current_chunk_latents), so we
        # do the local randn only (no broadcast) -> the cluster stays pure-p2p.
        # The randn still consumes RNG identically, preserving rank0 noise
        # alignment with the reference.
        from diffusers.utils.torch_utils import randn_tensor
        shape = (1, self.cfg.dit_config.out_channels, self.cfg.max_num_actions,
                 self.cfg.latent_height, self.cfg.latent_width)
        return randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)

    @torch.inference_mode()
    def reset(self):
        # base.reset() touches self._video_buffer (None here) and calls
        # global_barrier (a collective); replicate it minus both (the cluster is
        # pure-p2p; per-rank init is independent + deterministic).
        self._session_ctx["latent_chunk_idx"] = 0
        self._session_ctx["current_frame_idx"] = 0
        self._session_ctx["first_image_condition"] = None
        self._session_ctx["first_image_pixels"] = None
        self._latents = self.prepare_latents(torch.float32)
        if self.vae_runner is not None:
            self.vae_runner.clear()
        self._on_reset()


def shm_names(prefix):
    # audio features + control (client->rank0); FINAL LATENTS (last rank->client
    # latents queue). The VAE decode is done by the (non-dist) consumer, NOT on
    # the cluster: the eager causal-VAE cudnn benchmark stalls for many minutes on
    # a rank that is inside the torch.distributed world (NCCL watchdog threads
    # contend with the benchmark), whereas the SAME VAE runs fast in a plain
    # process. So the cluster is 4 pure DiT-step ranks; the consumer decodes.
    return dict(
        audio=f"{prefix}_ppaudio",
        ctrl=f"{prefix}_ppctrl",
        latents=f"{prefix}_pplatents",
        ref=f"{prefix}_ppref",
    )


def run_rank(rank: int, world_size: int, cfg_path: str, prefix: str):
    set_torch_options()
    # Deterministic conv algos: cudnn.benchmark=True picks a per-GPU-fastest algo,
    # so step k on GPU k and the in-process oracle on GPU 0 can choose DIFFERENT
    # algos -> small per-forward numeric drift that the 4 steps + VAE amplify into
    # a visible video diff (gen-parity fails). Disabling benchmark uses the same
    # default algo on every H200, making the distributed result match the oracle.
    torch.backends.cudnn.benchmark = False
    cfg = RTConfig.from_yaml(cfg_path, is_path=True)
    n_steps = int(cfg.num_inference_steps)
    is_last = (rank == world_size - 1)
    assert n_steps % world_size == 0, \
        f"pp_steps needs num_inference_steps ({n_steps}) divisible by world_size ({world_size})"
    spr = n_steps // world_size                              # steps THIS rank runs
    my_steps = list(range(rank * spr, rank * spr + spr))     # global step indices

    device = torch.device("cuda:0")  # each rank: its own CVD -> cuda:0
    torch.cuda.set_device(device)

    # ---- distributed world (pure DiT pipeline, ring p2p) ----
    init_distributed_environment(world_size=world_size, rank=rank,
                                 local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=1)
    grp = get_world_group()
    prev = (rank - 1) % world_size
    nxt = (rank + 1) % world_size

    # ---- model (single cache for this step) ----
    pipe = SingleCachePipeline(cfg=cfg, device=device)
    pipe.start_instance()

    # fixed payload shapes
    gen = int(cfg.first_chunk_size)
    C = int(cfg.dit_config.out_channels)
    lat_shape = torch.Size((1, C, gen, int(cfg.latent_height), int(cfg.latent_width)))
    ref_shape = torch.Size((1, C, 1, int(cfg.latent_height), int(cfg.latent_width)))
    # audio feature shape [1, n_wav2vec_layers, 1024, step_frames]
    step_frames = int(cfg.chunk_size) * int(cfg.vae_config.scale_factor_temporal)
    af_layers = 25  # wav2vec2-large hidden_states (embed + 24 layers)
    af_shape = torch.Size((1, af_layers, 1024, step_frames))
    dtype = pipe.dtype

    ctx = LAContext(
        cfg=cfg, dit_runner=pipe.dit_runner, vae_runner=pipe.vae_runner,
        timesteps=pipe._timesteps, sigmas=pipe._sigmas, device=device, dtype=dtype,
        cond_prefix_tokens=pipe._cond_prefix_tokens,
        motion_frames_raw=pipe._motion_frames_raw,
        motion_frames_latent=pipe._motion_frames_latent,
    )
    dit_ops = [DiTDenoiseStep(s) for s in my_steps]   # this rank's steps, in order
    # NOTE: VAE decode is NOT run on the cluster (see shm_names docstring); the
    # last rank ships final latents and the non-dist consumer decodes.

    names = shm_names(prefix)
    if rank == 0:
        audio_buf = SharedTensorBuffer(names["audio"], frame_shape=tuple(af_shape[1:]),
                                       dtype=np.float32, max_len=8192, create=False)
        ctrl_buf = SharedControlBuffer(names["ctrl"], create=False)
        audio_read = 0
    if is_last:
        # ship FINAL LATENTS into the client-owned latents queue; the (non-dist)
        # consumer runs the VAE decode. One latent-frame group per chunk.
        latents_buf = SharedTensorBuffer(names["latents"],
                                         frame_shape=(C, gen, int(cfg.latent_height), int(cfg.latent_width)),
                                         dtype=np.float32, max_len=8192,
                                         create=False)
    # updated ref_latents (chunk0's first latent) is published by the last rank
    # via SHARED MEMORY, not a dist back-edge. A dist 3->0 edge would close the
    # ring 0->1->2->3->0 into a cycle and deadlock the (un-grouped) pynccl p2p;
    # shm keeps the dist p2p purely linear. Small (135 KB), one write per session.
    ref_buf = SharedTensorBuffer(names["ref"],
                                 frame_shape=(C, 1, int(cfg.latent_height), int(cfg.latent_width)),
                                 dtype=np.float32, max_len=256, create=False)
    ref_read = 0

    def read_ref_blocking():
        nonlocal ref_read
        while True:
            nxt_idx, arr = ref_buf.read(ref_read, 1)
            if arr is not None:
                ref_read = nxt_idx
                return torch.from_numpy(arr[0]).unsqueeze(0).to(device=device, dtype=dtype)
            time.sleep(0.002)

    def make_state():
        # one cache per global step this rank owns (my_steps[i] -> local cache i)
        return SimpleState({
            **{f"cache_{my_steps[i]}": pipe._step_kv_caches[i] for i in range(spr)},
            "ref_latents": pipe._ref_latents,
            "motion_latents": pipe._motion_latents,
        })

    def run_my_steps(latents, audio):
        # run this rank's denoising steps in order (the within-rank latent chain)
        for op in dit_ops:
            latents = op.execute({"latents": latents, "audio_features": audio},
                                 ctx, state)["latents"]
        return latents

    def init_session():
        set_global_seed(cfg.seed)
        pipe.init_session(prompt=cfg.prompt, negative_prompt=None, image_path=cfg.image_path)

    state = None
    chunk_idx = 0

    def gen_noise():
        return torch.randn((1, C, ctx.generate_latent_num, int(cfg.latent_height),
                            int(cfg.latent_width)), device=device, dtype=torch.float32).to(dtype)

    def send_ctrl(op):
        grp.send(torch.tensor([op], dtype=torch.int64, device=device), dst=nxt)

    def recv_ctrl():
        return int(grp.recv(torch.Size((1,)), torch.int64, src=prev).item())

    print(f"[pp rank {rank}/{world_size}] ready (steps {my_steps}, last={is_last}) pid={os.getpid()}", flush=True)

    # ============================ rank 0 (head) ============================
    if rank == 0:
        ref_for_next = None
        last_seq = 0
        terminate = False
        while True:
            # control value is (seq << 3) | op so consecutive same-ops are distinct
            v = int(ctrl_buf.recv())
            seq, op = v >> 3, v & 0x7
            if seq != last_seq:                   # act once per control transition
                last_seq = seq
                if op == OP_TERM:
                    send_ctrl(OP_TERM)
                    ctrl_buf.commit(); terminate = True
                elif op in (OP_INIT, OP_RESET):
                    send_ctrl(OP_INIT)
                    init_session(); state = make_state(); chunk_idx = 0; ref_for_next = None
                    print(f"[pp rank 0] init_session done (seq={seq})", flush=True)
                    ctrl_buf.commit()
                    continue
            if terminate:
                break
            # poll for a new audio-feature chunk
            audio_read, af = audio_buf.read(audio_read, 1)
            if af is None:
                continue
            audio = torch.from_numpy(af[0]).unsqueeze(0).to(device=device, dtype=dtype)
            ctx.prepare_chunk(chunk_idx)
            noise = gen_noise()
            if chunk_idx == 1:                      # updated ref via shm (blocks for last rank's chunk0)
                state.set("ref_latents", read_ref_blocking())
            latents = run_my_steps(noise, audio)
            send_ctrl(OP_DATA)                      # purely-linear ring: always plain DATA
            grp.send(latents.contiguous(), dst=nxt)
            grp.send(audio.contiguous(), dst=nxt)
            print(f"[pp r0] sent chunk {chunk_idx}", flush=True)
            chunk_idx += 1
        return

    # ========================= ranks 1..last (tail) =========================
    while True:
        op = recv_ctrl()
        if op == OP_TERM:
            if not is_last:
                send_ctrl(OP_TERM)
            break
        if op in (OP_INIT, OP_RESET):
            if not is_last:
                send_ctrl(OP_INIT)
            init_session(); state = make_state(); chunk_idx = 0
            print(f"[pp rank {rank}] init_session done", flush=True)
            continue
        latents = grp.recv(lat_shape, dtype, src=prev)
        audio = grp.recv(af_shape, dtype, src=prev)
        ctx.prepare_chunk(chunk_idx)
        if chunk_idx == 1:                          # updated ref via shm (published by last rank)
            state.set("ref_latents", read_ref_blocking())
        latents_out = run_my_steps(latents, audio)
        if not is_last:
            send_ctrl(OP_DATA)
            grp.send(latents_out.contiguous(), dst=nxt)
            grp.send(audio.contiguous(), dst=nxt)
            print(f"[pp r{rank}] done chunk {chunk_idx}", flush=True)
        else:
            # last rank: publish chunk0's updated ref to shm (so the other ranks
            # unblock ASAP), then ship the FINAL LATENTS to the client's latents
            # queue. The .cpu() here syncs only the (already-needed) DiT step
            # output -- a ~7MB D2H copy, NOT the multi-minute eager VAE benchmark
            # that stalled when the decode ran inside the dist world.
            if chunk_idx == 0:
                ref_buf.write(latents_out[:, :, :1].detach()[0].float().cpu().numpy())
            latents_buf.write(latents_out.detach()[0].float().cpu().numpy())
            print(f"[pp r{rank}] wrote latents chunk {chunk_idx} {tuple(latents_out.shape)}", flush=True)
        chunk_idx += 1


def main():
    rank = int(os.environ["PP_RANK"])
    world_size = int(os.environ["PP_WORLD"])
    cfg_path = os.environ["PP_CFG"]
    prefix = os.environ["PP_PREFIX"]
    run_rank(rank, world_size, cfg_path, prefix)


if __name__ == "__main__":
    main()
