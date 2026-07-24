"""5-stage cluster: 4 DiT denoising-step ranks + 1 dedicated VAE-decode rank.

This is the full 5-GPU pipeline parallelism the IR analysis surfaced
(`find_pipeline_stages` -> 6 stages: wav2vec, dit_step_0..3, vae_decode). pp_steps
realized only the 4 DiT stages (ranks 0..3) and ran the VAE OFF the cluster on the
worker GPU; here the causal VAE decode is promoted to a first-class **5th rank**
(rank = world_size-1). Per-chunk data flow:

    worker --shm(audio)--> rank0 -p2p-> rank1 -p2p-> rank2 -p2p-> rank3(last DiT)
                                                              -p2p-> rank4(VAE)
    rank4 --VAE decode--> shm(video) --> worker --> adapter output buffer

So the dist graph is still PURELY LINEAR (0->1->2->3->4, no back-edge; rank3's
chunk-0 ref_latents go to the DiT ranks over a shared-memory `ref` buffer, exactly
as in pp_steps). The worker no longer VAE-decodes -- it relays already-decoded
frames -- and latents reach the VAE over NVLink p2p instead of a host round-trip.

WHY THIS IS NOW LEGAL (pp_steps bug #6 revisited): pp_steps moved the VAE off the
cluster because, on the *old* node, the eager causal-VAE's cudnn work stalled for
minutes on a rank inside the NCCL world. That node lacked NVLink and ran the
cluster with GPU-to-GPU P2P disabled; the stall was tied to that configuration.
On NVLink-class nodes (P2P enabled) the VAE can live inside the world as its own
stage.

Correctness: every rank reuses the SAME Phase-2-validated IR ops
(`DiTDenoiseStep(k)`, `VAEDecode`) on the SAME data as the reference, so the
generated video is bit-exact with the reference by the same construction that made
pp_steps bit-exact. The ONLY change from pp_steps is *where*
the VAEDecode op runs (rank4 vs the worker) -- the op, its inputs, and the
continuous vae_cache evolution are identical.

STATUS: validated STATICALLY ONLY on this node -- GPU 3 is down, which fails every
multi-GPU launch, so the E2E correctness/benchmark run could not be executed here.
"""
from __future__ import annotations

import os
import sys
import time
import faulthandler
import signal

import numpy as np
import torch

# SIGUSR1 -> dump all threads' Python tracebacks to stderr (the rank log).
faulthandler.register(signal.SIGUSR1)


from wllm.serving.rt_config import RTConfig
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel, get_world_group,
)
from wllm.apps.liveavatar.backend.cuda.ir.ops import LAContext, DiTDenoiseStep, VAEDecode
# Reuse the proven pp_steps building blocks (single-cache pipeline + opcodes).
from wllm.apps.liveavatar.backend.cuda.pp_steps.cluster import (
    SimpleState, SingleCachePipeline,
    OP_NONE, OP_INIT, OP_DATA, OP_DATA_REF, OP_TERM, OP_RESET,
)
from wllm.apps.liveavatar.backend.cuda.runtime_common import _detach_shm_resource_tracker

# cluster processes ATTACH to shm created by the client; never unlink on exit.
_detach_shm_resource_tracker()


class DiTStepPipeline(SingleCachePipeline):
    """SingleCachePipeline whose number of denoising-step caches is derived from
    PP_NDIT (the count of DiT ranks), NOT the dist world size.

    In pp_steps the world *is* the DiT ranks, so the base class reads PP_WORLD to
    get steps-per-rank. Here the world also contains the VAE rank, so
    world (=5) != #DiT ranks (=4); using PP_WORLD would compute the wrong
    steps-per-rank. PP_NDIT (= world_size-1) is the correct divisor."""

    def _allocate_step_caches(self) -> None:
        if self.dit_runner is None:
            return
        n_dit = int(os.environ.get("PP_NDIT", int(self.cfg.num_inference_steps)))
        spr = int(self.cfg.num_inference_steps) // n_dit
        base = self.dit_runner.kv_memory
        cache_cls = base.__class__
        self._step_kv_caches = [base] + [cache_cls(self.cfg, self.device)
                                         for _ in range(1, spr)]
        self._activate_step_cache(0)


def shm_names(prefix):
    # audio features + control (worker->rank0); VIDEO (rank4 VAE -> worker); ref
    # (rank3 -> all DiT ranks, shm). Unlike pp_steps there is NO latents queue:
    # the VAE decode runs on rank4, so the worker receives decoded frames directly.
    return dict(
        audio=f"{prefix}_ppaudio",
        ctrl=f"{prefix}_ppctrl",
        video=f"{prefix}_ppvideo",
        ref=f"{prefix}_ppref",
    )


def run_rank(rank: int, world_size: int, cfg_path: str, prefix: str):
    set_torch_options()
    # Deterministic conv algos across GPUs (pp_steps bug #7): a per-GPU-fastest
    # cudnn algo would let step k on GPU k and the oracle on GPU0 diverge.
    torch.backends.cudnn.benchmark = False
    cfg = RTConfig.from_yaml(cfg_path, is_path=True)
    n_steps = int(cfg.num_inference_steps)
    n_dit = world_size - 1                       # the last rank is the VAE stage
    is_vae = (rank == world_size - 1)
    is_last_dit = (rank == n_dit - 1)            # rank that runs the final denoising step
    assert n_dit >= 1, "5-stage cluster needs >=2 ranks (>=1 DiT + 1 VAE)"
    assert n_steps % n_dit == 0, \
        f"need num_inference_steps ({n_steps}) divisible by #DiT ranks ({n_dit})"
    spr = n_steps // n_dit                        # denoising steps this DiT rank runs
    my_steps = [] if is_vae else list(range(rank * spr, rank * spr + spr))

    device = torch.device("cuda:0")              # each rank: its own CVD -> cuda:0
    torch.cuda.set_device(device)

    # ---- distributed world (linear ring p2p; tp=sp=1, manual get_world_group) ----
    init_distributed_environment(world_size=world_size, rank=rank,
                                 local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=1)
    grp = get_world_group()
    prev = (rank - 1) % world_size
    nxt = (rank + 1) % world_size

    # ---- model (DiT ranks: 1 cache for their step; VAE rank: uses vae_runner) ----
    pipe = DiTStepPipeline(cfg=cfg, device=device)
    pipe.start_instance()

    # The Wan VAE decoder auto-tiles its decode across the torch world: at build it
    # records `self.world_size = get_world_size()` (= this cluster's 5), and its
    # forward does `if self.world_size > 1: split_tile(...)`. vae_plan.split_tile
    # only supports world 2/3/4 -> with 5 ranks it raises "Unsupported world_size=5"
    # and ALL ranks crash during the init_session VAE prime (the original failure).
    # The VAE here runs STANDALONE on the VAE rank (full-image decode, identical to
    # the single-GPU reference), so pin the decoder to its non-distributed path on
    # EVERY rank — all ranks prime the VAE, and a world_size mismatch would deadlock
    # the tiling all_gather. world_size==1 also makes the decode bit-exact w/ the ref.
    if pipe.vae_runner is not None and getattr(pipe.vae_runner, "vae", None) is not None:
        pipe.vae_runner.vae.decoder.world_size = 1
        pipe.vae_runner.vae.decoder.rank = 0

    # fixed payload shapes (first_chunk_size == chunk_size, so latents are fixed)
    gen = int(cfg.first_chunk_size)
    C = int(cfg.dit_config.out_channels)
    lat_shape = torch.Size((1, C, gen, int(cfg.latent_height), int(cfg.latent_width)))
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

    names = shm_names(prefix)

    def send_ctrl(op):
        grp.send(torch.tensor([op], dtype=torch.int64, device=device), dst=nxt)

    def recv_ctrl():
        return int(grp.recv(torch.Size((1,)), torch.int64, src=prev).item())

    def init_session():
        set_global_seed(cfg.seed)
        pipe.init_session(prompt=cfg.prompt, negative_prompt=None, image_path=cfg.image_path)

    # ========================= VAE rank (terminal) =========================
    if is_vae:
        vae_op = VAEDecode()
        video_buf = SharedTensorBuffer(
            names["video"], frame_shape=(int(cfg.height), int(cfg.width), 3),
            dtype=np.uint8, max_len=int(cfg.max_num_frames), create=False)
        chunk_idx = 0
        print(f"[pp rank {rank}/{world_size}] ready (vae decode) pid={os.getpid()}", flush=True)
        while True:
            op = recv_ctrl()
            if op == OP_TERM:
                break                                   # terminal: do not forward
            if op in (OP_INIT, OP_RESET):
                init_session()                          # clears + re-primes vae_cache
                chunk_idx = 0
                print(f"[pp rank {rank}] init_session done (vae)", flush=True)
                continue
            # OP_DATA: recv this chunk's FINAL latents (+ audio, which the VAE does
            # not use but must be received to balance rank3's send), decode, ship.
            latents = grp.recv(lat_shape, dtype, src=prev)
            _audio = grp.recv(af_shape, dtype, src=prev)
            # The exact Phase-2 VAEDecode op + the continuous causal vae_cache; this
            # mirrors LiveAvatarPipeline.step's decode loop frame-by-frame.
            vstate = SimpleState({"vae_cache": pipe.vae_runner})
            video = vae_op.execute({"latents": latents}, ctx, vstate)["video"]  # (frames,H,W,3) uint8
            video_buf.write(np.ascontiguousarray(video))
            print(f"[pp r{rank}] decoded+wrote chunk {chunk_idx} {tuple(video.shape)}", flush=True)
            chunk_idx += 1
        return

    # ========================= DiT ranks (0..n_dit-1) =========================
    dit_ops = [DiTDenoiseStep(s) for s in my_steps]   # this rank's steps, in order
    if rank == 0:
        audio_buf = SharedTensorBuffer(names["audio"], frame_shape=tuple(af_shape[1:]),
                                       dtype=np.float32, max_len=8192, create=False)
        ctrl_buf = SharedControlBuffer(names["ctrl"], create=False)
        audio_read = 0
    # updated ref_latents (chunk0's first latent) is published by the LAST DiT rank
    # via shared memory and read by every DiT rank on its chunk 1. shm (not a dist
    # back-edge) keeps the p2p graph purely linear -- a dist last->0 edge would
    # close a cycle and deadlock the un-grouped pynccl p2p (pp_steps bug #4).
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
        return SimpleState({
            **{f"cache_{my_steps[i]}": pipe._step_kv_caches[i] for i in range(spr)},
            "ref_latents": pipe._ref_latents,
            "motion_latents": pipe._motion_latents,
        })

    def run_my_steps(latents, audio):
        for op in dit_ops:
            latents = op.execute({"latents": latents, "audio_features": audio},
                                 ctx, state)["latents"]
        return latents

    state = None
    chunk_idx = 0

    def gen_noise():
        return torch.randn((1, C, ctx.generate_latent_num, int(cfg.latent_height),
                            int(cfg.latent_width)), device=device, dtype=torch.float32).to(dtype)

    print(f"[pp rank {rank}/{world_size}] ready (steps {my_steps}, last_dit={is_last_dit}) "
          f"pid={os.getpid()}", flush=True)

    # ----- rank 0 (head): reads audio + control from the worker over shm -----
    if rank == 0:
        last_seq = 0
        terminate = False
        while True:
            v = int(ctrl_buf.recv())              # (seq << 3) | op
            seq, op = v >> 3, v & 0x7
            if seq != last_seq:                   # act once per control transition
                last_seq = seq
                if op == OP_TERM:
                    send_ctrl(OP_TERM)
                    ctrl_buf.commit(); terminate = True
                elif op in (OP_INIT, OP_RESET):
                    send_ctrl(OP_INIT)
                    init_session(); state = make_state(); chunk_idx = 0
                    print(f"[pp rank 0] init_session done (seq={seq})", flush=True)
                    ctrl_buf.commit()
                    continue
            if terminate:
                break
            audio_read, af = audio_buf.read(audio_read, 1)
            if af is None:
                continue
            audio = torch.from_numpy(af[0]).unsqueeze(0).to(device=device, dtype=dtype)
            ctx.prepare_chunk(chunk_idx)
            noise = gen_noise()
            if chunk_idx == 1:                    # one-time ref dependency (startup bubble)
                state.set("ref_latents", read_ref_blocking())
            latents = run_my_steps(noise, audio)
            send_ctrl(OP_DATA)
            grp.send(latents.contiguous(), dst=nxt)
            grp.send(audio.contiguous(), dst=nxt)
            print(f"[pp r0] sent chunk {chunk_idx}", flush=True)
            chunk_idx += 1
        return

    # ----- DiT tail ranks 1..n_dit-1 (rank n_dit-1 also publishes ref + feeds VAE) -----
    while True:
        op = recv_ctrl()
        if op == OP_TERM:
            send_ctrl(OP_TERM)                    # forward toward the VAE rank
            break
        if op in (OP_INIT, OP_RESET):
            send_ctrl(OP_INIT)
            init_session(); state = make_state(); chunk_idx = 0
            print(f"[pp rank {rank}] init_session done", flush=True)
            continue
        latents = grp.recv(lat_shape, dtype, src=prev)
        audio = grp.recv(af_shape, dtype, src=prev)
        ctx.prepare_chunk(chunk_idx)
        if chunk_idx == 1:
            state.set("ref_latents", read_ref_blocking())
        latents_out = run_my_steps(latents, audio)
        if is_last_dit and chunk_idx == 0:        # publish chunk0's updated ref for all DiT ranks
            ref_buf.write(latents_out[:, :, :1].detach()[0].float().cpu().numpy())
        # every DiT rank forwards downstream -- the last DiT rank feeds rank4 (VAE).
        send_ctrl(OP_DATA)
        grp.send(latents_out.contiguous(), dst=nxt)
        grp.send(audio.contiguous(), dst=nxt)
        print(f"[pp r{rank}] done chunk {chunk_idx} (last_dit={is_last_dit})", flush=True)
        chunk_idx += 1


def main():
    rank = int(os.environ["PP_RANK"])
    world_size = int(os.environ["PP_WORLD"])
    cfg_path = os.environ["PP_CFG"]
    prefix = os.environ["PP_PREFIX"]
    run_rank(rank, world_size, cfg_path, prefix)


if __name__ == "__main__":
    main()
