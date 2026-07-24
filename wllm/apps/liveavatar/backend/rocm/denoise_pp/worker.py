"""Variant `denoise_pp4` — denoising-step pipeline across chunks (IR L2).

IR basis: the model graph has 4 disjoint `step_k_kv` state objects, one per
denoising step, and analyze_cross_chunk_dependencies shows denoise_step_k(chunk
N+1) is independent of denoise_step_j(chunk N) for j!=k. So each denoising step
can live on its own GPU and consecutive chunks pipeline through them: while chunk
N is at step 1, chunk N+1 is at step 0. Steady-state throughput becomes ~1 chunk
per max-step-time instead of sum-of-4-steps -> attacks the sustainable-rate
target.

Layout: `world_size` = num_inference_steps ranks (default 4), rank k on cuda:k
runs a FULL unsharded DiT (sp_size=1) that only ever executes denoising step k,
owning one step-k KV cache. Rank 0 is also the driver: it runs ASR/LLM/TTS
(black boxes, on their own GPUs via cfg gpu indices), wav2vec, the VAE, the shm
I/O and the VAD loop, and denoising step 0. Latents flow rank0->rank1->..->
rank(N-1)->rank0 by RCCL P2P (~0.4 MB bf16/chunk); the driver overlaps feeding
step 0 with draining step N-1 + VAE via async isend/irecv.

Correctness: reuses the validated IR op geometry (LAContext) for the per-step
DiT call, so each step is byte-identical to the reference's step loop; the only
change is which GPU runs which step. The one cross-chunk dependency the IR flags
as session-scoped (ref_latents, updated at end of session chunk 0) is handled by
an explicit broadcast after chunk 0 (a one-time bubble). Encoder (cross-attn) KV,
ref_latents and motion_latents are broadcast to all ranks at session init.

Vendored: none — builds on shared DiTRunner/VAERunner + the shared
parallel_state / world-group P2P (see repo-root AGENTS.md "Multi-GPU
coordination"). Black-box engines are created on rank 0 only, with the
torch.distributed env scrubbed first so their internal vLLM world does not
inherit rank 0's world (the env-var pitfall).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

from wllm.serving.logger import init_logger
from wllm.apps.liveavatar.reference.config import LiveAvatarReferenceConfig
from wllm.apps.liveavatar.reference.pipeline import LiveAvatarPipeline
from wllm.apps.liveavatar.reference.worker import LiveAvatarWorker
from wllm.apps.liveavatar.backend.rocm.ir.ops import LAContext
from wllm.apps.liveavatar.backend.rocm.runtime_common import free_port

logger = init_logger(__name__)

# control codes broadcast from rank 0 to the step ranks
CTRL_STOP = 0
CTRL_INIT_SESSION = 1
CTRL_RUN_CHUNKS = 2
CTRL_RESET = 3

_DIST_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "TORCHELASTIC_ERROR_FILE",
)


def init_dist(rank: int, world: int, sp_size: int = 1, timeout_min: int = 60):
    """Init the process group with a LONG timeout so a rank blocked in a
    collective (e.g. a step rank waiting for rank 0's slow first-launch warmup)
    does not trip the default 600 s NCCL watchdog. Then set up the shared
    parallel_state groups and return the world GroupCoordinator."""
    import datetime
    from wllm.serving.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, get_world_group,
    )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=world, rank=rank,
            timeout=datetime.timedelta(minutes=timeout_min))
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank,
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=sp_size)
    return get_world_group()


def _latent_shape(cfg, chunk_i: int) -> tuple:
    fcs, cs = int(cfg.first_chunk_size), int(cfg.chunk_size)
    gen = fcs if chunk_i == 0 else cs
    return (1, int(cfg.dit_config.out_channels), gen,
            int(cfg.latent_height), int(cfg.latent_width))


def _audio_shape(cfg) -> tuple:
    target = int(cfg.chunk_size) * int(cfg.vae_config.scale_factor_temporal)
    return (1, 25, 1024, target)


@torch.inference_mode()
def run_denoise_step(dit, cache, ctx: LAContext, step_idx: int, num_steps: int,
                     latents, audio, chunk_i, ref_latents, motion_latents):
    """One denoising step k (mirrors ops.DenoiseStep.execute). Returns latents,
    and (on session chunk 0 at the last step) the updated ref_latents."""
    dit.kv_memory = cache
    geo = ctx.cache_geometry(chunk_i)
    gen = geo["gen"]
    cp = int(ctx.cond_prefix_tokens)

    if chunk_i <= 1:  # need_cond_prefill
        prefill_t = torch.zeros((1,), device=ctx.device, dtype=ctx.timesteps.dtype)
        dit.run(latents=latents, timestep=prefill_t, is_cache=True,
                cache_start=0, cache_end=cp, rope_start=0, rope_end=cp,
                prefill_cond=True, cond_prefix_tokens=cp,
                ref_latents=ref_latents, motion_latents=motion_latents,
                motion_frames_raw=ctx.motion_frames_raw,
                motion_frames_latent=ctx.motion_frames_latent)

    t_val = ctx.timesteps[step_idx]
    timestep = ctx.build_full_timestep(t_val, chunk_i)
    timestep_slice = timestep[:, -gen:].flatten()
    noise_pred = dit.run(latents=latents, timestep=timestep_slice, is_cache=False,
                         cache_start=geo["cache_start"], cache_end=geo["cache_end"],
                         rope_start=geo["rope_start"], rope_end=geo["rope_end"],
                         audio_input=audio, cond_latents=ctx.zero_cond(gen),
                         cond_prefix_tokens=cp,
                         motion_frames_raw=ctx.motion_frames_raw,
                         motion_frames_latent=ctx.motion_frames_latent)
    dt = ctx.sigmas[step_idx + 1] - ctx.sigmas[step_idx]
    latents = latents + dt * noise_pred
    new_ref = None
    if step_idx == num_steps - 1 and chunk_i == 0:
        new_ref = latents[:, :, :1].detach().clone()
    return latents, new_ref


# --------------------------------------------------------------------------- #
#  Step rank (ranks 1..N-1): full DiT for one step, serve loop.
# --------------------------------------------------------------------------- #
class DenoiseStepRank:
    def __init__(self, cfg_path: str, rank: int, world: int):
        from wllm.serving.runner.dit_runner import DiTRunner
        from wllm.serving.memory.preallocated_cache import PreAllocatedKVCache
        from wllm.serving.utils.torch_utils import set_torch_options
        set_torch_options()

        self.rank, self.world = rank, world
        # denoise_pp4: rank 0 does step 0, ranks 1..N-1 do steps 1..N-1 (step_idx=rank).
        # denoise_pp_driver (DENOISE_PP_DRIVER_ONLY=1): rank 0 is a pure driver,
        # ranks 1..N-1 do steps 0..N-2 (step_idx=rank-1), N=num_steps+1.
        self.driver_only = os.environ.get("DENOISE_PP_DRIVER_ONLY", "0") == "1"
        self.step_idx = (rank - 1) if self.driver_only else rank
        self.cfg = LiveAvatarReferenceConfig.from_yaml(cfg_path).to_runtime_config()
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16

        self.wg = init_dist(rank, world, sp_size=1)
        self.pg = self.wg.device_group

        self.dit = DiTRunner(self.cfg, self.dtype, self.device)
        self.cache = PreAllocatedKVCache(self.cfg, self.device)
        self.dit.kv_memory = self.cache

        self._build_ctx()
        self.ref_latents = None
        self.motion_latents = None
        self.chunk_counter = 0
        logger.info("[denoise_pp rank %d] step %d up on %s", rank, self.step_idx, self.device)

    def _build_ctx(self):
        scheduler = _timestep_schedule(self.cfg, self.device)
        self.ctx = LAContext(
            cfg=self.cfg, dit_runner=self.dit, vae_runner=None,
            timesteps=scheduler[0], sigmas=scheduler[1], device=self.device, dtype=self.dtype,
            cond_prefix_tokens=int(getattr(self.cfg, "kv_cond_tokens", 0) or 0),
            motion_frames_raw=int(getattr(self.cfg, "motion_prefix_frames", 0) or 0),
            motion_frames_latent=int(getattr(self.cfg, "motion_prefix_latent_frames", 0) or 0),
            num_inference_steps=int(self.cfg.num_inference_steps))

    def _recv_ctrl(self) -> int:
        code = torch.zeros(1, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(code, src=0, group=self.pg)
        return int(code.item())

    def _init_session(self):
        # receive prompt_embeds, ref_latents, motion_latents (broadcast from rank 0)
        pe = torch.empty((1, int(self.cfg.max_sequence_length), int(self.cfg.dit_config.text_dim)),
                         dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(pe, src=0, group=self.pg)
        # fill encoder (cross-attn) KV for this step's cache
        self.dit.kv_memory = self.cache
        self.dit.encode(pe)
        z = int(self.cfg.vae_config.z_dim)
        rl = torch.empty((1, z, 1, int(self.cfg.latent_height), int(self.cfg.latent_width)),
                         dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(rl, src=0, group=self.pg)
        ml = torch.empty((1, z, int(self.cfg.motion_prefix_latent_frames),
                          int(self.cfg.latent_height), int(self.cfg.latent_width)),
                         dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(ml, src=0, group=self.pg)
        self.ref_latents, self.motion_latents = rl, ml
        self.chunk_counter = 0

    def _run_chunks(self):
        n = torch.zeros(1, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(n, src=0, group=self.pg)
        m = int(n.item())
        prev = self.rank - 1
        nxt = (self.rank + 1) % self.world  # last step sends back to rank 0
        for _ in range(m):
            ci = self.chunk_counter
            shp = _latent_shape(self.cfg, ci)
            latent = torch.empty(shp, dtype=self.dtype, device=self.device)
            audio = torch.empty(_audio_shape(self.cfg), dtype=self.dtype, device=self.device)
            torch.distributed.recv(latent, src=prev, group=self.pg)
            torch.distributed.recv(audio, src=prev, group=self.pg)
            latent, new_ref = run_denoise_step(
                self.dit, self.cache, self.ctx, self.step_idx,
                int(self.cfg.num_inference_steps), latent, audio, ci,
                self.ref_latents, self.motion_latents)
            torch.distributed.send(latent.contiguous(), dst=nxt, group=self.pg)
            if self.rank < self.world - 1:  # last rank sends only latent back to driver
                torch.distributed.send(audio.contiguous(), dst=nxt, group=self.pg)
            # ref_latents sync after session chunk 0
            if ci == 0:
                if new_ref is None:
                    new_ref = torch.empty_like(self.ref_latents)
                torch.distributed.broadcast(new_ref, src=self.world - 1, group=self.pg)
                self.ref_latents = new_ref
            self.chunk_counter += 1

    @torch.inference_mode()
    def serve(self):
        while True:
            code = self._recv_ctrl()
            if code == CTRL_STOP:
                break
            elif code == CTRL_INIT_SESSION:
                self._init_session()
            elif code == CTRL_RESET:
                self.chunk_counter = 0
            elif code == CTRL_RUN_CHUNKS:
                self._run_chunks()
        logger.info("[denoise_pp rank %d] stopping", self.rank)


def _timestep_schedule(cfg, device):
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    s = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    s.set_timesteps(cfg.num_inference_steps)
    return s.timesteps.to(device), s.sigmas.to(device)


# --------------------------------------------------------------------------- #
#  Driver pipeline (rank 0): full LiveAvatarPipeline but without the world
#  collectives (rank 0 distributes latents itself via P2P).
# --------------------------------------------------------------------------- #
class DriverPipeline(LiveAvatarPipeline):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        # The Wan VAE tile-parallelizes its decode when parallel_state.get_world_size()
        # > 1 (a WORLD-group all_gather in vae_plan.gather_tile). But in these variants
        # only rank 0 runs the VAE, so that collective would deadlock. Force single-GPU
        # decode (world_size=1) — identical to the reference's single-GPU VAE output.
        for m in self.vae_runner.vae.modules():
            if hasattr(m, "world_size"):
                m.world_size = 1

    def prepare_latents(self, dtype=None, generator=None, latents=None):
        shape = (1, self.cfg.dit_config.out_channels, self.cfg.max_num_actions,
                 self.cfg.latent_height, self.cfg.latent_width)
        from diffusers.utils.torch_utils import randn_tensor
        return randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)

    def reset(self):
        # same as BasePipeline.reset but without global_barrier()
        self._session_ctx["latent_chunk_idx"] = 0
        self._session_ctx["current_frame_idx"] = 0
        self._session_ctx["first_image_condition"] = None
        self._session_ctx["first_image_pixels"] = None
        self._latents = self.prepare_latents(torch.float32)
        if self.vae_runner is not None:
            self.vae_runner.clear()
        from wllm.serving.distributed.parallel_state import get_world_rank
        if get_world_rank() == 0:
            self._video_buffer.clear()
        self._on_reset()


# --------------------------------------------------------------------------- #
#  Rank 0 driver worker.
# --------------------------------------------------------------------------- #
class DenoisePipelineWorker(LiveAvatarWorker):
    def __init__(self, cfg_path: str):
        self._driver_only = os.environ.get("DENOISE_PP_DRIVER_ONLY", "0") == "1"
        self.world = int(os.environ.get("DENOISE_PP_WORLD", "5" if self._driver_only else "4"))
        self._cfg_path = cfg_path
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(free_port()))
        self._step_procs = self._spawn_step_ranks(cfg_path)

        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE=str(self.world))
        torch.cuda.set_device(0)
        self.wg = init_dist(0, self.world, sp_size=1)
        self.pg = self.wg.device_group
        # scrub dist env so the vLLM/Omni engines started in super().__init__ get a clean world
        for v in _DIST_VARS:
            os.environ.pop(v, None)

        self._pp_ctx = None
        self._session_broadcast_done = False
        super().__init__(cfg_path)  # creates DriverPipeline (step0) + black boxes + (distributed) warmup

    def _ensure_pp_ctx(self):
        if self._pp_ctx is None:
            self._pp_ctx = LAContext(
                cfg=self.cfg, dit_runner=self.pipe.dit_runner, vae_runner=self.pipe.vae_runner,
                timesteps=self.pipe._timesteps, sigmas=self.pipe._sigmas, device=self.device,
                dtype=self.pipe.dtype,
                cond_prefix_tokens=int(getattr(self.cfg, "kv_cond_tokens", 0) or 0),
                motion_frames_raw=int(getattr(self.cfg, "motion_prefix_frames", 0) or 0),
                motion_frames_latent=int(getattr(self.cfg, "motion_prefix_latent_frames", 0) or 0),
                num_inference_steps=int(self.cfg.num_inference_steps))
        return self._pp_ctx

    def _spawn_step_ranks(self, cfg_path):
        procs = []
        for k in range(1, self.world):
            env = os.environ.copy()
            env.update(RANK=str(k), LOCAL_RANK=str(k), WORLD_SIZE=str(self.world))
            # NOTE: no setsid — keep the step ranks in rank 0's process group so
            # the harness's os.killpg(rank0) reaps them even on a hard kill.
            procs.append(subprocess.Popen(
                [sys.executable, "-u", "-m", "wllm.apps.liveavatar.backend.rocm.denoise_pp.worker",
                 "--step-rank", "--rank", str(k), "--world", str(self.world),
                 "--config", cfg_path], env=env))
        return procs

    def _create_pipeline(self):
        return DriverPipeline(cfg=self.cfg, device=self.device)

    def warmup(self):
        """Distributed warmup: prime the session and run a few dummy chunks
        through the 4-GPU pipeline so ALL ranks warm their DiT forwards together
        (no rank sits idle in a collective long enough to trip the watchdog, and
        round 0 of the real benchmark is already warm)."""
        from wllm.serving.utils.rand import set_global_seed
        self._ensure_pp_ctx()
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None,
                               image_path=self.cfg.image_path)
        self._broadcast_session()
        n_warm = max(3, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 1)
        dummy = [np.zeros((int(self.cfg.tts_chunk_size),), dtype=np.float32) for _ in range(n_warm)]
        self._bcast_ctrl(CTRL_RUN_CHUNKS)
        torch.distributed.broadcast(torch.tensor([n_warm], dtype=torch.int64, device=self.device),
                                    src=0, group=self.pg)
        self._pipeline_chunks(dummy, write=False)
        self.pipe.reset()
        self._bcast_ctrl(CTRL_RESET)
        self._session_broadcast_done = False

    def _bcast_ctrl(self, code: int):
        t = torch.tensor([code], dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(t, src=0, group=self.pg)

    def _broadcast_session(self):
        self._bcast_ctrl(CTRL_INIT_SESSION)
        pe = self.pipe._session_ctx["prompt_embeds"].to(self.device, self.pipe.dtype).contiguous()
        torch.distributed.broadcast(pe, src=0, group=self.pg)
        rl = self.pipe._ref_latents.to(self.device, self.pipe.dtype).contiguous()
        torch.distributed.broadcast(rl, src=0, group=self.pg)
        ml = self.pipe._motion_latents.to(self.device, self.pipe.dtype).contiguous()
        torch.distributed.broadcast(ml, src=0, group=self.pg)
        self._session_broadcast_done = True

    def start(self):
        super().start()  # sets seed, init_session (primes ref/motion + encoder KV on rank0)
        self._broadcast_session()

    def reset(self):
        self._bcast_ctrl(CTRL_RESET)
        super().reset()
        self._session_broadcast_done = False

    def terminate(self):
        try:
            self._bcast_ctrl(CTRL_STOP)
        except Exception:
            pass
        super().terminate()
        for p in self._step_procs:
            try:
                p.terminate(); p.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), 9)
                except Exception:
                    pass

    def _run_liveavatar_reference(self, audio_samples: np.ndarray):
        """Feed the response's chunks through the 4-GPU denoise pipeline and
        stream each decoded chunk's frames out as produced."""
        self._ensure_pp_ctx()
        if not self._session_broadcast_done:
            self._broadcast_session()
        frames_per_latent = int(self.cfg.vae_config.scale_factor_temporal)
        step_frames = int(self.cfg.chunk_size) * frames_per_latent
        step_samples = int(self.cfg.tts_chunk_size)

        # split audio into 480 ms chunks
        chunks = []
        off = 0
        while off < audio_samples.size or not chunks:
            ca = audio_samples[off:off + step_samples]
            if ca.size == 0 and chunks:
                break
            if ca.size < step_samples:
                ca = np.pad(ca, (0, step_samples - ca.size), mode="constant")
            chunks.append(ca)
            off += step_samples
        m = len(chunks)

        self._bcast_ctrl(CTRL_RUN_CHUNKS)
        torch.distributed.broadcast(torch.tensor([m], dtype=torch.int64, device=self.device),
                                    src=0, group=self.pg)
        self._pipeline_chunks(chunks)
        return None, None

    @torch.inference_mode()
    def _pipeline_chunks(self, chunks, write=True):
        pipe = self.pipe
        ctx = self._pp_ctx
        n_steps = int(self.cfg.num_inference_steps)
        last_rank = self.world - 1
        m = len(chunks)
        lat_shape_for = lambda ci: _latent_shape(self.cfg, ci)
        DEPTH = self.world  # chunks in flight

        fed = 0
        drained = 0
        pending = {}  # chunk_idx -> (irecv_req, buffer)
        keepalive = []
        while drained < m:
            # feed
            if fed < m and (fed - drained) < DEPTH:
                ci = self._pp_ctx.chunk_idx = pipe._session_ctx["latent_chunk_idx"]
                # draw noise (chunk-ordered, from the same global RNG the reference
                # uses) + wav2vec features for this chunk
                noise = self._draw_noise(ci)
                audio = self.extract_audio_features(chunks[fed], target_frames=int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal))
                if self._driver_only:
                    # pure driver: rank 1 runs step 0 -> just forward the noise
                    latent, new_ref = noise, None
                else:
                    # rank 0 runs step 0 itself
                    latent, new_ref = run_denoise_step(
                        pipe.dit_runner, pipe._step_kv_caches[0], ctx, 0, n_steps,
                        noise, audio, ci, pipe._ref_latents, pipe._motion_latents)
                # forward to rank 1
                torch.distributed.send(latent.contiguous(), dst=1, group=self.pg)
                torch.distributed.send(audio.contiguous(), dst=1, group=self.pg)
                # post irecv for the final latent of this chunk from the last rank
                buf = torch.empty(lat_shape_for(ci), dtype=pipe.dtype, device=self.device)
                req = torch.distributed.irecv(buf, src=last_rank, group=self.pg)
                pending[ci] = (req, buf, fed)
                keepalive.append(latent)
                # ref_latents sync after session chunk 0 (step0 already forwarded)
                if ci == 0:
                    nr = new_ref if new_ref is not None else torch.empty_like(pipe._ref_latents)
                    torch.distributed.broadcast(nr, src=last_rank, group=self.pg)
                    pipe._ref_latents = nr
                pipe._session_ctx["latent_chunk_idx"] = ci + 1
                fed += 1
            # drain (in order)
            drain_ci = min(pending) if pending else None
            if drain_ci is not None and (pending[drain_ci][0].is_completed() or (fed - drained) >= DEPTH or fed >= m):
                req, buf, fidx = pending.pop(drain_ci)
                req.wait()
                self._decode_and_emit(buf, chunks[fidx], write=write)
                drained += 1
                keepalive.clear()

    def _draw_noise(self, ci):
        gs, ge = self._pp_ctx.chunk_global_range(ci)
        gen = ge - gs
        return torch.randn((1, int(self.cfg.dit_config.out_channels), gen,
                            int(self.cfg.latent_height), int(self.cfg.latent_width)),
                           device=self.device, dtype=torch.float32).to(self.pipe.dtype)

    def _decode_and_emit(self, latents, chunk_audio, write=True):
        frames = []
        for fi in range(int(latents.shape[2])):
            li = latents[:, :, fi:fi + 1, :, :].clone()
            vi = self.pipe.vae_runner.run(li, is_first_chunk=False)
            vi = vi.repeat_interleave(2, dim=1)
            frames.append(vi[0].cpu().numpy())
        if not write:
            return
        video = np.concatenate(frames, axis=0)
        audio_frames = self._frame_audio(np.asarray(chunk_audio, dtype=np.float32))
        n = min(video.shape[0], audio_frames.shape[0])
        if n > 0:
            self.pipe._video_buffer.write(video[:n])
            self.audio_output_buffer.write(audio_frames[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-rank", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--config", required=True)
    args, _ = ap.parse_known_args()
    if args.step_rank:
        DenoiseStepRank(args.config, args.rank, args.world).serve()
    else:
        DenoisePipelineWorker(cfg_path=args.config).loop()


if __name__ == "__main__":
    main()
