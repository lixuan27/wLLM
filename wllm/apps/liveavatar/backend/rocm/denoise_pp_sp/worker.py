"""Variant `denoise_pp_sp` — L2+L3: denoise-step pipeline AND per-stage SP.

The obligated combination of the two non-dominated winners: denoise_pp4 (L2,
best throughput) and dit_sp3 (L3, best latency).

**Feasibility constraint (IR + hardware).** The DiT's Ulysses SP shards the frame
dim (`wllm/models/dit/liveavatar.py:601`, `sequence_model_parallel_shard(...,
dim=2)`), and `sequence_model_parallel_shard` is a plain `torch.chunk` with the
padding path commented out (`wllm/distributed/communication_op.py:113`), so the
sharded frame count must be divisible by sp. A chunk is `chunk_size=3` latent
frames, so **legal sp ∈ {1, 3}** (this is the same wall that killed dit_sp2:
`torch.chunk(3, 2) -> [2,1]` uneven -> Ulysses all_to_all shape mismatch ->
`camera_rope.py` `cos_sin_cache.shape==(S,D)` assert). A one-step-per-stage
pipeline (4 stages) with legal sp=3 would need 4*3=12 GPUs > 8. The feasible legal
form is a **2-stage pipeline (2 denoise steps per stage) x sp=3 = 6 GPUs** (DiT on
cuda:0-5, LLM/TTS on cuda:6,7). This is what the defaults below select.

Parametrised by `DENOISE_PP_STAGES` (default 2) and `DENOISE_PP_SP` (default 3);
`steps_per_stage = num_steps // n_stages`. Running `DENOISE_PP_STAGES=4
DENOISE_PP_SP=2` reproduces the illegal-shard crash on disk (kept intentionally so
the infeasibility of the 4x2 form is reproducible, not merely asserted).

2D dataflow: `world = n_stages*sp`. `initialize_model_parallel(sp_size=sp)` makes
the SP groups `{0..sp-1},{sp..2sp-1},...` = exactly the stage groups. Inter-stage
transfer happens only between each stage's sp_rank-0 (rank s*sp -> (s+1)*sp);
intra-stage the sp_rank-0 SP-group-broadcasts the (latent, audio) so all sp ranks
of the stage run the stage's steps in lockstep. Rank 0 (stage 0, sp_rank 0) is the
driver: ASR/LLM/TTS + wav2vec + VAE + I/O, feeds stage 0, drains the last stage
(async depth-`n_stages` pipeline, like denoise_pp). Streams per chunk.

Reuses run_denoise_step + init_dist + DriverPipeline + control protocol from
denoise_pp, and the SP cond-prefill/audio patches from dit_sp.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

from wllm.serving.logger import init_logger
from wllm.apps.liveavatar.reference.config import LiveAvatarReferenceConfig
from wllm.apps.liveavatar.reference.worker import LiveAvatarWorker
from wllm.apps.liveavatar.backend.rocm.ir.ops import LAContext
from wllm.apps.liveavatar.backend.rocm.denoise_pp.worker import (
    run_denoise_step, DriverPipeline, _timestep_schedule, _audio_shape, _latent_shape,
    CTRL_STOP, CTRL_INIT_SESSION, CTRL_RUN_CHUNKS, CTRL_RESET, _DIST_VARS, init_dist,
)
from wllm.apps.liveavatar.backend.rocm.dit_sp.worker import _apply_sp_patches  # applies SP DiT fixes
from wllm.apps.liveavatar.backend.rocm.runtime_common import free_port

logger = init_logger(__name__)
_apply_sp_patches()


def _ctx(cfg, dit, vae, device):
    ts, sg = _timestep_schedule(cfg, device)
    return LAContext(
        cfg=cfg, dit_runner=dit, vae_runner=vae, timesteps=ts, sigmas=sg, device=device,
        dtype=torch.bfloat16, cond_prefix_tokens=int(getattr(cfg, "kv_cond_tokens", 0) or 0),
        motion_frames_raw=int(getattr(cfg, "motion_prefix_frames", 0) or 0),
        motion_frames_latent=int(getattr(cfg, "motion_prefix_latent_frames", 0) or 0),
        num_inference_steps=int(cfg.num_inference_steps))


def _topology(cfg):
    """(n_stages, sp, steps_per_stage) from env, validated against num_steps."""
    n_steps = int(cfg.num_inference_steps)
    n_stages = int(os.environ.get("DENOISE_PP_STAGES", "2"))
    sp = int(os.environ.get("DENOISE_PP_SP", "3"))
    assert n_steps % n_stages == 0, f"num_steps {n_steps} not divisible by n_stages {n_stages}"
    return n_stages, sp, n_steps // n_stages


class PPSPStepRank:
    """A non-driver rank at (stage=rank//sp, sp_rank=rank%sp). Runs its stage's
    `steps_per_stage` consecutive denoise steps, sp-sharded, in lockstep."""

    def __init__(self, cfg_path, rank, world, sp, n_stages):
        from wllm.serving.runner.dit_runner import DiTRunner
        from wllm.serving.memory.preallocated_cache import PreAllocatedKVCache
        from wllm.serving.utils.torch_utils import set_torch_options
        set_torch_options()
        from wllm.serving.distributed.parallel_state import get_sp_group

        self.rank, self.world, self.sp, self.n_stages = rank, world, sp, n_stages
        self.stage = rank // sp
        self.sp_rank = rank % sp
        self.cfg = LiveAvatarReferenceConfig.from_yaml(cfg_path).to_runtime_config()
        self.n_steps = int(self.cfg.num_inference_steps)
        self.steps_per_stage = self.n_steps // n_stages
        self.step_lo = self.stage * self.steps_per_stage           # first global step of my stage
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16
        self.wg = init_dist(rank, world, sp_size=sp)
        self.pg = self.wg.device_group
        self.spg = get_sp_group()

        self.dit = DiTRunner(self.cfg, self.dtype, self.device)
        # one KV cache per step this stage owns
        self.caches = [PreAllocatedKVCache(self.cfg, self.device) for _ in range(self.steps_per_stage)]
        self.ctx = _ctx(self.cfg, self.dit, None, self.device)
        self.ref = None
        self.motion = None
        self.counter = 0
        logger.info("[pp_sp rank %d] stage %d sp_rank %d steps %d..%d on %s", rank, self.stage,
                    self.sp_rank, self.step_lo, self.step_lo + self.steps_per_stage - 1, self.device)

    def _recv_ctrl(self):
        c = torch.zeros(1, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(c, src=0, group=self.pg)
        return int(c.item())

    def _init_session(self):
        pe = torch.empty((1, int(self.cfg.max_sequence_length), int(self.cfg.dit_config.text_dim)),
                         dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(pe, src=0, group=self.pg)
        for c in self.caches:                     # prompt-encode each owned step's cache
            self.dit.kv_memory = c
            self.dit.encode(pe)
        z = int(self.cfg.vae_config.z_dim)
        self.ref = torch.empty((1, z, 1, int(self.cfg.latent_height), int(self.cfg.latent_width)),
                               dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(self.ref, src=0, group=self.pg)
        self.motion = torch.empty((1, z, int(self.cfg.motion_prefix_latent_frames),
                                   int(self.cfg.latent_height), int(self.cfg.latent_width)),
                                  dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(self.motion, src=0, group=self.pg)
        self.counter = 0

    def _run_stage_steps(self, latent, audio, ci):
        new_ref = None
        for j in range(self.steps_per_stage):
            gstep = self.step_lo + j
            latent, nr = run_denoise_step(self.dit, self.caches[j], self.ctx, gstep, self.n_steps,
                                          latent, audio, ci, self.ref, self.motion)
            if nr is not None:
                new_ref = nr
        return latent, new_ref

    def _run_chunks(self):
        n = torch.zeros(1, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(n, src=0, group=self.pg)
        m = int(n.item())
        prev0 = (self.stage - 1) * self.sp     # sp_rank-0 of the previous stage
        nxt0 = (self.stage + 1) * self.sp      # sp_rank-0 of the next stage
        last0 = (self.n_stages - 1) * self.sp  # sp_rank-0 of the last stage
        for _ in range(m):
            ci = self.counter
            latent = torch.empty(_latent_shape(self.cfg, ci), dtype=self.dtype, device=self.device)
            audio = torch.empty(_audio_shape(self.cfg), dtype=self.dtype, device=self.device)
            if self.sp_rank == 0:
                # pull this chunk from the previous stage's sp_rank 0 (driver if stage 1)
                torch.distributed.recv(latent, src=prev0, group=self.pg)
                torch.distributed.recv(audio, src=prev0, group=self.pg)
            # fan (latent, audio) out to my stage's SP group so all sp ranks run in lockstep
            self.spg.broadcast(latent, src=0)
            self.spg.broadcast(audio, src=0)
            latent, new_ref = self._run_stage_steps(latent, audio, ci)
            if self.sp_rank == 0:
                if self.stage < self.n_stages - 1:
                    torch.distributed.send(latent.contiguous(), dst=nxt0, group=self.pg)
                    torch.distributed.send(audio.contiguous(), dst=nxt0, group=self.pg)
                else:
                    torch.distributed.send(latent.contiguous(), dst=0, group=self.pg)   # -> driver
            # ref sync after session chunk 0 (last stage's last step produced new_ref)
            if ci == 0:
                nr = new_ref if new_ref is not None else torch.empty_like(self.ref)
                torch.distributed.broadcast(nr, src=last0, group=self.pg)
                self.ref = nr
            self.counter += 1

    def serve(self):
        while True:
            code = self._recv_ctrl()
            if code == CTRL_STOP:
                break
            elif code == CTRL_INIT_SESSION:
                self._init_session()
            elif code == CTRL_RESET:
                self.counter = 0
            elif code == CTRL_RUN_CHUNKS:
                self._run_chunks()


class DenoisePPSPWorker(LiveAvatarWorker):
    # subclasses (e.g. stream_pp_sp) launch their own step-rank module by overriding this
    STEP_MODULE = "wllm.apps.liveavatar.backend.rocm.denoise_pp_sp.worker"

    def __init__(self, cfg_path):
        from wllm.serving.distributed.parallel_state import get_sp_group
        cfg = LiveAvatarReferenceConfig.from_yaml(cfg_path).to_runtime_config()
        self.n_stages, self.sp, self.steps_per_stage = _topology(cfg)
        self.world = self.n_stages * self.sp
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(free_port()))
        self._step_procs = []
        for k in range(1, self.world):
            env = os.environ.copy()
            env.update(RANK=str(k), LOCAL_RANK=str(k), WORLD_SIZE=str(self.world))
            self._step_procs.append(subprocess.Popen(
                [sys.executable, "-u", "-m", self.STEP_MODULE,
                 "--step-rank", "--rank", str(k), "--world", str(self.world),
                 "--sp", str(self.sp), "--stages", str(self.n_stages), "--config", cfg_path], env=env))
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE=str(self.world))
        torch.cuda.set_device(0)
        self.wg = init_dist(0, self.world, sp_size=self.sp)
        self.pg = self.wg.device_group
        self.spg = get_sp_group()
        for v in _DIST_VARS:
            os.environ.pop(v, None)
        self._ctx = None
        self._bcast_done = False
        super().__init__(cfg_path)

    def _create_pipeline(self):
        return DriverPipeline(cfg=self.cfg, device=self.device)

    def _ensure_ctx(self):
        if self._ctx is None:
            self._ctx = _ctx(self.cfg, self.pipe.dit_runner, self.pipe.vae_runner, self.device)
        return self._ctx

    def _bcast_ctrl(self, code):
        torch.distributed.broadcast(torch.tensor([code], dtype=torch.int64, device=self.device),
                                    src=0, group=self.pg)

    def _broadcast_session(self):
        self._bcast_ctrl(CTRL_INIT_SESSION)
        pe = self.pipe._session_ctx["prompt_embeds"].to(self.device, self.pipe.dtype).contiguous()
        torch.distributed.broadcast(pe, src=0, group=self.pg)
        torch.distributed.broadcast(self.pipe._ref_latents.to(self.device, self.pipe.dtype).contiguous(),
                                    src=0, group=self.pg)
        torch.distributed.broadcast(self.pipe._motion_latents.to(self.device, self.pipe.dtype).contiguous(),
                                    src=0, group=self.pg)
        self._bcast_done = True

    def warmup(self):
        from wllm.serving.utils.rand import set_global_seed
        self._ensure_ctx()
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=self.cfg.image_path)
        self._broadcast_session()
        n_warm = max(3, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 1)
        dummy = [np.zeros((int(self.cfg.tts_chunk_size),), dtype=np.float32) for _ in range(n_warm)]
        self._bcast_ctrl(CTRL_RUN_CHUNKS)
        torch.distributed.broadcast(torch.tensor([n_warm], dtype=torch.int64, device=self.device), src=0, group=self.pg)
        self._pipeline(dummy, write=False)
        self.pipe.reset()
        self._bcast_ctrl(CTRL_RESET)
        self._bcast_done = False

    def start(self):
        super().start()
        self._broadcast_session()

    def reset(self):
        self._bcast_ctrl(CTRL_RESET)
        super().reset()
        self._bcast_done = False

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
                pass

    def _draw_noise(self, ci):
        gs, ge = self._ctx.chunk_global_range(ci)
        return torch.randn((1, int(self.cfg.dit_config.out_channels), ge - gs,
                            int(self.cfg.latent_height), int(self.cfg.latent_width)),
                           device=self.device, dtype=torch.float32).to(self.pipe.dtype)

    def _run_liveavatar_reference(self, audio_samples):
        self._ensure_ctx()
        if not self._bcast_done:
            self._broadcast_session()
        step_samples = int(self.cfg.tts_chunk_size)
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
        torch.distributed.broadcast(torch.tensor([m], dtype=torch.int64, device=self.device), src=0, group=self.pg)
        self._pipeline(chunks, write=True)
        return None, None

    def _run_stage0(self, noise, audio, ci):
        """Driver runs stage 0's steps (sp-sharded with stage-0 step ranks)."""
        latent = noise
        new_ref = None
        for j in range(self.steps_per_stage):
            latent, nr = run_denoise_step(self.pipe.dit_runner, self.pipe._step_kv_caches[j], self._ctx,
                                          j, int(self.cfg.num_inference_steps), latent, audio, ci,
                                          self.pipe._ref_latents, self.pipe._motion_latents)
            if nr is not None:
                new_ref = nr
        return latent, new_ref

    def _pipeline(self, chunks, write=True):
        pipe = self.pipe
        ctx = self._ctx
        tgt = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
        last0 = (self.n_stages - 1) * self.sp    # sp_rank 0 of the last stage
        stage1_sp0 = self.sp                     # sp_rank 0 of stage 1 (driver feeds this)
        m = len(chunks)
        DEPTH = self.n_stages
        fed = drained = 0
        pending = {}
        while drained < m:
            if fed < m and (fed - drained) < DEPTH:
                ci = ctx.chunk_idx = pipe._session_ctx["latent_chunk_idx"]
                noise = self._draw_noise(ci)
                audio = self.extract_audio_features(chunks[fed], target_frames=tgt)
                # stage 0 = driver's SP group: fan out (noise,audio), run stage-0 steps in lockstep
                self.spg.broadcast(noise, src=0)
                self.spg.broadcast(audio, src=0)
                latent, new_ref = self._run_stage0(noise, audio, ci)
                # hand off to stage 1's sp_rank 0
                torch.distributed.send(latent.contiguous(), dst=stage1_sp0, group=self.pg)
                torch.distributed.send(audio.contiguous(), dst=stage1_sp0, group=self.pg)
                # async-recv the final latent back from the last stage's sp_rank 0
                buf = torch.empty(_latent_shape(self.cfg, ci), dtype=pipe.dtype, device=self.device)
                req = torch.distributed.irecv(buf, src=last0, group=self.pg)
                pending[ci] = (req, buf, fed)
                if ci == 0:   # ref produced by the last stage on chunk 0 -> sync to all ranks
                    nr = new_ref if new_ref is not None else torch.empty_like(pipe._ref_latents)
                    torch.distributed.broadcast(nr, src=last0, group=self.pg)
                    pipe._ref_latents = nr
                pipe._session_ctx["latent_chunk_idx"] = ci + 1
                fed += 1
            dci = min(pending) if pending else None
            if dci is not None and (pending[dci][0].is_completed() or (fed - drained) >= DEPTH or fed >= m):
                req, buf, fidx = pending.pop(dci)
                req.wait()
                self._decode(buf, chunks[fidx], write)
                drained += 1

    def _decode(self, latents, chunk_audio, write):
        frames = []
        for fi in range(int(latents.shape[2])):
            vi = self.pipe.vae_runner.run(latents[:, :, fi:fi + 1, :, :].clone(), is_first_chunk=False)
            frames.append(vi.repeat_interleave(2, dim=1)[0].cpu().numpy())
        if not write:
            return
        video = np.concatenate(frames, axis=0)
        af = self._frame_audio(np.asarray(chunk_audio, dtype=np.float32))
        n = min(video.shape[0], af.shape[0])
        if n > 0:
            self.pipe._video_buffer.write(video[:n])
            self.audio_output_buffer.write(af[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-rank", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=6)
    ap.add_argument("--sp", type=int, default=3)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--config", required=True)
    args, _ = ap.parse_known_args()
    if args.step_rank:
        PPSPStepRank(args.config, args.rank, args.world, args.sp, args.stages).serve()
    else:
        DenoisePPSPWorker(cfg_path=args.config).loop()


if __name__ == "__main__":
    main()
