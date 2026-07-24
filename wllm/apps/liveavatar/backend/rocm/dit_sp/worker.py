"""Variant `dit_sp{N}` — within-chunk DiT sequence parallelism (IR L3, below-IR).

IR basis: not surfaced by the IR analysis tools (the atomic denoise-step operators
don't decompose along this axis); found by reading the DiT — its forward already
does Ulysses sequence parallelism (`sequence_model_parallel_shard` on the frame
dim + all-to-all in `WanRopeSelfAttention`), with heads (40) auto-padded to a
multiple of SP. Sharding one chunk's DiT work across N GPUs shortens each denoise
step's critical path -> attacks single-chunk latency (and, by shortening each
step, the sustainable rate). Streams per chunk like stream_liveavatar.

Structure (SPMD): `world_size` = sp_size ranks. All ranks run the SAME 4-step
denoise per chunk in lockstep; each `dit.run` shards the latents and all-gathers
the noise_pred, so every rank holds the synced full latents (euler is local, no
inter-step transfer, and ref_latents needs no broadcast). Rank 0 broadcasts the
per-chunk (noise, audio); all ranks run the steps; rank 0 does the VAE + write.
Rank 0 also runs ASR/LLM/TTS (black boxes, off the SP GPUs) + wav2vec + I/O.

Reuses the validated IR op geometry (run_denoise_step) + the multi-GPU launch /
control protocol from denoise_pp.
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
from wllm.apps.liveavatar.backend.rocm.runtime_common import free_port

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
#  Vendored SP fix (per repo-root AGENTS.md "Read-only areas are not a blocker —
#  vendor the file"): the shared LiveAvatarTransformer3DModel._prefill_condition_cache
#  builds the condition tokens but does NOT sequence-parallel-shard them, while the
#  generation forward DOES shard its frames. Under SP the attention's Ulysses
#  all_to_all then sees a full (unsharded) cond sequence and the rope kernel asserts
#  `cos_sin_cache.shape==(S,D)`. Fix: shard the cond tokens on dim=1 before the block
#  loop (cond_tokens=2640 is divisible by SP=3 -> 880 each; the all_to_all gathers back
#  to the full 2640 which matches the full cond_rotary). This is a NO-OP under sp=1
#  (sequence_model_parallel_shard returns the input unchanged), so it is safe and only
#  affects the dit_sp process. Rest of the method mirrors the original verbatim.
def _sp_prefill_condition_cache(self, hidden_states, timestep, forward_batch_info):
    import torch as _t
    from wllm.serving.distributed.communication_op import sequence_model_parallel_shard as _shard
    ref_latents = forward_batch_info.ref_latents
    if ref_latents is None:
        ref_latents = hidden_states[:, :, :1].detach().clone()
    ref_latents = ref_latents[:, :, :1].contiguous()
    motion_latents = forward_batch_info.motion_latents
    if motion_latents is None:
        motion_len = max(1, self._motion_latent_frames_default)
        motion_latents = ref_latents.repeat(1, 1, motion_len, 1, 1)
    motion_latents = self._normalize_motion_latents(motion_latents, ref_latents)
    cond_tokens, cond_rotary, ref_token_count, motion_token_count = self._build_condition_tokens(
        ref_latents, motion_latents)
    assert cond_rotary.shape[0] == cond_tokens.shape[1]
    cond_mask = _t.ones(cond_tokens.shape[0], cond_tokens.shape[1], dtype=_t.long,
                        device=cond_tokens.device)
    if cond_tokens.shape[1] > ref_token_count:
        cond_mask[:, ref_token_count:] = 2
    cond_tokens = cond_tokens + self.trainable_cond_mask(cond_mask).to(cond_tokens.dtype)
    expected_tokens = int(forward_batch_info.cache_end - forward_batch_info.cache_start)
    assert cond_tokens.shape[1] == expected_tokens
    zero_t = _t.zeros((cond_tokens.shape[0],), device=cond_tokens.device, dtype=timestep.dtype)
    _, timestep_proj = self.condition_embedder(zero_t)
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
    hidden = _shard(cond_tokens, dim=1)   # <-- SP fix (no-op under sp=1)
    for block in self.blocks:
        hidden = block(hidden, timestep_proj, cond_rotary, forward_batch_info)


#  Second SP-unaware path: _inject_audio reshapes the (SP-sharded) hidden by the
#  FULL audio's frame count, mixing this rank's frames with all audio frames. Fix:
#  shard the audio emb on the frame dim (dim=1) so num_frames matches the sharded
#  hidden. No-op under sp=1. Rest mirrors the original verbatim.
def _sp_inject_audio(self, block_idx, hidden_states, num_noisy_tokens, forward_batch_info):
    from wllm.serving.distributed.communication_op import sequence_model_parallel_shard as _shard
    if block_idx not in self.audio_injector.injected_block_id:
        return hidden_states
    attn_id = self.audio_injector.injected_block_id[block_idx]
    audio_emb = _shard(self._audio_local_emb, dim=1)   # <-- SP fix (no-op under sp=1)
    num_frames = audio_emb.shape[1]
    if num_frames <= 0:
        return hidden_states
    assert (num_noisy_tokens % num_frames) == 0
    vis = hidden_states[:, :num_noisy_tokens].clone()
    hw = num_noisy_tokens // num_frames
    vis = vis.reshape(vis.shape[0] * num_frames, hw, vis.shape[-1])
    if self.audio_injector.enable_adain:
        audio_global = _shard(self._audio_global_emb, dim=1).to(hidden_states.dtype)  # <-- SP fix
        audio_global_flat = audio_global.reshape(
            audio_global.shape[0] * num_frames, 1, audio_global.shape[-1])
        vis = self.audio_injector.injector_adain_layers[attn_id](vis, temb=audio_global_flat[:, 0])
    else:
        vis = self.audio_injector.injector_pre_norm_feat[attn_id](vis)
    vis = vis.to(hidden_states.dtype)
    audio_flat = audio_emb.reshape(
        audio_emb.shape[0] * num_frames, audio_emb.shape[2], audio_emb.shape[3]).to(vis.dtype)
    residual = self.audio_injector.injector[attn_id](
        x=vis, context=audio_flat, forward_batch_info=forward_batch_info)
    residual = residual.reshape(
        hidden_states.shape[0], num_frames * hw, hidden_states.shape[-1]).to(hidden_states.dtype)
    hidden_states[:, :num_noisy_tokens] = hidden_states[:, :num_noisy_tokens] + residual
    return hidden_states


def _apply_sp_patches():
    from wllm.serving.models.dit.liveavatar import LiveAvatarTransformer3DModel
    LiveAvatarTransformer3DModel._prefill_condition_cache = _sp_prefill_condition_cache
    LiveAvatarTransformer3DModel._inject_audio = _sp_inject_audio


_apply_sp_patches()


def _build_ctx(cfg, dit, vae, device):
    ts, sg = _timestep_schedule(cfg, device)
    return LAContext(
        cfg=cfg, dit_runner=dit, vae_runner=vae, timesteps=ts, sigmas=sg,
        device=device, dtype=torch.bfloat16,
        cond_prefix_tokens=int(getattr(cfg, "kv_cond_tokens", 0) or 0),
        motion_frames_raw=int(getattr(cfg, "motion_prefix_frames", 0) or 0),
        motion_frames_latent=int(getattr(cfg, "motion_prefix_latent_frames", 0) or 0),
        num_inference_steps=int(cfg.num_inference_steps))


class SPStepRank:
    """Ranks 1..N-1 under sequence parallelism: DiT (sharded) + 4 caches, mirror."""

    def __init__(self, cfg_path, rank, world):
        from wllm.serving.runner.dit_runner import DiTRunner
        from wllm.serving.memory.preallocated_cache import PreAllocatedKVCache
        from wllm.serving.utils.torch_utils import set_torch_options
        set_torch_options()

        self.rank, self.world = rank, world
        self.cfg = LiveAvatarReferenceConfig.from_yaml(cfg_path).to_runtime_config()
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16
        self.wg = init_dist(rank, world, sp_size=world)
        self.pg = self.wg.device_group

        self.dit = DiTRunner(self.cfg, self.dtype, self.device)
        n_steps = int(self.cfg.num_inference_steps)
        self.caches = [self.dit.kv_memory] + [PreAllocatedKVCache(self.cfg, self.device)
                                              for _ in range(n_steps - 1)]
        self.ctx = _build_ctx(self.cfg, self.dit, None, self.device)
        self.ref = None
        self.motion = None
        self.counter = 0
        logger.info("[dit_sp rank %d/%d] up on %s (sp)", rank, world, self.device)

    def _recv_ctrl(self):
        c = torch.zeros(1, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(c, src=0, group=self.pg)
        return int(c.item())

    def _init_session(self):
        pe = torch.empty((1, int(self.cfg.max_sequence_length), int(self.cfg.dit_config.text_dim)),
                         dtype=self.dtype, device=self.device)
        torch.distributed.broadcast(pe, src=0, group=self.pg)
        for c in self.caches:
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

    def _run_chunks(self):
        n = torch.zeros(1, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(n, src=0, group=self.pg)
        m = int(n.item())
        n_steps = int(self.cfg.num_inference_steps)
        for _ in range(m):
            ci = self.counter
            shp = _latent_shape(self.cfg, ci)
            noise = torch.empty(shp, dtype=self.dtype, device=self.device)
            audio = torch.empty(_audio_shape(self.cfg), dtype=self.dtype, device=self.device)
            torch.distributed.broadcast(noise, src=0, group=self.pg)
            torch.distributed.broadcast(audio, src=0, group=self.pg)
            latents = noise
            new_ref = None
            for step in range(n_steps):
                latents, nr = run_denoise_step(self.dit, self.caches[step], self.ctx, step,
                                               n_steps, latents, audio, ci, self.ref, self.motion)
                if nr is not None:
                    new_ref = nr
            if ci == 0 and new_ref is not None:
                self.ref = new_ref
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
        logger.info("[dit_sp rank %d] stopping", self.rank)


class DiTSPWorker(LiveAvatarWorker):
    def __init__(self, cfg_path):
        self.world = int(os.environ.get("DIT_SP_WORLD", "3"))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(free_port()))
        self._step_procs = []
        for k in range(1, self.world):
            env = os.environ.copy()
            env.update(RANK=str(k), LOCAL_RANK=str(k), WORLD_SIZE=str(self.world))
            self._step_procs.append(subprocess.Popen(
                [sys.executable, "-u", "-m", "wllm.apps.liveavatar.backend.rocm.dit_sp.worker",
                 "--step-rank", "--rank", str(k), "--world", str(self.world),
                 "--config", cfg_path], env=env))
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE=str(self.world))
        torch.cuda.set_device(0)
        self.wg = init_dist(0, self.world, sp_size=self.world)
        self.pg = self.wg.device_group
        for v in _DIST_VARS:
            os.environ.pop(v, None)
        self._ctx = None
        self._bcast_done = False
        super().__init__(cfg_path)

    def _ensure_ctx(self):
        if self._ctx is None:
            self._ctx = _build_ctx(self.cfg, self.pipe.dit_runner, self.pipe.vae_runner, self.device)
        return self._ctx

    def _create_pipeline(self):
        return DriverPipeline(cfg=self.cfg, device=self.device)

    def _bcast_ctrl(self, code):
        torch.distributed.broadcast(torch.tensor([code], dtype=torch.int64, device=self.device),
                                    src=0, group=self.pg)

    def warmup(self):
        # distributed SPMD warmup: prime session + run a few chunks with dummy audio,
        # mirrored by the step ranks, so all ranks warm the sharded DiT together.
        from wllm.serving.utils.rand import set_global_seed
        self._ensure_ctx()
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None,
                               image_path=self.cfg.image_path)
        self._broadcast_session()
        n_warm = max(3, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 1)
        self._bcast_ctrl(CTRL_RUN_CHUNKS)
        torch.distributed.broadcast(torch.tensor([n_warm], dtype=torch.int64, device=self.device),
                                    src=0, group=self.pg)
        tgt = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
        for _ in range(n_warm):
            dummy = np.zeros((int(self.cfg.tts_chunk_size),), dtype=np.float32)
            audio = self.extract_audio_features(dummy, target_frames=tgt)
            self._run_chunk_spmd(audio, np.zeros((int(self.cfg.tts_chunk_size),), np.float32),
                                 write=False)
        self.pipe.reset()
        self._bcast_ctrl(CTRL_RESET)
        self._bcast_done = False

    def _broadcast_session(self):
        self._bcast_ctrl(CTRL_INIT_SESSION)
        pe = self.pipe._session_ctx["prompt_embeds"].to(self.device, self.pipe.dtype).contiguous()
        torch.distributed.broadcast(pe, src=0, group=self.pg)
        torch.distributed.broadcast(self.pipe._ref_latents.to(self.device, self.pipe.dtype).contiguous(),
                                    src=0, group=self.pg)
        torch.distributed.broadcast(self.pipe._motion_latents.to(self.device, self.pipe.dtype).contiguous(),
                                    src=0, group=self.pg)
        self._bcast_done = True

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

    def _run_chunk_spmd(self, audio_feats, chunk_audio, write=True):
        pipe = self.pipe
        ci = pipe._session_ctx["latent_chunk_idx"]
        self._ctx.chunk_idx = ci
        noise = self._draw_noise(ci)
        torch.distributed.broadcast(noise.contiguous(), src=0, group=self.pg)
        torch.distributed.broadcast(audio_feats.contiguous(), src=0, group=self.pg)
        latents = noise
        n_steps = int(self.cfg.num_inference_steps)
        new_ref = None
        for step in range(n_steps):
            latents, nr = run_denoise_step(pipe.dit_runner, pipe._step_kv_caches[step], self._ctx,
                                           step, n_steps, latents, audio_feats, ci,
                                           pipe._ref_latents, pipe._motion_latents)
            if nr is not None:
                new_ref = nr
        if ci == 0 and new_ref is not None:
            pipe._ref_latents = new_ref
        pipe._session_ctx["latent_chunk_idx"] = ci + 1
        if write:
            frames = []
            for fi in range(int(latents.shape[2])):
                vi = pipe.vae_runner.run(latents[:, :, fi:fi + 1, :, :].clone(), is_first_chunk=False)
                frames.append(vi.repeat_interleave(2, dim=1)[0].cpu().numpy())
            video = np.concatenate(frames, axis=0)
            af = self._frame_audio(np.asarray(chunk_audio, dtype=np.float32))
            n = min(video.shape[0], af.shape[0])
            if n > 0:
                pipe._video_buffer.write(video[:n])
                self.audio_output_buffer.write(af[:n])

    def _run_liveavatar_reference(self, audio_samples):
        self._ensure_ctx()
        if not self._bcast_done:
            self._broadcast_session()
        step_samples = int(self.cfg.tts_chunk_size)
        tgt = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
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
        for ca in chunks:
            audio = self.extract_audio_features(ca, target_frames=tgt)
            self._run_chunk_spmd(audio, ca, write=True)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-rank", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--config", required=True)
    args, _ = ap.parse_known_args()
    if args.step_rank:
        SPStepRank(args.config, args.rank, args.world).serve()
    else:
        DiTSPWorker(cfg_path=args.config).loop()


if __name__ == "__main__":
    main()
