"""IR operators for the LiveAvatar exposed computation graph.

The exposed graph is the sound-to-video generation: per audio chunk,
    wav2vec feature extraction
      -> 4 sequential DiT denoising steps (each with its OWN KV cache)
      -> causal VAE decode (frame by frame)

These operators are a faithful, op-by-op transcription of
`wllm.apps.liveavatar.reference.pipeline.LiveAvatarPipeline.step` (the reference).
Each operator calls the SAME `DiTRunner.run` / `VAERunner.run` with the SAME
arguments the reference uses, so the SequentialExecutor over this graph
reproduces the reference's exact sequence of model calls (validated
bit-exact against the reference).

State model (what makes the per-step pipeline legal):
  - `cache_k`  (chunk_persistent): step k's private KV cache. Written by step k
    of chunk N (the new chunk's tokens + cond prefix), read by step k of chunk
    N+1. step i and step j (i != j) touch DIFFERENT caches -> independent across
    chunks -> the 4 steps are a 4-stage cross-chunk pipeline.
  - `vae_cache` (chunk_persistent): the causal VAE decoder's internal temporal
    cache (mutated in place by VAERunner.run with is_first_chunk=False).
  - `ref_latents` / `motion_latents` (session_init): condition tensors for the
    prefill. session_init scope => read-only during the periodic phase => they
    do NOT couple the per-step stages. (ref_latents is rebound once, on chunk 0;
    that is a startup bubble, not a steady-state cross-chunk constraint.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from wllm.serving.ir import IROperator, OpType, TensorPort, StreamMode


# --------------------------------------------------------------------------
# Per-chunk context.  Holds model handles + the per-chunk index math, mirrored
# line-for-line from LiveAvatarPipeline.step / ._chunk_global_range /
# ._build_full_timestep / ._get_zero_cond_latents.
# --------------------------------------------------------------------------
@dataclass
class LAContext:
    cfg: Any
    dit_runner: Any
    vae_runner: Any
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    cond_prefix_tokens: int
    motion_frames_raw: int
    motion_frames_latent: int
    # wav2vec extraction callable: (audio_samples_np, target_frames) -> features
    extract_audio_features: Optional[Callable] = None
    _zero_cond_cache: dict = field(default_factory=dict)

    # per-chunk fields (set by prepare_chunk)
    chunk_i: int = 0
    generate_latent_num: int = 0
    cache_start: int = 0
    cache_end: int = 0
    rope_start: int = 0
    rope_end: int = 0
    need_cond_prefill: bool = True

    def _chunk_global_range(self, chunk_i: int):
        if chunk_i <= 0:
            return 0, int(self.cfg.first_chunk_size)
        start = int(self.cfg.first_chunk_size) + (chunk_i - 1) * int(self.cfg.chunk_size)
        end = start + int(self.cfg.chunk_size)
        return start, end

    def build_full_timestep(self, timestep_value: torch.Tensor, chunk_i: int) -> torch.Tensor:
        if chunk_i > 0:
            t_now = torch.full((1, self.cfg.chunk_size), timestep_value,
                               device=self.device, dtype=self.timesteps.dtype)
            t_ctx = torch.full(
                (1, self.cfg.first_chunk_size + (chunk_i - 1) * self.cfg.chunk_size),
                self.cfg.stabilization_level - 1,
                device=self.device, dtype=self.timesteps.dtype)
            return torch.cat([t_ctx, t_now], dim=1)
        return torch.full((1, int(self.cfg.first_chunk_size)), timestep_value,
                          device=self.device, dtype=self.timesteps.dtype)

    def get_zero_cond_latents(self, num_frames: int) -> torch.Tensor:
        if num_frames in self._zero_cond_cache:
            return self._zero_cond_cache[num_frames]
        cond = torch.zeros(1, self.cfg.dit_config.out_channels, num_frames,
                           self.cfg.latent_height, self.cfg.latent_width,
                           dtype=self.dtype, device=self.device)
        self._zero_cond_cache[num_frames] = cond
        return cond

    def prepare_chunk(self, chunk_i: int) -> None:
        """Compute the per-chunk index math (mirror of pipeline.step lines)."""
        self.chunk_i = chunk_i
        g_start, g_end = self._chunk_global_range(chunk_i)
        self.generate_latent_num = int(g_end - g_start)

        kv_spatial = int(self.cfg.kv_spatial)
        cond_prefix = int(self.cond_prefix_tokens)
        noisy_ring_capacity_latents = max(
            int(self.cfg.chunk_size),
            int(self.cfg.context_window_size) + int(self.cfg.chunk_size))
        write_span_tokens = self.generate_latent_num * kv_spatial
        write_start_tokens = (g_start % noisy_ring_capacity_latents) * kv_spatial
        self.cache_start = cond_prefix + write_start_tokens
        cache_tokens = min(g_end, noisy_ring_capacity_latents) * kv_spatial
        self.cache_end = cond_prefix + cache_tokens
        self.rope_start = cond_prefix + g_start * kv_spatial
        self.rope_end = self.rope_start + write_span_tokens
        self.need_cond_prefill = chunk_i <= 1


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
class Wav2VecExtract(IROperator):
    """Exposed: wav2vec feature extraction for one audio chunk."""

    def __init__(self):
        super().__init__(
            name="wav2vec_extract",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("audio_samples")],
            outputs=[TensorPort("audio_features")],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: LAContext, state):
        frames_per_latent = int(ctx.cfg.vae_config.scale_factor_temporal)
        step_frames = int(ctx.cfg.chunk_size) * frames_per_latent
        feats = ctx.extract_audio_features(inputs["audio_samples"], target_frames=step_frames)
        return {"audio_features": feats}


class DiTDenoiseStep(IROperator):
    """Exposed: one denoising step k. Reads/writes its private cache_k.

    Mirrors the body of the `for step_idx` loop in pipeline.step: optional cond
    prefill (chunk_i<=1), the generate forward (writes cache_k self-attn KV),
    and the Euler update latents += dt * noise_pred. On chunk 0 the last step
    rebinds ref_latents (session_init)."""

    def __init__(self, step_idx: int):
        cache = f"cache_{step_idx}"
        writes = [cache]
        # the final step on chunk 0 rebinds ref_latents
        writes_ref = ["ref_latents"]
        super().__init__(
            name=f"dit_step_{step_idx}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents"), TensorPort("audio_features")],
            outputs=[TensorPort("latents")],
            state_reads=[cache, "ref_latents", "motion_latents"],
            state_writes=writes + writes_ref,
        )
        self.step_idx = step_idx
        self.cache = cache

    def execute(self, inputs, ctx: LAContext, state):
        runner = ctx.dit_runner
        cache = state.get(self.cache)          # in-place-mutated KV cache for step k
        runner.kv_memory = cache
        ref_latents = state.get("ref_latents")
        motion_latents = state.get("motion_latents")

        latents = inputs["latents"]
        audio_input = inputs["audio_features"]
        cond_prefix = int(ctx.cond_prefix_tokens)
        cond_latents_gen = ctx.get_zero_cond_latents(ctx.generate_latent_num)

        if ctx.need_cond_prefill:
            prefill_t = torch.zeros((1,), device=ctx.device, dtype=ctx.timesteps.dtype)
            runner.run(
                latents=latents, timestep=prefill_t, is_cache=True,
                cache_start=0, cache_end=cond_prefix, rope_start=0, rope_end=cond_prefix,
                viewmats=None, Ks=None, action=None, i2v_condition=None, cond_latents=None,
                prefill_cond=True, cond_prefix_tokens=cond_prefix,
                ref_latents=ref_latents, motion_latents=motion_latents,
                motion_frames_raw=ctx.motion_frames_raw,
                motion_frames_latent=ctx.motion_frames_latent,
            )

        timestep_value = ctx.timesteps[self.step_idx]
        timestep = ctx.build_full_timestep(timestep_value, ctx.chunk_i)
        timestep_slice = timestep[:, -ctx.generate_latent_num:].flatten()

        noise_pred = runner.run(
            latents=latents, timestep=timestep_slice, is_cache=False,
            cache_start=ctx.cache_start, cache_end=ctx.cache_end,
            rope_start=ctx.rope_start, rope_end=ctx.rope_end,
            viewmats=None, Ks=None, action=None, i2v_condition=None,
            audio_input=audio_input, cond_latents=cond_latents_gen,
            cond_prefix_tokens=cond_prefix,
            motion_frames_raw=ctx.motion_frames_raw,
            motion_frames_latent=ctx.motion_frames_latent,
        )

        sigma = ctx.sigmas[self.step_idx]
        sigma_next = ctx.sigmas[self.step_idx + 1]
        dt = sigma_next - sigma
        latents_out = latents + dt * noise_pred

        if self.step_idx == int(ctx.cfg.num_inference_steps) - 1 and ctx.chunk_i == 0:
            state.set("ref_latents", latents_out[:, :, :1].detach().clone())

        return {"latents": latents_out}


class VAEDecode(IROperator):
    """Exposed: causal VAE decode of the chunk's latents, frame by frame.

    Mutates the VAE decoder's internal temporal cache in place (chunk_persistent)
    -- modeled via the `vae_cache` state name whose stored value is the
    vae_runner itself (in-place mutation pattern)."""

    def __init__(self):
        super().__init__(
            name="vae_decode",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents")],
            outputs=[TensorPort("video")],
            state_reads=["vae_cache"],
            state_writes=["vae_cache"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: LAContext, state):
        _ = state.get("vae_cache")   # declares in-place mutation of the VAE temporal cache
        runner = ctx.vae_runner
        latents = inputs["latents"]
        import numpy as np
        chunk_video = []
        for frame_i in range(int(latents.shape[2])):
            latent_i = latents[:, :, frame_i:frame_i + 1, :, :].clone()
            video_i = runner.run(latent_i, is_first_chunk=False)
            video_i = video_i.repeat_interleave(2, dim=1)
            chunk_video.append(video_i[0].cpu().numpy())
        return {"video": np.concatenate(chunk_video, axis=0)}
