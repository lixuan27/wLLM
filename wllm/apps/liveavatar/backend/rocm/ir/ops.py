"""IR operators for the exposed LiveAvatar sound-to-video model.

These decompose the reference `LiveAvatarPipeline.step()`
(wllm/apps/liveavatar/reference/pipeline.py) into fine-grained IROperators so the
analysis tools can surface:

  * the 4 denoising steps each own a *separate* KV cache (step_k_kv,
    chunk_persistent) -> denoise_k(chunk N) and denoise_j(chunk N+1) with j!=k
    share no persistent state -> pipeline-parallel across chunks (one GPU per
    denoising step); and
  * the VAE decode owns a separate temporal cache (vae_cache) -> its own stage.

Each op carries its own `execute`. Ops wrap the shared runtime runners
(DiTRunner / VAERunner) held on the context; there is no model code here.

Faithfulness to the reference (see LiveAvatarPipeline.step):
  - draw_noise draws current_chunk_latents exactly as the reference does
    (randn (1,C,gen,H,W) float32 -> dtype) on the DiT device.
  - denoise_step_k: activate step-k cache, prefill the condition cache on
    chunks 0/1 (need_cond_prefill = chunk_i <= 1), run the generation forward
    with the same cache/rope ring-buffer indices, then Euler update
    latents += (sigma_{k+1}-sigma_k) * noise_pred.
  - On chunk 0, after the last step, ref_latents := latents[:, :, :1] (matches
    the reference's `if chunk_i == 0: self._ref_latents = ...`).
  - vae_decode: decode each of the gen latents frame-by-frame (is_first_chunk
    =False), repeat_interleave(2, dim=1), concatenate -> (frames,H,W,3) uint8.

ref_latents / motion_latents are session_init scope: written at session init
(and ref_latents once more at the end of chunk 0), then read-only. They are
replicated (broadcast) across pipeline stages rather than creating a per-chunk
dependency; the one-time chunk-0 ref_latents update is handled explicitly by
deployments (broadcast at the chunk 0->1 boundary). This keeps the 4 step
caches as 4 independent pipeline stages in the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from wllm.serving.ir import IROperator, OpType, StreamMode, TensorPort


@dataclass
class LAContext:
    """Per-session context handed to every op's execute()."""

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
    num_inference_steps: int
    chunk_idx: int = 0                # mutated by the harness before each chunk
    noise_seed: Optional[int] = None  # if set, draw_noise reseeds per chunk
    _zero_cond_cache: dict = field(default_factory=dict)

    # -- geometry (mirrors LiveAvatarPipeline) -------------------------------
    def chunk_global_range(self, chunk_i: int) -> tuple[int, int]:
        fcs = int(self.cfg.first_chunk_size)
        cs = int(self.cfg.chunk_size)
        if chunk_i <= 0:
            return 0, fcs
        start = fcs + (chunk_i - 1) * cs
        return start, start + cs

    def zero_cond(self, num_frames: int) -> torch.Tensor:
        if num_frames not in self._zero_cond_cache:
            self._zero_cond_cache[num_frames] = torch.zeros(
                1, int(self.cfg.dit_config.out_channels), num_frames,
                int(self.cfg.latent_height), int(self.cfg.latent_width),
                dtype=self.dtype, device=self.device,
            )
        return self._zero_cond_cache[num_frames]

    def build_full_timestep(self, timestep_value: torch.Tensor, chunk_i: int) -> torch.Tensor:
        cs = int(self.cfg.chunk_size)
        fcs = int(self.cfg.first_chunk_size)
        if chunk_i > 0:
            t_now = torch.full((1, cs), timestep_value, device=self.device,
                               dtype=self.timesteps.dtype)
            t_ctx = torch.full((1, fcs + (chunk_i - 1) * cs),
                               self.cfg.stabilization_level - 1,
                               device=self.device, dtype=self.timesteps.dtype)
            return torch.cat([t_ctx, t_now], dim=1)
        return torch.full((1, fcs), timestep_value, device=self.device,
                          dtype=self.timesteps.dtype)

    def cache_geometry(self, chunk_i: int) -> dict:
        gs, ge = self.chunk_global_range(chunk_i)
        gen = ge - gs
        kv = int(self.cfg.kv_spatial)
        cp = int(self.cond_prefix_tokens)
        ring = max(int(self.cfg.chunk_size),
                   int(self.cfg.context_window_size) + int(self.cfg.chunk_size))
        write_start = (gs % ring) * kv
        write_span = gen * kv
        return {
            "gen": gen,
            "cache_start": cp + write_start,
            "cache_end": cp + min(ge, ring) * kv,
            "rope_start": cp + gs * kv,
            "rope_end": cp + gs * kv + write_span,
        }


class DrawNoise(IROperator):
    """Draw the chunk's initial latents exactly like the reference step()."""

    def __init__(self):
        super().__init__(
            name="draw_noise", op_type=OpType.EXPOSED,
            inputs=[], outputs=[TensorPort("noise")],
            state_reads=[], state_writes=[], stream_mode=StreamMode.BATCH,
        )

    def execute(self, inputs, ctx: LAContext, state):
        if ctx.noise_seed is not None:
            from wllm.serving.utils.rand import set_global_seed
            set_global_seed(ctx.noise_seed + ctx.chunk_idx)
        gs, ge = ctx.chunk_global_range(ctx.chunk_idx)
        gen = ge - gs
        noise = torch.randn(
            (1, int(ctx.cfg.dit_config.out_channels), gen,
             int(ctx.cfg.latent_height), int(ctx.cfg.latent_width)),
            device=ctx.device, dtype=torch.float32,
        ).to(dtype=ctx.dtype)
        return {"noise": noise}


class DenoiseStep(IROperator):
    """One denoising step k, owning KV cache step_k_kv (mirrors the step loop)."""

    def __init__(self, step_idx: int, num_steps: int):
        cache = f"step_{step_idx}_kv"
        writes = [cache]
        # the final step updates ref_latents on chunk 0
        if step_idx == num_steps - 1:
            writes = [cache, "ref_latents"]
        super().__init__(
            name=f"denoise_step_{step_idx}", op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents"), TensorPort("audio_features")],
            outputs=[TensorPort("latents")],
            state_reads=[cache, "ref_latents", "motion_latents"],
            state_writes=writes,
            stream_mode=StreamMode.BATCH,
        )
        self.step_idx = step_idx
        self.num_steps = num_steps
        self.cache = cache
        self.is_last = step_idx == num_steps - 1

    def execute(self, inputs, ctx: LAContext, state):
        dit = ctx.dit_runner
        dit.kv_memory = state.get(self.cache)  # in-place mutation of this cache
        ref_latents = state.get("ref_latents")
        motion_latents = state.get("motion_latents")
        chunk_i = ctx.chunk_idx
        geo = ctx.cache_geometry(chunk_i)
        gen = geo["gen"]
        cp = int(ctx.cond_prefix_tokens)
        latents = inputs["latents"]
        audio = inputs["audio_features"]

        if chunk_i <= 1:  # need_cond_prefill
            prefill_t = torch.zeros((1,), device=ctx.device, dtype=ctx.timesteps.dtype)
            dit.run(
                latents=latents, timestep=prefill_t, is_cache=True,
                cache_start=0, cache_end=cp, rope_start=0, rope_end=cp,
                prefill_cond=True, cond_prefix_tokens=cp,
                ref_latents=ref_latents, motion_latents=motion_latents,
                motion_frames_raw=ctx.motion_frames_raw,
                motion_frames_latent=ctx.motion_frames_latent,
            )

        t_val = ctx.timesteps[self.step_idx]
        timestep = ctx.build_full_timestep(t_val, chunk_i)
        timestep_slice = timestep[:, -gen:].flatten()

        noise_pred = dit.run(
            latents=latents, timestep=timestep_slice, is_cache=False,
            cache_start=geo["cache_start"], cache_end=geo["cache_end"],
            rope_start=geo["rope_start"], rope_end=geo["rope_end"],
            audio_input=audio, cond_latents=ctx.zero_cond(gen),
            cond_prefix_tokens=cp,
            motion_frames_raw=ctx.motion_frames_raw,
            motion_frames_latent=ctx.motion_frames_latent,
        )
        dt = ctx.sigmas[self.step_idx + 1] - ctx.sigmas[self.step_idx]
        latents = latents + dt * noise_pred

        if self.is_last and chunk_i == 0:
            state.set("ref_latents", latents[:, :, :1].detach().clone())
        return {"latents": latents}


class VaeDecode(IROperator):
    """Decode the chunk latents to RGB frames (owns the VAE temporal cache)."""

    def __init__(self):
        super().__init__(
            name="vae_decode", op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents")], outputs=[TensorPort("video")],
            state_reads=["vae_cache"], state_writes=["vae_cache"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: LAContext, state):
        vae = state.get("vae_cache")  # the VAERunner; internal temporal cache mutated
        latents = inputs["latents"]
        frames = []
        for fi in range(int(latents.shape[2])):
            latent_i = latents[:, :, fi:fi + 1, :, :].clone()
            video_i = vae.run(latent_i, is_first_chunk=False)
            video_i = video_i.repeat_interleave(2, dim=1)
            frames.append(video_i[0].cpu().numpy())
        return {"video": np.concatenate(frames, axis=0)}
