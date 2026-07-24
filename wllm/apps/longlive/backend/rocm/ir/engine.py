"""Shared LongLive generation core.

This module re-implements the *exact* numerical behavior of the user's
reference backend (``wllm/apps/longlive/reference/pipeline.py``) against the shared
runtime primitives (``wllm.runner.*``), with **no** import of
the reference code. It is the single source of truth for the LongLive
generation math that both

  * the IR operators (``ir/ops.py``), used for Phase-2
    validation, and
  * the deployment variants (the per-variant packages)

build on. Keeping the math in one place guarantees the IR that we validate is
the same computation the variants execute.

The reference performs, per chunk (``chunk_size`` latent frames):
  1. sample fresh noise,
  2. a DMD few-step denoise (``num_inference_steps`` DiT forward passes, with
     re-noising between steps),
  3. a ``t=0`` DiT pass (``is_cache=True``) that writes this chunk's *clean*
     K/V into the sliding-window ring so later chunks attend against it,
  4. per-latent-frame causal VAE decode into RGB pixel frames.

Cross-chunk state:
  * ``kv_cache`` — the DiT sliding-window KV ring (lives inside
    ``DiTRunner.kv_memory``). Every DiT pass writes the current chunk's K/V at
    its ring slot and attends over ``[0:cache_end]``.
  * ``vae_feat_cache`` — the causal VAE feature cache (lives inside the VAE).
  * ``encoder_kv`` — cross-attention K/V from the current prompt (prompt-scoped,
    refilled by ``set_prompt``).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from wllm.serving.rt_config import RTConfig
from wllm.serving.runner.dit_runner import DiTRunner
from wllm.serving.runner.text_encoder_runner import TextEncoderRunner
from wllm.serving.runner.vae_runner import VAERunner
from wllm.serving.utils.dtype import parse_dtype_getattr
from wllm.serving.utils.prompt_utils import prompt_clean


class ChunkIndices:
    """Per-chunk ring-buffer / RoPE index bundle (a pure function of session
    counters). Mirrors the index arithmetic at the top of the reference
    ``LongLivePipeline.step``."""

    __slots__ = (
        "cache_start",
        "cache_end",
        "rope_start",
        "rope_end",
        "cache_chunk_idx",
        "new_max_filled_slot",
    )

    def __init__(
        self,
        cache_start: int,
        cache_end: int,
        rope_start: int,
        rope_end: int,
        cache_chunk_idx: int,
        new_max_filled_slot: int,
    ) -> None:
        self.cache_start = cache_start
        self.cache_end = cache_end
        self.rope_start = rope_start
        self.rope_end = rope_end
        self.cache_chunk_idx = cache_chunk_idx
        self.new_max_filled_slot = new_max_filled_slot


class LongLiveCore:
    """Reference-faithful LongLive generation, decomposed into reusable steps.

    Runners are loaded on ``device``. For sequence-parallel variants, construct
    one ``LongLiveCore`` per rank after the SP process group is initialized; the
    runners read ``get_sp_world_size()`` when they allocate the KV cache and the
    attention layers do the all-to-all internally.
    """

    def __init__(
        self,
        cfg: RTConfig,
        device: torch.device,
        *,
        build_text_encoder: bool = True,
        build_vae: bool = True,
        build_dit: bool = True,
        noise_hook=None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = parse_dtype_getattr(cfg.dtype)
        # Optional in-place hook applied to every freshly drawn noise tensor.
        # Used by sequence-parallel variants to broadcast rank-0's noise to all
        # ranks (``global_broadcast(t, src=0)``), so every rank shards the
        # *identical* full noise the reference drew -> bit-faithful output.
        self._noise_hook = noise_hook

        self.text_encoder_runner: Optional[TextEncoderRunner] = (
            TextEncoderRunner(cfg, self.dtype, device) if build_text_encoder else None
        )
        self.vae_runner: Optional[VAERunner] = (
            VAERunner(cfg, self.dtype, device) if build_vae else None
        )
        self.dit_runner: Optional[DiTRunner] = (
            DiTRunner(cfg, self.dtype, device) if build_dit else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

        self._timesteps, self._sigmas = self._build_timestep_schedule()
        self._timesteps = self._timesteps.to(device)
        self._sigmas = self._sigmas.to(device)

        self._noise_generator = torch.Generator(device=device)

        # session counters (reset() initializes them)
        self._prompt_embeds: Optional[torch.Tensor] = None
        self.reset()

    # ------------------------------------------------------------------
    # schedule (identical to reference)
    # ------------------------------------------------------------------
    def _build_timestep_schedule(self):
        num_steps = int(self.cfg.num_inference_steps)
        shift = float(self.cfg.timestep_shift)
        sigmas_lin = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)[:-1]
        sigmas = shift * sigmas_lin / (1.0 + (shift - 1.0) * sigmas_lin)
        timesteps = sigmas * 1000.0
        return timesteps, sigmas

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------
    def seed(self) -> None:
        self._noise_generator.manual_seed(int(self.cfg.seed))

    def reset(self) -> None:
        self._prompt_embeds = None
        self._block_idx = 0
        self._latent_decode_count = 0
        self._rolling_writes = 0
        self._max_filled_slot = -1
        self._shot_index = 0
        self._temporal_offset_latents = 0
        self._pinned_slot = -1
        self._scene_cut_pending = False
        self._perturbed = False
        if self.vae_runner is not None:
            self.vae_runner.clear()

    @property
    def block_idx(self) -> int:
        return self._block_idx

    @property
    def latent_decode_count(self) -> int:
        return self._latent_decode_count

    # ------------------------------------------------------------------
    # prompt encoding (replicates BasePipeline._get_t5_prompt_embeds)
    # ------------------------------------------------------------------
    def _get_t5_prompt_embeds(self, prompt: Union[str, List[str]]):
        max_sequence_length = int(self.cfg.max_sequence_length)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder_runner.run(
            text_input_ids.to(self.device), mask.to(self.device)
        )
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )
        return prompt_embeds

    def set_prompt(self, prompt: Union[str, List[str]]) -> torch.Tensor:
        """Encode the prompt and refill the DiT cross-attention KV cache.

        Returns the prompt embeds (also stashed on ``self``)."""
        prompt_embeds = self._get_t5_prompt_embeds(prompt)
        self._prompt_embeds = prompt_embeds
        if self.dit_runner is not None:
            self.dit_runner.encode(prompt_embeds)
        return prompt_embeds

    def apply_scene_cut(self, prompt: str) -> bool:
        """Replicate the reference multi-shot scene-cut bookkeeping. Returns
        True if this prompt update was treated as a scene cut."""
        is_scene_cut = (
            isinstance(prompt, str)
            and int(self.cfg.multi_shot_rope_offset) > 0
            and self._sink_chunks() > 0
            and self._block_idx > 0
            and prompt.startswith(self.cfg.scene_cut_prefix)
        )
        if is_scene_cut:
            self._shot_index += 1
            self._temporal_offset_latents = self._shot_index * int(
                self.cfg.multi_shot_rope_offset
            )
            self._scene_cut_pending = True
        return is_scene_cut

    # ------------------------------------------------------------------
    # ring-buffer index arithmetic (identical to reference step())
    # ------------------------------------------------------------------
    def _ring_capacity_latents(self) -> int:
        chunk = int(self.cfg.chunk_size)
        return max(chunk, int(self.cfg.context_window_size) + chunk)

    def _sink_chunks(self) -> int:
        return int(self.cfg.sink_size) // int(self.cfg.chunk_size)

    def compute_chunk_indices(self) -> ChunkIndices:
        chunk_size = int(self.cfg.chunk_size)
        kv_spatial = int(self.cfg.kv_spatial)
        chunk_tokens = chunk_size * kv_spatial
        ring_capacity_latents = self._ring_capacity_latents()
        ring_capacity_chunks = ring_capacity_latents // chunk_size
        ring_capacity_tokens = ring_capacity_chunks * chunk_tokens
        sink_chunks = self._sink_chunks()
        rolling_capacity_chunks = ring_capacity_chunks - sink_chunks
        assert rolling_capacity_chunks > 0

        if self._block_idx < sink_chunks:
            cache_chunk_idx = self._block_idx
        elif self._pinned_slot < 0:
            cache_chunk_idx = sink_chunks + (
                self._rolling_writes % rolling_capacity_chunks
            )
        else:
            available_slots = [
                sink_chunks + i
                for i in range(rolling_capacity_chunks)
                if (sink_chunks + i) != self._pinned_slot
            ]
            assert available_slots
            cache_chunk_idx = available_slots[
                self._rolling_writes % len(available_slots)
            ]
        cache_start = cache_chunk_idx * chunk_tokens
        new_max_filled_slot = max(self._max_filled_slot, cache_chunk_idx)
        cache_end = (new_max_filled_slot + 1) * chunk_tokens

        global_start_latent = (
            self._block_idx * chunk_size + self._temporal_offset_latents
        )
        rope_start = global_start_latent * kv_spatial
        rope_end = rope_start + chunk_tokens

        assert cache_start + chunk_tokens <= ring_capacity_tokens
        assert cache_end <= ring_capacity_tokens
        return ChunkIndices(
            cache_start, cache_end, rope_start, rope_end,
            cache_chunk_idx, new_max_filled_slot,
        )

    def advance_chunk(self, idx: ChunkIndices) -> None:
        """Post-chunk session-counter update (identical to reference tail)."""
        self._max_filled_slot = idx.new_max_filled_slot
        if self._block_idx >= self._sink_chunks():
            self._rolling_writes += 1
        if self._scene_cut_pending:
            self._pinned_slot = idx.cache_chunk_idx
            self._scene_cut_pending = False
        self._block_idx += 1

    # ------------------------------------------------------------------
    # per-chunk compute primitives
    # ------------------------------------------------------------------
    def _draw(self, shape) -> torch.Tensor:
        t = torch.randn(
            shape, device=self.device, dtype=self.dtype,
            generator=self._noise_generator,
        )
        if self._noise_hook is not None:
            t = self._noise_hook(t)
        return t

    def sample_noise(self) -> torch.Tensor:
        noise_shape = (
            1,
            int(self.cfg.dit_config.out_channels),
            int(self.cfg.chunk_size),
            int(self.cfg.latent_height),
            int(self.cfg.latent_width),
        )
        return self._draw(noise_shape)

    def denoise_step(
        self, latents: torch.Tensor, step_idx: int, idx: ChunkIndices
    ) -> torch.Tensor:
        """One DMD denoise step: DiT flow prediction + x0 projection + (unless
        last) re-noise. Returns the latents fed to the next step (or the final
        clean latents on the last step). Mirrors the reference exactly,
        including the fp64 conversions and the fresh-noise draw ordering."""
        chunk_size = int(self.cfg.chunk_size)
        num_steps = len(self._timesteps)
        t = self._timesteps[step_idx]
        sigma_t = self._sigmas[step_idx]
        timestep = torch.full(
            (chunk_size,), t.item(), device=self.device, dtype=torch.float32
        )
        flow_pred = self.dit_runner.run(
            latents=latents,
            timestep=timestep,
            is_cache=False,
            cache_start=idx.cache_start,
            cache_end=idx.cache_end,
            rope_start=idx.rope_start,
            rope_end=idx.rope_end,
        )
        x0_pred = (
            latents.to(torch.float64)
            - sigma_t.to(torch.float64) * flow_pred.to(torch.float64)
        ).to(latents.dtype)
        if step_idx < num_steps - 1:
            sigma_next = self._sigmas[step_idx + 1].to(torch.float64)
            fresh_noise = self._draw(latents.shape)
            latents = (
                (1.0 - sigma_next) * x0_pred.to(torch.float64)
                + sigma_next * fresh_noise.to(torch.float64)
            ).to(latents.dtype)
        else:
            latents = x0_pred
        return latents

    def cache_write(self, latents: torch.Tensor, idx: ChunkIndices) -> None:
        """t=0 DiT pass that persists this chunk's clean K/V into the ring."""
        chunk_size = int(self.cfg.chunk_size)
        zero_timestep = torch.zeros(
            (chunk_size,), device=self.device, dtype=torch.float32
        )
        self.dit_runner.run(
            latents=latents,
            timestep=zero_timestep,
            is_cache=True,
            cache_start=idx.cache_start,
            cache_end=idx.cache_end,
            rope_start=idx.rope_start,
            rope_end=idx.rope_end,
        )

    def decode_frame(self, latents: torch.Tensor, local_frame: int) -> np.ndarray:
        """Causal VAE decode of one latent frame -> uint8 [T,H,W,3] pixel
        frames. Increments the decode counter (controls the is_first flag)."""
        latent_one = latents[:, :, local_frame:local_frame + 1].contiguous()
        is_first = (self._latent_decode_count == 0)
        video_chunk = self.vae_runner.run(latent_one, is_first)
        self._latent_decode_count += 1
        return video_chunk[0]

    # ------------------------------------------------------------------
    # convenience: a whole reference-identical chunk (single GPU)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def step_reference(self) -> List[np.ndarray]:
        """Generate one chunk exactly like the reference and return the list of
        per-latent-frame decoded pixel arrays (host uint8). Used by the IR
        executor's convenience path and by the single-GPU baseline variant."""
        idx = self.compute_chunk_indices()
        latents = self.sample_noise()
        for i in range(len(self._timesteps)):
            latents = self.denoise_step(latents, i, idx)
        self.cache_write(latents, idx)
        frames = []
        for l in range(int(self.cfg.chunk_size)):
            frames.append(self.decode_frame(latents, l).cpu().numpy())
        self.advance_chunk(idx)
        return frames
