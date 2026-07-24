"""LongLive core pipeline (single-GPU, reference-faithful).

Mirrors ``wllm/apps/longlive/reference/pipeline.py`` numerics but factors the
per-chunk math out into ``generation.py`` so the IR operators and the
multi-GPU variant workers share one implementation. Built from shared-runtime
primitives only (DiTRunner / VAERunner / TextEncoderRunner via BasePipeline);
nothing is imported from the reference tree.

This class is intentionally a *library* object: it does not own IPC buffers or
a control loop. Workers (single- or multi-GPU) drive it.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch

from wllm.serving.pipeline.base import BasePipeline
from wllm.serving.rt_config import RTConfig
from wllm.serving.runner.dit_runner import DiTRunner

from wllm.apps.longlive.backend.cuda import generation as G


class LongLiveCore(BasePipeline):
    """Reference-faithful LongLive generation core, single device."""

    def __init__(self, cfg: RTConfig, device: torch.device):
        super().__init__(cfg, device)
        self._timesteps, self._sigmas = self._build_timestep_schedule()
        self._timesteps = self._timesteps.to(device)
        self._sigmas = self._sigmas.to(device)
        self.noise_generator = torch.Generator(device=device)
        self.ring = G.new_ring_state()
        self._prompt_set = False

    # -- BasePipeline hooks -------------------------------------------------
    def _create_dit_runner(self):
        return DiTRunner(self.cfg, self.dtype, self.device)

    def _build_timestep_schedule(self):
        num_steps = int(self.cfg.num_inference_steps)
        shift = float(self.cfg.timestep_shift)
        sigmas_lin = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)[:-1]
        sigmas = shift * sigmas_lin / (1.0 + (shift - 1.0) * sigmas_lin)
        timesteps = sigmas * 1000.0
        return timesteps, sigmas

    # -- exposed handles for generation.py ----------------------------------
    @property
    def timesteps(self):
        return self._timesteps

    @property
    def sigmas(self):
        return self._sigmas

    # -- lifecycle ----------------------------------------------------------
    def warmup_vae(self):
        if self.vae_runner is None:
            return
        dummy = torch.zeros(1, self.cfg.vae_config.z_dim, 1,
                            self.cfg.latent_height, self.cfg.latent_width,
                            device=self.device, dtype=self.dtype)
        self.vae_runner.run(dummy, True)
        self.vae_runner.run(dummy, False)
        self.vae_runner.clear()

    def reset(self):
        self.ring = G.new_ring_state()
        if self.vae_runner is not None:
            self.vae_runner.clear()

    def seed(self, seed: Optional[int] = None):
        self.noise_generator.manual_seed(int(self.cfg.seed if seed is None else seed))

    def set_prompt(self, prompt: Union[str, List[str]],
                   negative_prompt: Union[str, List[str]] = None):
        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=False, num_videos_per_prompt=1,
            max_sequence_length=self.cfg.max_sequence_length, device=self.device,
        )
        self.dit_runner.encode(prompt_embeds)
        self._prompt_set = True
        return prompt_embeds

    def init_session(self, prompt: Union[str, List[str]]):
        self.reset()
        self.seed()
        self.set_prompt(prompt)

    def maybe_scene_cut(self, prompt: str):
        """Replicates LongLivePipeline.update_prompt scene-cut bookkeeping."""
        is_scene_cut = (
            isinstance(prompt, str)
            and int(self.cfg.multi_shot_rope_offset) > 0
            and G.sink_chunks(self.cfg) > 0
            and self.ring["block_idx"] > 0
            and prompt.startswith(self.cfg.scene_cut_prefix)
        )
        if is_scene_cut:
            self.ring["shot_index"] += 1
            self.ring["temporal_offset_latents"] = (
                self.ring["shot_index"] * int(self.cfg.multi_shot_rope_offset)
            )
            self.ring["scene_cut_pending"] = True

    def update_prompt(self, prompt: Union[str, List[str]]):
        self.set_prompt(prompt)
        self.maybe_scene_cut(prompt)

    # -- one chunk (reference-faithful, single GPU) -------------------------
    @torch.inference_mode()
    def step(self):
        plan = G.plan_chunk(self, self.ring)
        latents = G.initial_noise(self)
        for i in range(len(self._timesteps)):
            latents = G.denoise_one_step(self, latents, plan, i)
        G.write_clean_cache(self, latents, plan)
        frames = []
        for l in range(int(self.cfg.chunk_size)):
            is_first = (self.ring["latent_decode_count"] == 0)
            frames.append(G.decode_latent_frame(self, latents, l, is_first))
            self.ring["latent_decode_count"] += 1
        G.advance_ring(self, self.ring, plan)
        return latents, frames

    # -- split compute / decode for multi-GPU workers -----------------------
    # advance_ring touches only DiT bookkeeping (block_idx / slots), which the
    # VAE decode does not read, so advancing in step_compute (before decode) is
    # numerically identical to the reference's advance-after-decode ordering;
    # this is what lets decode[N] overlap compute[N+1] in the decoupled variant.
    @torch.inference_mode()
    def step_compute(self):
        plan = G.plan_chunk(self, self.ring)
        latents = G.initial_noise(self)
        for i in range(len(self._timesteps)):
            latents = G.denoise_one_step(self, latents, plan, i)
        G.write_clean_cache(self, latents, plan)
        G.advance_ring(self, self.ring, plan)
        return latents

    @torch.inference_mode()
    def decode_chunk(self, latents):
        frames = []
        for l in range(int(self.cfg.chunk_size)):
            frames.append(self.decode_one(latents, l))
        return frames

    @torch.inference_mode()
    def decode_one(self, latents, l):
        """Decode a single latent frame (streamed). Streaming the per-frame
        write — instead of batching all chunk_size decodes — is what gives the
        reference its low latency-to-first-frame; the mono worker writes each
        frame as soon as it is decoded."""
        is_first = (self.ring["latent_decode_count"] == 0)
        frame = G.decode_latent_frame(self, latents, l, is_first)
        self.ring["latent_decode_count"] += 1
        return frame

    def set_vae_world_size(self, ws: int):
        """tile-parallel VAE decode uses get_world_size() captured at build;
        override it (ws in {2,3,4} to tile, 1 to decode locally)."""
        self.vae_runner.vae.decoder.world_size = int(ws)
