from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.pipeline.base import BasePipeline
from wllm.serving.rt_config import RTConfig
from wllm.serving.runner.dit_runner import DiTRunner

logger = init_logger(__name__)


class LongLivePipeline(BasePipeline):
    def __init__(self, cfg: RTConfig, device: torch.device):
        self._prompt_embeds: torch.Tensor | None = None
        self._block_idx: int = 0
        self._latent_decode_count: int = 0
        self._rolling_writes: int = 0
        self._max_filled_slot: int = -1
        self._shot_index: int = 0
        self._temporal_offset_latents: int = 0
        self._pinned_slot: int = -1
        self._scene_cut_pending: bool = False
        self._noise_generator = torch.Generator(device=device)
        super().__init__(cfg, device)

    # ------------------------------------------------------------------
    # BasePipeline hooks
    # ------------------------------------------------------------------

    def _create_dit_runner(self):
        return DiTRunner(self.cfg, self.dtype, self.device)

    def _build_timestep_schedule(self):
        num_steps = int(self.cfg.num_inference_steps)
        shift = float(self.cfg.timestep_shift)
        sigmas_lin = torch.linspace(
            1.0, 0.0, num_steps + 1, dtype=torch.float32
        )[:-1]
        sigmas = shift * sigmas_lin / (1.0 + (shift - 1.0) * sigmas_lin)
        timesteps = sigmas * 1000.0
        return timesteps, sigmas

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def start_instance(self):
        self._timesteps, self._sigmas = self._build_timestep_schedule()
        self._timesteps = self._timesteps.to(self.device)
        self._sigmas = self._sigmas.to(self.device)

        # Video shared-memory buffer.
        self._video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=self.cfg.max_num_frames,
            dtype=np.uint8,
            create=True,
        )

        # Warm up VAE so the first user-visible chunk doesn't pay the
        # one-time decoder allocation cost.
        if self.vae_runner is not None:
            dummy_latent = torch.zeros(
                1, self.cfg.vae_config.z_dim, 1,
                self.cfg.latent_height, self.cfg.latent_width,
                device=self.device, dtype=self.dtype,
            )
            self.vae_runner.run(dummy_latent, True)
            self.vae_runner.run(dummy_latent, False)
            self.vae_runner.clear()

        self.reset()

    @torch.inference_mode()
    def terminate_instance(self):
        if self.dit_runner is not None:
            self.dit_runner.clear()
        if self.vae_runner is not None:
            self.vae_runner.clear()
        if getattr(self, "_video_buffer", None) is not None:
            self._video_buffer.unlink()

    @torch.inference_mode()
    def init_session(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        image_path=None,
    ):
        _ = image_path
        logger.info("LongLivePipeline.init_session: prompt=%r", prompt)
        self.reset()
        self._noise_generator.manual_seed(int(self.cfg.seed))
        self._set_prompt(prompt, negative_prompt=negative_prompt)

    @torch.inference_mode()
    def update_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
    ):
        self._set_prompt(prompt, negative_prompt=negative_prompt)

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
            logger.info(
                "LongLivePipeline.update_prompt: scene cut (shot %d), "
                "rope_offset=%d latents, block_idx=%d, prompt=%r",
                self._shot_index,
                self._temporal_offset_latents,
                self._block_idx,
                prompt,
            )
        else:
            logger.info(
                "LongLivePipeline.update_prompt: plain prompt (no prefix), "
                "block_idx=%d, prompt=%r",
                self._block_idx,
                prompt,
            )

    @torch.inference_mode()
    def reset(self):
        self._prompt_embeds = None
        self._block_idx = 0
        self._latent_decode_count = 0
        self._rolling_writes = 0
        self._max_filled_slot = -1
        self._shot_index = 0
        self._temporal_offset_latents = 0
        self._pinned_slot = -1
        self._scene_cut_pending = False
        if self.vae_runner is not None:
            self.vae_runner.clear()
        if getattr(self, "_video_buffer", None) is not None:
            self._video_buffer.clear()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _set_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
    ):
        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=self.cfg.max_sequence_length,
            device=self.device,
        )
        self._prompt_embeds = prompt_embeds
        self.dit_runner.encode(prompt_embeds)

    def _sample_noise(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(
            shape,
            device=self.device,
            dtype=self.dtype,
            generator=self._noise_generator,
        )

    def _ring_capacity_latents(self) -> int:
        chunk = int(self.cfg.chunk_size)
        return max(chunk, int(self.cfg.context_window_size) + chunk)

    def _sink_chunks(self) -> int:
        return int(self.cfg.sink_size) // int(self.cfg.chunk_size)

    @torch.inference_mode()
    def step(self):
        if self._prompt_embeds is None:
            raise RuntimeError(
                "LongLivePipeline.step() called before init_session(); a "
                "prompt must be set before generation can start."
            )

        chunk_size = int(self.cfg.chunk_size)
        kv_spatial = int(self.cfg.kv_spatial)
        chunk_tokens = chunk_size * kv_spatial
        ring_capacity_latents = self._ring_capacity_latents()
        ring_capacity_chunks = ring_capacity_latents // chunk_size
        ring_capacity_tokens = ring_capacity_chunks * chunk_tokens
        sink_chunks = self._sink_chunks()
        rolling_capacity_chunks = ring_capacity_chunks - sink_chunks
        assert rolling_capacity_chunks > 0, (
            f"cfg.sink_size leaves no room for a rolling window: "
            f"sink_chunks={sink_chunks} >= ring_capacity_chunks="
            f"{ring_capacity_chunks}; raise context_window_size or lower "
            f"sink_size."
        )

        if self._block_idx < sink_chunks:
            cache_chunk_idx = self._block_idx
        elif self._pinned_slot < 0:
            # Pre-scene-cut: simple modulo over the rolling region.
            cache_chunk_idx = sink_chunks + (
                self._rolling_writes % rolling_capacity_chunks
            )
        else:
            available_slots = [
                sink_chunks + i for i in range(rolling_capacity_chunks)
                if (sink_chunks + i) != self._pinned_slot
            ]
            assert available_slots, (
                "no rolling slots left after pinning; raise "
                "context_window_size or lower sink_size"
            )
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

        assert cache_start + chunk_tokens <= ring_capacity_tokens, (
            f"chunk write [{cache_start}, {cache_start + chunk_tokens}] "
            f"escapes ring of {ring_capacity_tokens} tokens"
        )
        assert cache_end <= ring_capacity_tokens, (
            f"cache_end {cache_end} > ring of {ring_capacity_tokens} tokens"
        )

        # Step 1: sample fresh noise for this chunk.
        noise_shape = (
            1,
            self.cfg.dit_config.out_channels,
            chunk_size,
            self.cfg.latent_height,
            self.cfg.latent_width,
        )
        latents = self._sample_noise(noise_shape)

        # Step 2: DMD few-step denoise
        num_steps = len(self._timesteps)
        for i in range(num_steps):
            t = self._timesteps[i]
            sigma_t = self._sigmas[i]
            timestep = torch.full(
                (chunk_size,),
                t.item(),
                device=self.device,
                dtype=torch.float32,
            )
            flow_pred = self.dit_runner.run(
                latents=latents,
                timestep=timestep,
                is_cache=False,
                cache_start=cache_start,
                cache_end=cache_end,
                rope_start=rope_start,
                rope_end=rope_end,
            )
            # fp64 conversion mirrors LongLive's ``_convert_flow_pred_to_x0``.
            x0_pred = (
                latents.to(torch.float64)
                - sigma_t.to(torch.float64) * flow_pred.to(torch.float64)
            ).to(latents.dtype)
            if i < num_steps - 1:
                sigma_next = self._sigmas[i + 1].to(torch.float64)
                fresh_noise = self._sample_noise(latents.shape)
                latents = (
                    (1.0 - sigma_next) * x0_pred.to(torch.float64)
                    + sigma_next * fresh_noise.to(torch.float64)
                ).to(latents.dtype)
            else:
                latents = x0_pred

        # Step 3: t=0 pass to write this chunk's *clean* K/V into the ring
        # at its (wrapping) cache slot, so subsequent chunks can attend
        # against it.
        zero_timestep = torch.zeros(
            (chunk_size,),
            device=self.device,
            dtype=torch.float32,
        )
        self.dit_runner.run(
            latents=latents,
            timestep=zero_timestep,
            is_cache=True,
            cache_start=cache_start,
            cache_end=cache_end,
            rope_start=rope_start,
            rope_end=rope_end,
        )

        # Step 4: decode and stream out one latent frame at a time so the
        # frontend sees pixels as soon as possible.
        for l in range(chunk_size):
            latent_one = latents[:, :, l:l + 1].contiguous()
            is_first = (self._latent_decode_count == 0)
            video_chunk = self.vae_runner.run(latent_one, is_first)
            self._latent_decode_count += 1
            self._video_buffer.write(video_chunk[0].cpu().numpy())

        self._max_filled_slot = new_max_filled_slot
        if self._block_idx >= sink_chunks:
            self._rolling_writes += 1
        if self._scene_cut_pending:
            self._pinned_slot = cache_chunk_idx
            self._scene_cut_pending = False
        self._block_idx += 1
