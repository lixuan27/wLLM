"""
This is the *naive* version of the Krea-Realtime v2v pipeline used by the
reference Krea+SAM backend.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from wllm.serving.logger import init_logger
from wllm.serving.pipeline.base import BasePipeline
from wllm.serving.rt_config import RTConfig
from wllm.serving.runner.dit_runner import DiTRunner

logger = init_logger(__name__)


class KreaSAMPipeline(BasePipeline):
    """Reference single-GPU Krea-Realtime pipeline.

    State held across ``step()`` calls within a session:
      - ``_clean_latent_context`` : recent already-denoised latent
        chunks that condition the DiT through the prefix KV cache.
      - ``_block_idx`` : monotonically-incremented chunk index since
        session start. Drives the streaming VAE encoder's per-chunk
        frame count + ``stream`` flag and the VAE decoder's first-chunk
        flag.

    The streaming input-VAE encoder carries its own causal temporal
    cache across chunks (inside ``VAERunner``), so this pipeline holds
    no rolling pixel buffer of its own.
    """

    def __init__(self, cfg: RTConfig, device: torch.device):
        self.scheduler: FlowMatchEulerDiscreteScheduler | None = None
        self._zero_padded_timesteps: torch.Tensor | None = None
        self._denoise_timesteps: torch.Tensor | None = None
        self._prompt_embeds: torch.Tensor | None = None
        self._clean_latent_context: torch.Tensor | None = None
        self._noise_generator = torch.Generator(device=device)
        self._block_idx = 0
        super().__init__(cfg, device)

    def _create_dit_runner(self):
        return DiTRunner(self.cfg, self.dtype, self.device)

    def input_frames_for_next_step(self) -> int:
        """Number of raw input frames the next ``step()`` expects.

        The causal WAN VAE encoder maps the first latent frame to a
        single pixel frame and every later latent frame to
        ``scale_factor_temporal`` pixel frames. So the very first chunk
        needs ``1 + (chunk_size - 1) * scale_factor_temporal`` frames,
        and every streaming chunk after it needs
        ``chunk_size * scale_factor_temporal`` frames.
        """
        scale_t = int(self.cfg.vae_config.scale_factor_temporal)
        chunk_size = int(self.cfg.chunk_size)
        if self._block_idx == 0:
            return 1 + max(0, chunk_size - 1) * scale_t
        return chunk_size * scale_t

    # ------------------------------------------------------------------
    # timestep schedule (flow-matching, sigma-blend renoise)
    # ------------------------------------------------------------------

    def _build_timestep_schedule(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=float(self.cfg.timestep_shift),
        )
        scheduler.set_timesteps(1000)
        zero_padded = torch.cat(
            [
                scheduler.timesteps.to(torch.float32).cpu(),
                torch.tensor([0], dtype=torch.float32),
            ],
            dim=0,
        )
        denoising_steps = self._compute_denoising_schedule(
            zero_padded.to(self.device),
            float(self.cfg.denoising_strength),
            int(self.cfg.num_inference_steps),
        )
        return denoising_steps, torch.zeros(
            denoising_steps.shape[0] + 1,
            device=denoising_steps.device,
            dtype=denoising_steps.dtype,
        )

    @staticmethod
    def _compute_denoising_schedule(
        zero_padded_timesteps: torch.Tensor,
        denoising_strength: float,
        steps: int,
    ) -> torch.Tensor:
        indices = torch.linspace(
            denoising_strength * 1000.0,
            0.0,
            steps,
            dtype=torch.float32,
            device=zero_padded_timesteps.device,
        ).to(torch.long)
        return zero_padded_timesteps[1000 - indices]

    # ------------------------------------------------------------------
    # noise sampling (single-rank, no broadcast)
    # ------------------------------------------------------------------

    def _sample_noise(self, shape: torch.Size) -> torch.Tensor:
        return torch.randn(
            shape,
            device=self.device,
            dtype=self.dtype,
            generator=self._noise_generator,
        )

    # ------------------------------------------------------------------
    # clean-latent context
    # ------------------------------------------------------------------

    def _current_context_latents(self) -> Optional[torch.Tensor]:
        if self._clean_latent_context is None or self._clean_latent_context.shape[2] == 0:
            return None

        max_ctx = int(self.cfg.context_window_size)
        context = self._clean_latent_context
        if context.shape[2] <= max_ctx:
            return context

        if self.cfg.keep_first_frame and max_ctx > 1:
            first = context[:, :, :1]
            tail = context[:, :, -max(0, max_ctx - 1):]
            return torch.cat([first, tail], dim=2)

        return context[:, :, -max_ctx:]

    def _append_clean_latents(self, new_latents: torch.Tensor) -> None:
        if self._clean_latent_context is None:
            combined = new_latents.detach().clone()
        else:
            combined = torch.cat(
                [self._clean_latent_context, new_latents.detach()], dim=2,
            )

        max_keep = int(self.cfg.context_window_size)
        if self.cfg.keep_first_frame and max_keep > 1 and combined.shape[2] > max_keep:
            combined = torch.cat(
                [combined[:, :, :1], combined[:, :, -(max_keep - 1):]], dim=2,
            )
        elif combined.shape[2] > max_keep:
            combined = combined[:, :, -max_keep:]

        self._clean_latent_context = combined.contiguous()

    # ------------------------------------------------------------------
    # prefix-cache fill for clean context, then per-step denoise
    # ------------------------------------------------------------------

    def _fill_clean_context_cache(self, clean_context: Optional[torch.Tensor]) -> int:
        if clean_context is None or clean_context.shape[2] == 0:
            return 0

        context_latents = clean_context.to(device=self.device, dtype=self.dtype)
        context_frames = int(context_latents.shape[2])
        context_tokens = context_frames * int(self.cfg.kv_spatial)
        zero_timestep = torch.zeros(
            (context_frames,),
            device=self.device,
            dtype=self._denoise_timesteps.dtype,
        )

        self.dit_runner.run(
            latents=context_latents,
            timestep=zero_timestep,
            is_cache=True,
            cache_start=0,
            cache_end=context_tokens,
            rope_start=0,
            rope_end=context_tokens,
        )
        return context_tokens

    @staticmethod
    def _latents_to_scheduler_frames(latents: torch.Tensor) -> torch.Tensor:
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _scheduler_frames_to_latents(frames: torch.Tensor) -> torch.Tensor:
        return frames.permute(0, 2, 1, 3, 4).contiguous()

    def _renoise(
        self,
        denoised_latents: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scheduler_frames = self._latents_to_scheduler_frames(denoised_latents)
        noise = self._sample_noise(scheduler_frames.shape)
        # FlowMatchEulerDiscreteScheduler in this diffusers version exposes
        # `scale_noise`, not `add_noise`. Use Krea's sigma-based blend
        # directly: x_t = (1 - sigma) * x_0 + sigma * eps.
        sigma = next_timestep.to(device=self.device, dtype=torch.float64) / 1000.0
        while sigma.ndim < scheduler_frames.ndim:
            sigma = sigma.unsqueeze(-1)
        renoised = (
            (1.0 - sigma) * scheduler_frames.to(torch.float64)
            + sigma * noise.to(torch.float64)
        ).to(scheduler_frames.dtype)
        return self._scheduler_frames_to_latents(renoised)

    def _denoise_latents(
        self,
        noisy_latents: torch.Tensor,
        context_tokens: int,
    ) -> torch.Tensor:
        current = noisy_latents.to(device=self.device, dtype=self.dtype)
        chunk_frames = int(current.shape[2])
        chunk_tokens = chunk_frames * int(self.cfg.kv_spatial)

        for idx, timestep_value in enumerate(self._denoise_timesteps):
            timestep = torch.full(
                (chunk_frames,),
                timestep_value,
                device=self.device,
                dtype=self._denoise_timesteps.dtype,
            )
            flow_pred = self.dit_runner.run(
                latents=current,
                timestep=timestep,
                is_cache=False,
                cache_start=context_tokens,
                cache_end=context_tokens + chunk_tokens,
                rope_start=context_tokens,
                rope_end=context_tokens + chunk_tokens,
            )
            # Krea outputs flow-matching velocity v = noise - x0; convert
            # to the x0 prediction expected by renoise / VAE decode.
            sigma_t = timestep_value.to(torch.float64) / 1000.0
            x0_pred = (
                current.to(torch.float64) - sigma_t * flow_pred.to(torch.float64)
            ).to(current.dtype)

            if idx < (len(self._denoise_timesteps) - 1):
                current = self._renoise(x0_pred, self._denoise_timesteps[idx + 1])
            else:
                current = x0_pred

        return current

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def start_instance(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=float(self.cfg.timestep_shift),
        )
        self.scheduler.set_timesteps(1000)
        self._zero_padded_timesteps = torch.cat(
            [
                self.scheduler.timesteps.to(torch.float32).cpu(),
                torch.tensor([0], dtype=torch.float32),
            ],
            dim=0,
        ).to(self.device)
        self._denoise_timesteps = self._compute_denoising_schedule(
            self._zero_padded_timesteps,
            float(self.cfg.denoising_strength),
            int(self.cfg.num_inference_steps),
        )

        if self.vae_runner is not None:
            dummy_latent = torch.zeros(
                1,
                self.cfg.vae_config.z_dim,
                1,
                self.cfg.latent_height,
                self.cfg.latent_width,
                device=self.device,
                dtype=self.dtype,
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

    @torch.inference_mode()
    def init_session(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        image_path=None,
    ):
        _ = image_path  # Krea v2v has no first-image conditioning.
        self.reset()
        self._noise_generator.manual_seed(int(self.cfg.seed))

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

    @torch.inference_mode()
    def reset(self):
        self._prompt_embeds = None
        self._clean_latent_context = None
        self._block_idx = 0
        if self.vae_runner is not None:
            # Also resets the streaming input-encoder cache (see VAERunner.clear).
            self.vae_runner.clear()

    # ------------------------------------------------------------------
    # one chunk of v2v: encode -> renoise -> denoise -> decode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def step(self, input_frames: torch.Tensor) -> Optional[np.ndarray]:
        """Run one v2v denoising chunk.

        Args:
            input_frames: ``[T, C, H, W]`` float-in-[-1, 1] pixel frames.
                          ``T`` must be ``input_frames_for_next_step()``:
                          ``1 + (chunk_size - 1) * scale_factor_temporal``
                          for the first chunk, then
                          ``chunk_size * scale_factor_temporal``.

        Returns:
            ``[T_out, H, W, 3]`` uint8 RGB decoded video for this chunk,
            or ``None`` if the streaming encoder did not yield a full
            chunk of latent frames.
        """
        if input_frames.ndim != 4:
            raise ValueError(
                f"expected input_frames in [T, C, H, W], got shape={tuple(input_frames.shape)}"
            )

        expected_frames = self.input_frames_for_next_step()
        if int(input_frames.shape[0]) != expected_frames:
            logger.warning(
                "Krea v2v got %d raw frames, expected %d for chunk %d",
                int(input_frames.shape[0]), expected_frames, self._block_idx,
            )

        # Streaming causal VAE encode. The encoder's temporal cache carries
        # across chunks (stream=True after the first), so this chunk's
        # latents correspond exactly to the frames passed in now
        pixels = input_frames.permute(1, 0, 2, 3).unsqueeze(0).to(
            device=self.device, dtype=self.dtype,
        )
        input_latents = self.vae_runner.encode(
            pixels, stream=self._block_idx > 0,
        ).to(device=self.device, dtype=self.dtype)
        chunk_frames = int(self.cfg.chunk_size)
        if int(input_latents.shape[2]) < chunk_frames:
            logger.info(
                "Krea priming input encoder: raw_frames=%d latent_frames=%d target=%d",
                int(input_frames.shape[0]),
                int(input_latents.shape[2]),
                chunk_frames,
            )
            return None
        input_latents = input_latents[:, :, -chunk_frames:].contiguous()

        init_strength = float(self._denoise_timesteps[0].item()) / 1000.0
        noise = self._sample_noise(input_latents.shape)
        noisy_latents = input_latents * (1.0 - init_strength) + noise * init_strength

        clean_context = self._current_context_latents()
        context_tokens = self._fill_clean_context_cache(clean_context)
        denoised_latents = self._denoise_latents(noisy_latents, context_tokens)
        self._append_clean_latents(denoised_latents)

        chunk_video: list[np.ndarray] = []
        for frame_i in range(int(denoised_latents.shape[2])):
            latent_i = denoised_latents[:, :, frame_i:frame_i + 1, :, :].clone()
            is_first = (self._block_idx == 0 and frame_i == 0)
            decoded_i = self.vae_runner.run(latent_i, is_first)
            chunk_video.append(decoded_i[0].cpu().numpy())

        self._block_idx += 1
        return np.concatenate(chunk_video, axis=0)
