from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from wllm.serving.pipeline.base import BasePipeline
from wllm.serving.runner.dit_runner import DiTRunner
from wllm.serving.rt_config import RTConfig
from wllm.serving.utils.fov import select_mem_frames_wan


class WorldPlayPipeline(BasePipeline):
    """Reference single-GPU WorldPlay pipeline."""

    def __init__(self, cfg: RTConfig, device: torch.device):
        super().__init__(cfg, device)

    def _create_dit_runner(self):
        return DiTRunner(self.cfg, self.dtype, self.device)

    def _build_timestep_schedule(self):
        # Few-step distilled schedule used by the WorldPlay-5B checkpoint.
        timesteps = torch.tensor([1000.0000, 960.0000, 888.8889, 727.2728])
        sigmas = torch.tensor([1.0000, 0.9600, 0.8889, 0.7273, 0.0])
        return timesteps, sigmas

    @torch.inference_mode()
    def step(
        self,
        viewmats: torch.Tensor = None,
        Ks: torch.Tensor = None,
        action: torch.Tensor = None,
    ) -> Optional[np.ndarray]:
        chunk_i = self._session_ctx["latent_chunk_idx"]
        first_image_condition = self._session_ctx["first_image_condition"]

        start_idx = chunk_i * self.cfg.chunk_size
        end_idx = start_idx + self.cfg.chunk_size

        self._viewmats[:, start_idx:end_idx, ...] = viewmats
        self._Ks[:, start_idx:end_idx, ...] = Ks
        self._action[:, start_idx:end_idx, ...] = action

        selected_frame_indices = None
        if chunk_i == 0:
            already_generate_num = self.cfg.first_chunk_size
            generate_latent_num = self.cfg.first_chunk_size
            self._latents[:, :, :1] = first_image_condition
            latents_curr = self._latents[:, :, :already_generate_num].to(
                device=self.device, dtype=self.dtype
            )
        else:
            already_generate_num = chunk_i * self.cfg.chunk_size + self.cfg.first_chunk_size
            latents_curr = self._latents[:, :, :already_generate_num].to(
                device=self.device, dtype=self.dtype
            )
            generate_latent_num = self.cfg.chunk_size

            current_frame_idx = chunk_i * self.cfg.chunk_size

            if (
                current_frame_idx >= self.cfg.context_window_size
                and current_frame_idx < self.cfg.max_num_actions
            ):
                selected_frame_indices = select_mem_frames_wan(
                    self._viewmats[0],
                    current_frame_idx,
                    memory_frames=self.cfg.context_window_size,
                    temporal_context_size=(self.cfg.context_window_size - self.cfg.chunk_size),
                    pred_latent_size=self.cfg.chunk_size,
                    points_local=self.points_local,
                    device=self.device,
                )
            else:
                selected_frame_indices = list(range(0, current_frame_idx))

        for i, t in enumerate(self._timesteps):
            t_value = t.item()
            extra_kwargs = self._dit_run_extra_kwargs(t_value)

            if chunk_i > 0:
                t_now = torch.full(
                    (1, self.cfg.chunk_size), t,
                    device=self.device, dtype=self._timesteps.dtype,
                )
                t_ctx = torch.full(
                    (1, self.cfg.first_chunk_size + (chunk_i - 1) * self.cfg.chunk_size),
                    self.cfg.stabilization_level - 1,
                    device=self.device, dtype=self._timesteps.dtype,
                )
                timestep = torch.cat([t_ctx, t_now], dim=1)
            else:
                t_now = torch.full(
                    (1, self.cfg.chunk_size - 1), t,
                    device=self.device, dtype=self._timesteps.dtype,
                )
                t_ctx = torch.full(
                    (1, 1), self.cfg.stabilization_level - 1,
                    device=self.device, dtype=self._timesteps.dtype,
                )
                timestep = torch.cat([t_ctx, t_now], dim=1)

            all_window_latent_num = latents_curr.shape[2]

            # KV cache fill (first denoising step of subsequent chunks).
            if chunk_i > 0 and i == 0:
                latents_cache = latents_curr[:, :, selected_frame_indices].clone()
                t_cache = timestep[:, selected_frame_indices]
                timestep_cache = t_cache.flatten()
                action_cache = self._action[:, selected_frame_indices]
                viewmats_cache = self._viewmats[:, selected_frame_indices]
                Ks_cache = self._Ks[:, selected_frame_indices]
                i2v_cond_cache = self._get_i2v_condition_slice(selected_frame_indices)

                kv_start_rope = 0
                kv_end_rope = len(selected_frame_indices) * self.cfg.kv_spatial
                self.dit_runner.run(
                    latents=latents_cache,
                    timestep=timestep_cache,
                    is_cache=True,
                    cache_start=kv_start_rope,
                    cache_end=kv_end_rope,
                    rope_start=kv_start_rope,
                    rope_end=kv_end_rope,
                    viewmats=viewmats_cache,
                    Ks=Ks_cache,
                    action=action_cache,
                    i2v_condition=i2v_cond_cache,
                    **extra_kwargs,
                )

            if selected_frame_indices is not None:
                now_window_latent_num = len(selected_frame_indices) + self.cfg.chunk_size
            else:
                now_window_latent_num = self.cfg.chunk_size

            latent_model_input = latents_curr[:, :, -generate_latent_num:].clone()
            timestep_slice = timestep[:, -generate_latent_num:].flatten()

            gen_frame_start = all_window_latent_num - generate_latent_num
            gen_frame_end = all_window_latent_num
            gen_frame_indices = list(range(gen_frame_start, gen_frame_end))
            i2v_cond_gen = self._get_i2v_condition_slice(gen_frame_indices)

            generate_rope_start = (now_window_latent_num - generate_latent_num) * self.cfg.kv_spatial
            generate_rope_end = now_window_latent_num * self.cfg.kv_spatial
            noise_pred = self.dit_runner.run(
                latents=latent_model_input,
                timestep=timestep_slice,
                is_cache=False,
                cache_start=generate_rope_start,
                cache_end=generate_rope_end,
                rope_start=generate_rope_start,
                rope_end=generate_rope_end,
                viewmats=self._viewmats[:, gen_frame_start:gen_frame_end],
                Ks=self._Ks[:, gen_frame_start:gen_frame_end],
                action=self._action[:, gen_frame_start:gen_frame_end],
                i2v_condition=i2v_cond_gen,
                **extra_kwargs,
            )

            # Euler step: x_{t-1} = x_t + dt * noise_pred
            sigma = self._sigmas[i]
            sigma_next = self._sigmas[i + 1]
            dt = sigma_next - sigma
            if chunk_i == 0:
                prev_sample = latent_model_input + dt * noise_pred
                if first_image_condition is not None:
                    latents_curr[:, :, -self.cfg.first_chunk_size + 1:] = prev_sample[:, :, 1:]
                else:
                    latents_curr[:, :, -self.cfg.first_chunk_size:] = prev_sample
            else:
                noise_pred_chunk = noise_pred[:, :, -self.cfg.chunk_size:]
                latents_curr_chunk = latent_model_input[:, :, -self.cfg.chunk_size:]
                latent_curr_pred_chunk = latents_curr_chunk + dt * noise_pred_chunk
                latents_curr[:, :, -self.cfg.chunk_size:] = latent_curr_pred_chunk

        self._latents[:, :, :already_generate_num, :, :] = latents_curr

        chunk_video: list[np.ndarray] = []
        for l_i in range(start_idx, end_idx):
            latent_i = self._latents[:, :, l_i:l_i + 1, :, :].clone()
            video_i = self.vae_runner.run(latent_i, (l_i == 0))
            chunk_video.append(video_i[0].cpu().numpy())

        self._session_ctx["latent_chunk_idx"] = chunk_i + 1
        if not chunk_video:
            return None
        return np.concatenate(chunk_video, axis=0)
