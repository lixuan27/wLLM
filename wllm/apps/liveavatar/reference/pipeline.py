import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from wllm.serving.pipeline.base import BasePipeline
from wllm.serving.runner.dit_runner import DiTRunner
from wllm.serving.runner.vae_runner import VAERunner
from wllm.serving.rt_config import RTConfig


class LiveAvatarPipeline(BasePipeline):
    """Sequential reference LiveAvatar pipeline.

    The reference implementation keeps one KV cache per denoising step and
    executes every denoising step sequentially for each latent chunk on a
    single GPU.
    """

    def __init__(self, cfg: RTConfig, device: torch.device):
        self._ref_latents = None
        self._motion_latents = None
        self._zero_cond_cache: dict[int, torch.Tensor] = {}
        self._step_kv_caches = []

        super().__init__(cfg, device)

        self._cond_prefix_tokens = int(getattr(self.cfg, "kv_cond_tokens", 0) or 0)
        self._motion_frames_raw = int(getattr(self.cfg, "motion_prefix_frames", 0) or 0)
        self._motion_frames_latent = int(getattr(self.cfg, "motion_prefix_latent_frames", 0) or 0)
        assert self._motion_frames_raw > 0 and self._motion_frames_latent > 0

        self._allocate_step_caches()

    def _allocate_step_caches(self) -> None:
        if self.dit_runner is None:
            return

        base_cache = self.dit_runner.kv_memory
        cache_cls = base_cache.__class__
        self._step_kv_caches = [base_cache]
        for _ in range(1, int(self.cfg.num_inference_steps)):
            self._step_kv_caches.append(cache_cls(self.cfg, self.device))
        self._activate_step_cache(0)

    def _activate_step_cache(self, step_idx: int) -> None:
        self.dit_runner.kv_memory = self._step_kv_caches[step_idx]

    def _sync_encoder_cache_to_all_steps(self) -> None:
        if not self._step_kv_caches:
            return

        source_cache = self._step_kv_caches[0]
        for target_cache in self._step_kv_caches[1:]:
            for layer_id in range(int(self.cfg.dit_config.num_layers)):
                encoder_k, encoder_v = source_cache.get_encoder_kv(layer_id)
                target_cache.set_encoder_kv(layer_id, encoder_k, encoder_v)

    def _create_dit_runner(self):
        return DiTRunner(self.cfg, self.dtype, self.device)

    def _create_vae_runner(self):
        return VAERunner(self.cfg, self.dtype, self.device)

    def _build_timestep_schedule(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
        scheduler.set_timesteps(self.cfg.num_inference_steps)
        return scheduler.timesteps, scheduler.sigmas

    def _allocate_step_state(self):
        return

    def _on_session_init(self):
        first_image_condition = self._session_ctx.get("first_image_condition")
        first_image_pixels = self._session_ctx.get("first_image_pixels")
        if first_image_condition is None:
            self._ref_latents = None
            self._motion_latents = None
            return

        self._ref_latents = first_image_condition[:, :, :1].detach().contiguous().clone().to(self.device, self.dtype)
        self._session_ctx["first_image_condition"] = self._ref_latents

        motion_pixels = first_image_pixels.repeat(1, 1, self._motion_frames_raw, 1, 1)
        self._motion_latents = self.vae_runner.encode(motion_pixels).to(self.device, self.dtype)

        # Prime the VAE decoder cache with the reference image up front so the
        # session's first chunk decodes warm (scale_factor_temporal frames per
        # latent) instead of the causal VAE's 1-frame cold start
        self.vae_runner.run(self._motion_latents[:, :, :7], is_first_chunk=True)

    def _on_reset(self):
        self._ref_latents = None
        self._motion_latents = None
        self._zero_cond_cache.clear()
        if self._step_kv_caches:
            self._activate_step_cache(0)

    @torch.inference_mode()
    def init_session(
        self,
        prompt=None,
        negative_prompt=None,
        image_path=None,
    ):
        self._activate_step_cache(0)
        super().init_session(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
        )
        self._sync_encoder_cache_to_all_steps()
        self._activate_step_cache(0)

    def _build_full_timestep(self, timestep_value: torch.Tensor, chunk_i: int) -> torch.Tensor:
        if chunk_i > 0:
            t_now = torch.full(
                (1, self.cfg.chunk_size),
                timestep_value,
                device=self.device,
                dtype=self._timesteps.dtype,
            )
            t_ctx = torch.full(
                (1, self.cfg.first_chunk_size + (chunk_i - 1) * self.cfg.chunk_size),
                self.cfg.stabilization_level - 1,
                device=self.device,
                dtype=self._timesteps.dtype,
            )
            return torch.cat([t_ctx, t_now], dim=1)

        return torch.full(
            (1, int(self.cfg.first_chunk_size)),
            timestep_value,
            device=self.device,
            dtype=self._timesteps.dtype,
        )

    def _chunk_global_range(self, chunk_i: int) -> tuple[int, int]:
        if chunk_i <= 0:
            return 0, int(self.cfg.first_chunk_size)

        start = int(self.cfg.first_chunk_size) + (chunk_i - 1) * int(self.cfg.chunk_size)
        end = start + int(self.cfg.chunk_size)
        return start, end

    def _get_zero_cond_latents(self, num_frames: int) -> torch.Tensor:
        if num_frames in self._zero_cond_cache:
            return self._zero_cond_cache[num_frames]

        cond = torch.zeros(
            1,
            self.cfg.dit_config.out_channels,
            num_frames,
            self.cfg.latent_height,
            self.cfg.latent_width,
            dtype=self.dtype,
            device=self.device,
        )
        self._zero_cond_cache[num_frames] = cond
        return cond

    @torch.inference_mode()
    def step(
        self,
        viewmats: torch.Tensor = None,
        Ks: torch.Tensor = None,
        action: torch.Tensor = None,
        audio_input: torch.Tensor = None,
    ):
        _ = viewmats
        _ = Ks
        _ = action

        chunk_i = int(self._session_ctx["latent_chunk_idx"])
        first_image_condition = self._session_ctx["first_image_condition"]
        if first_image_condition is not None:
            first_image_condition = first_image_condition[:, :, :1].contiguous()
            self._session_ctx["first_image_condition"] = first_image_condition

        global_start_latent, global_end_latent = self._chunk_global_range(chunk_i)
        generate_latent_num = int(global_end_latent - global_start_latent)

        current_chunk_latents = torch.randn(
            (
                1,
                int(self.cfg.dit_config.out_channels),
                generate_latent_num,
                int(self.cfg.latent_height),
                int(self.cfg.latent_width),
            ),
            device=self.device,
            dtype=torch.float32,
        ).to(dtype=self.dtype)

        kv_spatial = int(self.cfg.kv_spatial)
        cond_prefix = int(self._cond_prefix_tokens)
        cond_latents_gen = self._get_zero_cond_latents(generate_latent_num)

        noisy_ring_capacity_latents = max(
            int(self.cfg.chunk_size),
            int(self.cfg.context_window_size) + int(self.cfg.chunk_size),
        )
        noisy_ring_capacity_tokens = noisy_ring_capacity_latents * kv_spatial
        write_span_tokens = generate_latent_num * kv_spatial
        write_start_tokens = (global_start_latent % noisy_ring_capacity_latents) * kv_spatial

        assert write_start_tokens + write_span_tokens <= noisy_ring_capacity_tokens

        cache_start = cond_prefix + write_start_tokens
        cache_tokens = min(global_end_latent, noisy_ring_capacity_latents) * kv_spatial
        cache_end = cond_prefix + cache_tokens
        rope_start = cond_prefix + global_start_latent * kv_spatial
        rope_end = rope_start + write_span_tokens

        need_cond_prefill = chunk_i <= 1
        for step_idx in range(int(self.cfg.num_inference_steps)):
            self._activate_step_cache(step_idx)

            if need_cond_prefill:
                prefill_t = torch.zeros((1,), device=self.device, dtype=self._timesteps.dtype)
                self.dit_runner.run(
                    latents=current_chunk_latents,
                    timestep=prefill_t,
                    is_cache=True,
                    cache_start=0,
                    cache_end=cond_prefix,
                    rope_start=0,
                    rope_end=cond_prefix,
                    viewmats=None,
                    Ks=None,
                    action=None,
                    i2v_condition=None,
                    cond_latents=None,
                    prefill_cond=True,
                    cond_prefix_tokens=cond_prefix,
                    ref_latents=self._ref_latents,
                    motion_latents=self._motion_latents,
                    motion_frames_raw=self._motion_frames_raw,
                    motion_frames_latent=self._motion_frames_latent,
                )

            timestep_value = self._timesteps[step_idx]
            timestep = self._build_full_timestep(timestep_value, chunk_i)
            timestep_slice = timestep[:, -generate_latent_num:].flatten()

            noise_pred = self.dit_runner.run(
                latents=current_chunk_latents,
                timestep=timestep_slice,
                is_cache=False,
                cache_start=cache_start,
                cache_end=cache_end,
                rope_start=rope_start,
                rope_end=rope_end,
                viewmats=None,
                Ks=None,
                action=None,
                i2v_condition=None,
                audio_input=audio_input,
                cond_latents=cond_latents_gen,
                cond_prefix_tokens=cond_prefix,
                motion_frames_raw=self._motion_frames_raw,
                motion_frames_latent=self._motion_frames_latent,
            )

            sigma = self._sigmas[step_idx]
            sigma_next = self._sigmas[step_idx + 1]
            dt = sigma_next - sigma
            current_chunk_latents = current_chunk_latents + dt * noise_pred

        if chunk_i == 0:
            self._ref_latents = current_chunk_latents[:, :, :1].detach().clone()

        chunk_video: list[np.ndarray] = []
        for frame_i in range(int(current_chunk_latents.shape[2])):
            latent_i = current_chunk_latents[:, :, frame_i:frame_i + 1, :, :].clone()
            video_i = self.vae_runner.run(latent_i, is_first_chunk=False)
            video_i = video_i.repeat_interleave(2, dim=1)
            chunk_video.append(video_i[0].cpu().numpy())

        self._session_ctx["latent_chunk_idx"] = chunk_i + 1
        self._activate_step_cache(0)
        return np.concatenate(chunk_video, axis=0)
