from typing import List, Optional, Union
from abc import ABC, abstractmethod
import torch
import numpy as np
from transformers import AutoTokenizer
from wllm.serving.runner.vae_runner import VAERunner
from wllm.serving.runner.text_encoder_runner import TextEncoderRunner
from wllm.serving.utils.fov import select_mem_frames_wan, generate_points_in_sphere
from wllm.serving.utils.prompt_utils import prompt_clean
from wllm.serving.utils.dtype import parse_dtype_getattr
from wllm.serving.utils.image_process import resize_center_crop
from wllm.serving.distributed.parallel_state import get_world_rank
from wllm.serving.distributed.communication_op import global_barrier, global_broadcast
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.rt_config import RTConfig
from diffusers.utils.torch_utils import randn_tensor
import gc


class BasePipeline(ABC):
    def __init__(self, cfg: RTConfig, device: torch.device):
        self.cfg = cfg
        self.dtype = parse_dtype_getattr(self.cfg.dtype)
        self.device = device
        self.vae_runner = self._create_vae_runner()
        self.text_encoder_runner = self._create_text_encoder_runner()
        self.dit_runner = self._create_dit_runner()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_path)

    @abstractmethod
    def _create_dit_runner(self):
        pass

    def _create_text_encoder_runner(self):
        return TextEncoderRunner(self.cfg, self.dtype, self.device)

    def _create_vae_runner(self):
        return VAERunner(self.cfg, self.dtype, self.device)

    @abstractmethod
    def _build_timestep_schedule(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def _get_i2v_condition_slice(self, frame_indices):
        return None

    def _dit_run_extra_kwargs(self, timestep_value: float) -> dict:
        return {}

    def _allocate_step_state(self):
        self.points_local = generate_points_in_sphere(50000, 8.0).to(self.device)
        self._viewmats: torch.Tensor = torch.zeros(
            1, self.cfg.max_num_actions, 4, 4, device=self.device, dtype=torch.float32
        )
        self._Ks: torch.Tensor = torch.zeros(
            1, self.cfg.max_num_actions, 3, 3, device=self.device, dtype=torch.float32
        )
        self._action: torch.Tensor = torch.zeros(
            1, self.cfg.max_num_actions, device=self.device, dtype=torch.float32
        )

    def _on_session_init(self):
        pass

    def _on_reset(self):
        pass

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder_runner.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

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
            text_input_ids.to(device), mask.to(device)
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shape = (
            1,
            self.cfg.dit_config.out_channels,
            self.cfg.max_num_actions,
            self.cfg.latent_height,
            self.cfg.latent_width,
        )

        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)
        global_broadcast(latents, src=0)
        return latents

    @property
    def guidance_scale(self):
        return self.cfg.guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self.cfg.guidance_scale > 1.0

    @torch.inference_mode()
    def start_instance(self):
        self._timesteps, self._sigmas = self._build_timestep_schedule()
        self._timesteps = self._timesteps.to(self.device)
        self._sigmas = self._sigmas.to(self.device)
        self._latents: torch.Tensor = self.prepare_latents(torch.float32)
        self._num_timesteps: int = len(self._timesteps)
        self._allocate_step_state()

        self._video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=self.cfg.max_num_frames,
            dtype=np.uint8,
            create=(get_world_rank() == 0),
        )

        if self.vae_runner is not None:
            # warm up vae
            dummy_latents = [
                torch.zeros(
                    1, self.cfg.vae_config.z_dim, 1,
                    self.cfg.latent_height, self.cfg.latent_width,
                    device=self.device, dtype=self.dtype,
                )
                for _ in range(3)
            ]
            for i, dummy_latent in enumerate(dummy_latents):
                self.vae_runner.run(dummy_latent, (i == 0))
            self.vae_runner.clear()
        self._session_ctx = {
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "first_image_condition": None,
            "first_image_pixels": None,
            "current_frame_idx": 0,
            "latent_chunk_idx": 0,
        }

    @torch.inference_mode()
    def terminate_instance(self):
        if self.dit_runner is not None:
            self.dit_runner.clear()
        if get_world_rank() == 0:
            self._video_buffer.unlink()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def init_session(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        image_path=None,
    ):
        self.reset()
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=self.cfg.max_sequence_length,
            device=self.device,
        )

        assert negative_prompt_embeds is None
        first_image = resize_center_crop(
            image_path, (self.cfg.width, self.cfg.height), normalize=True, CHW=True
        )
        first_image = torch.from_numpy(first_image)
        first_image = first_image.to(self.device).to(self.dtype)
        first_image = first_image.unsqueeze(0).unsqueeze(2)  # [B C T H W]
        first_image_condition = self.vae_runner.encode(first_image)

        self._session_ctx["prompt_embeds"] = prompt_embeds
        self._session_ctx["negative_prompt_embeds"] = negative_prompt_embeds
        self._session_ctx["first_image_condition"] = first_image_condition
        self._session_ctx["first_image_pixels"] = first_image
        self._session_ctx["latent_chunk_idx"] = 0
        self._session_ctx["current_frame_idx"] = 0
        if self.dit_runner is not None:
            self.dit_runner.encode(prompt_embeds)

        self._on_session_init()

    @torch.inference_mode()
    def step(
        self,
        viewmats: torch.Tensor = None,
        Ks: torch.Tensor = None,
        action: torch.Tensor = None,
    ):
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

            # KV cache fill (first denoising step of subsequent chunks)
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

        for l_i in range(start_idx, end_idx):
            latent_i = self._latents[:, :, l_i:l_i + 1, :, :].clone()
            video_i = self.vae_runner.run(latent_i, (l_i == 0))
            if get_world_rank() == 0:
                vl_i = video_i.shape[1]
                voffset = self._session_ctx["current_frame_idx"]
                self._video_buffer.write(video_i[0].cpu().numpy())
                self._session_ctx["current_frame_idx"] = voffset + vl_i

        self._session_ctx["latent_chunk_idx"] = self._session_ctx["latent_chunk_idx"] + 1

    @torch.inference_mode()
    def reset(self):
        self._session_ctx["latent_chunk_idx"] = 0
        self._session_ctx["current_frame_idx"] = 0
        self._session_ctx["first_image_condition"] = None
        self._session_ctx["first_image_pixels"] = None
        self._latents: torch.Tensor = self.prepare_latents(torch.float32)
        if self.vae_runner is not None:
            self.vae_runner.clear()
        if get_world_rank() == 0:
            self._video_buffer.clear()
        self._on_reset()
        global_barrier()
