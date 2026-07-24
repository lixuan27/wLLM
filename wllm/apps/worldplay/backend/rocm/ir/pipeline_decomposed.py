"""Faithful operator-level decomposition of the reference WorldPlay chunk.

`WorldPlayDecomposedPipeline` subclasses the reference `WorldPlayPipeline`
(read-only) and re-expresses the monolithic `step()` as a sequence of small
methods, one per IR operator, **without changing the math**. The reference
`step()` (wllm/apps/worldplay/reference/pipeline.py) is the line-by-line
oracle: each method below is a verbatim slice of it, so running the ops in
order reproduces the reference frames exactly.

State that persists across chunks lives on the pipeline object exactly as in
the reference: `self._latents` (fp32 latent store), the DiT KV cache (inside
`self.dit_runner.kv_memory`), the VAE temporal causal cache (inside
`self.vae_runner.vae`), the `self._viewmats/_Ks/_action` accumulators, and the
camera pose accumulators `self._T/_C_inv`. Per-chunk scratch lives in
`self._scratch`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

import wllm.kernels_t

from wllm.apps.worldplay.reference.pipeline import WorldPlayPipeline
from wllm.serving.utils.fov import select_mem_frames_wan


class WorldPlayDecomposedPipeline(WorldPlayPipeline):
    def start_instance(self):
        super().start_instance()
        # camera pose accumulators (mirrors worker.WorldPlayWorker.T / C_inv)
        self._T = torch.eye(4, dtype=torch.float32)
        self._C_inv = torch.zeros((4, 4), dtype=torch.float32)
        self._scratch = {}

    def reset(self):
        super().reset()
        if hasattr(self, "_T"):
            self._T[:] = torch.eye(4, dtype=torch.float32)
            self._C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
            self._scratch = {}

    # ------------------------------------------------------------------
    # op: camera_decode  (worker.predict camera-kernel block)
    # ------------------------------------------------------------------
    def op_camera_decode(self, actions: np.ndarray):
        curr_pose = actions
        translation_codes, rotation_codes = (
            wllm.kernels_t.camera_action.decode_combined_actions(curr_pose)
        )
        curr_viewmats, curr_Ks, curr_action = (
            wllm.kernels_t.camera_action.motions_to_matrix_with_rotation(
                translation_codes,
                rotation_codes,
                self._T,
                self._C_inv,
                first_chunk=(self._session_ctx["latent_chunk_idx"] == 0),
            )
        )
        curr_action = wllm.kernels_t.camera_action.compute_worldplay_combined_label(
            curr_action, rotation_codes,
        )
        viewmats = curr_viewmats.unsqueeze(0)
        Ks = curr_Ks.unsqueeze(0)
        action = curr_action.unsqueeze(0)
        return viewmats, Ks, action

    # ------------------------------------------------------------------
    # op: prep  (pipeline.step lines 36-58: store conditioning, pick latents)
    # ------------------------------------------------------------------
    def op_prep(self, viewmats, Ks, action):
        chunk_i = self._session_ctx["latent_chunk_idx"]
        first_image_condition = self._session_ctx["first_image_condition"]

        start_idx = chunk_i * self.cfg.chunk_size
        end_idx = start_idx + self.cfg.chunk_size

        self._viewmats[:, start_idx:end_idx, ...] = viewmats
        self._Ks[:, start_idx:end_idx, ...] = Ks
        self._action[:, start_idx:end_idx, ...] = action

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

        self._scratch = {
            "chunk_i": chunk_i,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "already_generate_num": already_generate_num,
            "generate_latent_num": generate_latent_num,
            "latents_curr": latents_curr,
            "first_image_condition": first_image_condition,
            "all_window_latent_num": latents_curr.shape[2],
        }

    # ------------------------------------------------------------------
    # op: select_mem  (pipeline.step lines 61-77) -- chunk_i > 0 only
    # ------------------------------------------------------------------
    def op_select_mem(self):
        chunk_i = self._scratch["chunk_i"]
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
        self._scratch["selected_frame_indices"] = selected_frame_indices

    def _timestep_for(self, t):
        """pipeline.step lines 83-103 -- build the per-frame timestep vector."""
        chunk_i = self._scratch["chunk_i"]
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
        return timestep

    # ------------------------------------------------------------------
    # op: kv_fill  (pipeline.step lines 107-132) -- chunk_i > 0, once
    # ------------------------------------------------------------------
    def op_kv_fill(self):
        selected = self._scratch["selected_frame_indices"]
        latents_curr = self._scratch["latents_curr"]
        t = self._timesteps[0]
        timestep = self._timestep_for(t)

        latents_cache = latents_curr[:, :, selected].clone()
        t_cache = timestep[:, selected]
        timestep_cache = t_cache.flatten()
        action_cache = self._action[:, selected]
        viewmats_cache = self._viewmats[:, selected]
        Ks_cache = self._Ks[:, selected]
        i2v_cond_cache = self._get_i2v_condition_slice(selected)

        kv_start_rope = 0
        kv_end_rope = len(selected) * self.cfg.kv_spatial
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
        )

    # ------------------------------------------------------------------
    # op: denoise_i  (pipeline.step lines 134-178) -- gen fwd + Euler
    # ------------------------------------------------------------------
    def op_denoise(self, i: int):
        chunk_i = self._scratch["chunk_i"]
        latents_curr = self._scratch["latents_curr"]
        selected = self._scratch.get("selected_frame_indices", None)
        generate_latent_num = self._scratch["generate_latent_num"]
        all_window_latent_num = self._scratch["all_window_latent_num"]
        first_image_condition = self._scratch["first_image_condition"]

        t = self._timesteps[i]
        timestep = self._timestep_for(t)

        if selected is not None:
            now_window_latent_num = len(selected) + self.cfg.chunk_size
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
        )

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

    # ------------------------------------------------------------------
    # op: finalize  (pipeline.step line 180) -- write latents back to store
    # ------------------------------------------------------------------
    def op_finalize(self):
        already_generate_num = self._scratch["already_generate_num"]
        latents_curr = self._scratch["latents_curr"]
        start_idx = self._scratch["start_idx"]
        end_idx = self._scratch["end_idx"]
        self._latents[:, :, :already_generate_num, :, :] = latents_curr
        # the finalized latents for THIS chunk (handed to VAE as data, not state)
        return start_idx, end_idx

    # ------------------------------------------------------------------
    # op: vae_decode_j  (pipeline.step lines 183-186) -- one latent frame
    # ------------------------------------------------------------------
    def op_vae_decode(self, l_i: int) -> np.ndarray:
        latent_i = self._latents[:, :, l_i:l_i + 1, :, :].clone()
        video_i = self.vae_runner.run(latent_i, (l_i == 0))
        return video_i[0].cpu().numpy()

    def op_advance_chunk(self):
        self._session_ctx["latent_chunk_idx"] = self._session_ctx["latent_chunk_idx"] + 1
