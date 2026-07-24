"""Reusable multi-rank WorldPlay worker (sequence-parallel + tiled VAE).

This is the workhorse behind most agent variants. It is launched once per GPU
by ``torchrun`` (so RANK / WORLD_SIZE / LOCAL_RANK are set), brings up the
shared-runtime ``torch.distributed`` world with a sequence-parallel group of
size ``world_size`` (``sp_size = world_size``, ``tp_size = 1``), and runs the
*reference* WorldPlay pipeline on every rank.

Why this is correct as a transformation of the reference:

* The WorldPlay DiT (``WorldPlayWanTransformer3DModel``) already implements
  DeepSpeed-Ulysses sequence parallelism internally: it shards the latent
  frame/token sequence across the SP group, does an all-to-all so each rank
  owns a head-shard for full-sequence attention, then all-gathers the output.
  So with ``sp_size=N`` the *same* math runs, only split across N GPUs — a pure
  reduction-order change (bf16 noise), not a data-dependency change.
* The Wan VAE decoder auto-tiles along width across the world group when
  ``world_size>1`` (``split_tile``/``gather_tile`` with halo), so the VAE decode
  is also distributed — again the same output, different reduction order.
* Every rank runs the *identical* control flow, driven by rank-0 broadcasts of
  the per-chunk action codes + a control opcode, so all the collective ops
  (SP all-to-all, VAE all-gather, latent broadcast, barriers) stay in lockstep.
  Rank 0 alone owns the frontend-facing IPC buffers and writes video frames.

Flags:
* ``stream_vae``: write each latent's decoded frame burst to the video buffer
  the moment it is produced, instead of accumulating the whole chunk and writing
  once. Pure scheduling change of *when* frames are published (same frames),
  which cuts action-to-first-frame latency and smooths inter-frame gaps.

The denoise + VAE math is copied verbatim from the reference
``WorldPlayPipeline.step`` (wllm/apps/worldplay/reference/pipeline.py) so numerics
match; the only differences are (a) it runs on N ranks via the model's built-in
SP, (b) rank 0 owns IPC, (c) optional per-latent streaming of the VAE output.
"""

from __future__ import annotations

import os
import time
import argparse
from typing import List, Optional

import numpy as np
import torch

import wllm.kernels_t
from wllm.serving.logger import init_logger
from wllm.serving.rt_config import RTConfig
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.apps.worldplay.reference.pipeline import WorldPlayPipeline
from wllm.serving.utils.fov import select_mem_frames_wan
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options
from wllm.serving.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    get_world_rank, get_world_group, get_world_size, get_sp_world_size,
    destroy_model_parallel, destroy_distributed_environment,
)
from wllm.serving.distributed.communication_op import warmup_sequence_parallel_communication
from wllm.apps.worldplay.backend.cuda.runtime.dist_timeout import patch_pg_timeout

logger = init_logger(__name__)
set_torch_options()

# control opcodes broadcast from rank0 to all ranks
_START, _TERMINATE, _RESET, _STEP = 1, 2, 3, 4


class SPWorldPlayWorker:
    def __init__(self, cfg_path: str, stream_vae: bool = False, vae_mode: str = "tiled",
                 vae_batch: bool = False):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)
        self.stream_vae = bool(stream_vae)
        self.vae_batch = bool(vae_batch)
        if self.vae_batch:
            self.stream_vae = False  # batched decode writes the whole chunk once

        # --- distributed world: sp_size = world_size, tp = 1 ---
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=world_size)
        self.rank = get_world_rank()
        self.world_size = get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        # VAE placement. "tiled": width-tile the decoder across the world group
        # (split_tile/gather_tile, supported for world_size in {2,3,4}). "rank0":
        # decode on rank 0 only (the DiT all-gathers full latents to every rank,
        # so rank 0 holds the full latent). For world_size>4 tiling is both
        # unsupported by _plan_centers and inefficient (HALO=13 vs latent W=80),
        # so we force "rank0".
        self.vae_mode = vae_mode
        if self.world_size > 4 and self.vae_mode == "tiled":
            self.vae_mode = "rank0"

        # Seed BEFORE building the pipeline so per-rank state that is generated
        # (not broadcast) — notably the sphere points used by memory-frame
        # selection — is identical on every rank. Latents are broadcast from
        # rank0 inside prepare_latents, but points_local is not.
        set_global_seed(self.cfg.seed)

        self.pipe = WorldPlayPipeline(cfg=self.cfg, device=self.device)
        if self.vae_mode == "rank0":
            # Disable the decoder's world-tiling collective (rank 0 decodes the
            # full latent alone) BEFORE start_instance, whose VAE warmup would
            # otherwise tile world_size-way — unsupported / inefficient for >4.
            self.pipe.vae_runner.vae.decoder.world_size = 1
            self.pipe.vae_runner.vae.decoder.rank = 0
        self.pipe.start_instance()
        warmup_sequence_parallel_communication(self.device)

        # camera accumulators (mirror the reference worker's running pose)
        self.T = torch.eye(4, dtype=torch.float32)
        self.C_inv = torch.zeros((4, 4), dtype=torch.float32)
        self.num_executed_actions = 0
        self.session_started = False

        if self.rank == 0:
            self._init_ipc()

        self.warmup()
        if self.rank == 0:
            logger.info("serving: sequence-parallel worker world_size=%d sp=%d stream_vae=%s",
                        self.world_size, get_sp_world_size(), self.stream_vae)
            logger.info("WorldPlay backend READY")

    # ------------------------------------------------------------------
    def _init_ipc(self):
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        self.action_buffer = SharedTensorBuffer(
            self.cfg.action_buffer_name, frame_shape=(1,), dtype=np.int64,
            max_len=int(self.cfg.max_num_actions), create=True,
        )

    # ------------------------------------------------------------------
    # session lifecycle (all ranks act in lockstep; rank0 acks)
    # ------------------------------------------------------------------
    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None,
                               image_path=self.cfg.image_path)
        for _ in range(int(self.cfg.max_num_actions) // int(self.cfg.chunk_size)):
            dummy = np.zeros((int(self.cfg.chunk_size),), dtype=np.int64)
            self._predict(dummy)
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()

    def _do_start(self):
        set_global_seed(self.cfg.seed)
        self.num_executed_actions = 0
        self.session_started = True
        custom = os.path.join("/tmp", f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png")
        image_path = custom if os.path.exists(custom) else self.cfg.image_path
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=image_path)
        if self.rank == 0:
            self.ctrl_buffer.commit()

    def _do_reset(self):
        self.session_started = False
        self.num_executed_actions = 0
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()
        if self.rank == 0:
            self.action_buffer.clear()
            self.ctrl_buffer.commit()

    def _do_terminate(self):
        self.session_started = False
        self.pipe.terminate_instance()
        if self.rank == 0:
            self.ctrl_buffer.unlink()
            self.action_buffer.unlink()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # one chunk (all ranks; rank0 publishes video)
    # ------------------------------------------------------------------
    def _predict(self, codes: np.ndarray):
        pipe, cfg = self.pipe, self.cfg
        tcodes, rcodes = wllm.kernels_t.camera_action.decode_combined_actions(codes)
        vm, ks, act = wllm.kernels_t.camera_action.motions_to_matrix_with_rotation(
            tcodes, rcodes, self.T, self.C_inv,
            first_chunk=(pipe._session_ctx["latent_chunk_idx"] == 0),
        )
        act = wllm.kernels_t.camera_action.compute_worldplay_combined_label(act, rcodes)
        self._step(vm.unsqueeze(0), ks.unsqueeze(0), act.unsqueeze(0))

    @torch.inference_mode()
    def _step(self, viewmats, Ks, action):
        pipe, cfg = self.pipe, self.cfg
        chunk_i = pipe._session_ctx["latent_chunk_idx"]
        first_image_condition = pipe._session_ctx["first_image_condition"]
        start_idx = chunk_i * cfg.chunk_size
        end_idx = start_idx + cfg.chunk_size

        pipe._viewmats[:, start_idx:end_idx, ...] = viewmats
        pipe._Ks[:, start_idx:end_idx, ...] = Ks
        pipe._action[:, start_idx:end_idx, ...] = action

        selected_frame_indices = None
        if chunk_i == 0:
            already_generate_num = cfg.first_chunk_size
            generate_latent_num = cfg.first_chunk_size
            pipe._latents[:, :, :1] = first_image_condition
            latents_curr = pipe._latents[:, :, :already_generate_num].to(self.device, pipe.dtype)
        else:
            already_generate_num = chunk_i * cfg.chunk_size + cfg.first_chunk_size
            latents_curr = pipe._latents[:, :, :already_generate_num].to(self.device, pipe.dtype)
            generate_latent_num = cfg.chunk_size
            cur = chunk_i * cfg.chunk_size
            if cfg.context_window_size <= cur < cfg.max_num_actions:
                selected_frame_indices = select_mem_frames_wan(
                    pipe._viewmats[0], cur, memory_frames=cfg.context_window_size,
                    temporal_context_size=(cfg.context_window_size - cfg.chunk_size),
                    pred_latent_size=cfg.chunk_size, points_local=pipe.points_local, device=self.device,
                )
            else:
                selected_frame_indices = list(range(0, cur))

        for i, t in enumerate(pipe._timesteps):
            if chunk_i > 0:
                t_now = torch.full((1, cfg.chunk_size), t, device=self.device, dtype=pipe._timesteps.dtype)
                t_ctx = torch.full((1, cfg.first_chunk_size + (chunk_i - 1) * cfg.chunk_size),
                                   cfg.stabilization_level - 1, device=self.device, dtype=pipe._timesteps.dtype)
                timestep = torch.cat([t_ctx, t_now], dim=1)
            else:
                t_now = torch.full((1, cfg.chunk_size - 1), t, device=self.device, dtype=pipe._timesteps.dtype)
                t_ctx = torch.full((1, 1), cfg.stabilization_level - 1, device=self.device, dtype=pipe._timesteps.dtype)
                timestep = torch.cat([t_ctx, t_now], dim=1)

            all_window_latent_num = latents_curr.shape[2]
            if chunk_i > 0 and i == 0:
                latents_cache = latents_curr[:, :, selected_frame_indices].clone()
                timestep_cache = timestep[:, selected_frame_indices].flatten()
                kv_end_rope = len(selected_frame_indices) * cfg.kv_spatial
                pipe.dit_runner.run(
                    latents=latents_cache, timestep=timestep_cache, is_cache=True,
                    cache_start=0, cache_end=kv_end_rope, rope_start=0, rope_end=kv_end_rope,
                    viewmats=pipe._viewmats[:, selected_frame_indices],
                    Ks=pipe._Ks[:, selected_frame_indices],
                    action=pipe._action[:, selected_frame_indices], i2v_condition=None,
                )
            now_window = (len(selected_frame_indices) + cfg.chunk_size) if selected_frame_indices is not None else cfg.chunk_size
            latent_model_input = latents_curr[:, :, -generate_latent_num:].clone()
            timestep_slice = timestep[:, -generate_latent_num:].flatten()
            gen_s = all_window_latent_num - generate_latent_num
            gen_e = all_window_latent_num
            gen_rope_s = (now_window - generate_latent_num) * cfg.kv_spatial
            gen_rope_e = now_window * cfg.kv_spatial
            noise_pred = pipe.dit_runner.run(
                latents=latent_model_input, timestep=timestep_slice, is_cache=False,
                cache_start=gen_rope_s, cache_end=gen_rope_e, rope_start=gen_rope_s, rope_end=gen_rope_e,
                viewmats=pipe._viewmats[:, gen_s:gen_e], Ks=pipe._Ks[:, gen_s:gen_e],
                action=pipe._action[:, gen_s:gen_e], i2v_condition=None,
            )
            sigma = pipe._sigmas[i]; sigma_next = pipe._sigmas[i + 1]; dt = sigma_next - sigma
            if chunk_i == 0:
                prev_sample = latent_model_input + dt * noise_pred
                latents_curr[:, :, -cfg.first_chunk_size + 1:] = prev_sample[:, :, 1:]
            else:
                noise_pred_chunk = noise_pred[:, :, -cfg.chunk_size:]
                latents_curr[:, :, -cfg.chunk_size:] = latent_model_input[:, :, -cfg.chunk_size:] + dt * noise_pred_chunk

        pipe._latents[:, :, :already_generate_num, :, :] = latents_curr

        # VAE decode. In "tiled" mode every rank calls run() per latent (the
        # width-tiled decode is a collective across the world group); in "rank0"
        # mode only rank 0 decodes (no collective). Only rank 0 publishes frames;
        # with stream_vae it writes each latent's burst as decoded.
        vae_participates = (self.vae_mode == "tiled" and self.world_size > 1) or self.rank == 0
        if vae_participates:
            if self.vae_batch:
                # decode all of the chunk's latents in one call (the causal cache
                # chains them internally, identical to per-latent calls). is_first
                # is True only for the session's very first latent (chunk 0).
                chunk_lat = pipe._latents[:, :, start_idx:end_idx, :, :].clone()
                video = pipe.vae_runner.run(chunk_lat, (start_idx == 0))
                if self.rank == 0:
                    pipe._video_buffer.write(video[0].cpu().numpy())
            else:
                chunk_video: List[np.ndarray] = []
                for l_i in range(start_idx, end_idx):
                    latent_i = pipe._latents[:, :, l_i:l_i + 1, :, :].clone()
                    video_i = pipe.vae_runner.run(latent_i, (l_i == 0))
                    if self.rank == 0:
                        frames = video_i[0].cpu().numpy()
                        if self.stream_vae:
                            pipe._video_buffer.write(frames)
                        else:
                            chunk_video.append(frames)
                if self.rank == 0 and not self.stream_vae and chunk_video:
                    pipe._video_buffer.write(np.concatenate(chunk_video, axis=0))

        pipe._session_ctx["latent_chunk_idx"] = chunk_i + 1

    # ------------------------------------------------------------------
    # rank0 reactive action read (mirror reference worker.get_actions)
    # ------------------------------------------------------------------
    def _get_actions(self) -> Optional[np.ndarray]:
        actions: list = []
        for _ in range(int(self.cfg.chunk_size)):
            self.num_executed_actions, new_action = self.action_buffer.read(self.num_executed_actions, 1)
            if new_action is None:
                break
            actions.append(new_action.ravel())
        if len(actions) == 0:
            return None
        actions = np.concatenate(actions).ravel()
        base, rem = divmod(int(self.cfg.chunk_size), len(actions))
        repeats = np.full(len(actions), base, dtype=np.int64)
        repeats[:rem] += 1
        actions = np.repeat(actions, repeats).flatten()
        if self.num_executed_actions > int(self.cfg.max_num_actions):
            return None
        return actions

    def _poll(self):
        """rank0: block until there's a control opcode or an action chunk."""
        while True:
            v = int(self.ctrl_buffer.recv())
            if v == 2 and self.session_started:
                return (_TERMINATE, None)
            if v == 1 and not self.session_started:
                return (_START, None)
            if v == 3 and self.session_started:
                return (_RESET, None)
            if self.session_started:
                codes = self._get_actions()
                if codes is not None:
                    return (_STEP, codes.tolist())
            time.sleep(0.002)

    # ------------------------------------------------------------------
    def loop(self):
        wg = get_world_group()
        while True:
            msg = self._poll() if self.rank == 0 else None
            msg = wg.broadcast_object(msg, src=0)
            op, payload = msg
            if op == _TERMINATE:
                self._do_terminate()
                break
            elif op == _START:
                self._do_start()
            elif op == _RESET:
                self._do_reset()
            elif op == _STEP:
                self._predict(np.asarray(payload, dtype=np.int64))
        destroy_model_parallel()
        destroy_distributed_environment()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stream-vae", action="store_true")
    ap.add_argument("--vae-mode", default="tiled", choices=["tiled", "rank0"])
    ap.add_argument("--vae-batch", action="store_true")
    args = ap.parse_args()
    patch_pg_timeout()  # widen PG timeout so cold-cache rank desync isn't watchdog-killed
    SPWorldPlayWorker(args.cfg, stream_vae=args.stream_vae, vae_mode=args.vae_mode,
                      vae_batch=args.vae_batch).loop()


if __name__ == "__main__":
    main()
