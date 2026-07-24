"""Stage-split WorldPlay worker: DiT stage ∥ VAE stage, pipelined across chunks.

IR basis (worldplay_chunk analysis): find_pipeline_stages separates the DiT
(stage 0: ingest/kv_fill/denoise/writeback, state cam/latents/kv) from the VAE
(stage 1: vae_decode, state vae); analyze_cross_chunk_dependencies reports every
(denoise_step_*, vae_decode_*) pair as cross-chunk independent. So VAE(chunk N)
may run on a different device, concurrently with DiT(chunk N+1). This worker
realises that: the first ``dit_ranks`` ranks run the DiT (sequence-parallel over
themselves) and the last ``vae_ranks`` ranks run the VAE (width-tiled over
themselves). The DiT leader streams each chunk's latents to the VAE leader over a
point-to-point channel, so the VAE group decodes chunk N while the DiT group
already computes chunk N+1 — no collective forces the two stages into lockstep.

Distributed setup uses the shared parallel_state primitives: the world group for
the cross-stage p2p, and two custom sequence-parallel groups (DiT ranks /
VAE ranks) registered into the shared _SP/_TP/_DP globals via
init_model_parallel_group so the DiT model's Ulysses code and the VAE tiling both
read a consistent world. (initialize_model_parallel only builds uniform SP groups
and cannot express the asymmetric DiT/VAE split, so we compose the lower-level
shared helper instead, per the AGENTS.md multi-GPU rule; globals audited below:
the DiT model reads get_sp_group/get_sp_world_size/get_sp_parallel_rank +
model_parallel_is_initialized — all populated; the VAE decoder reads get_world_size
which we override per-process to the VAE sub-group size.)
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
import wllm.serving.distributed.parallel_state as ps
from wllm.serving.distributed.parallel_state import (
    init_distributed_environment, init_model_parallel_group, get_world_group,
    get_sp_group, get_sp_world_size, get_sp_parallel_rank, get_world_rank,
    destroy_model_parallel, destroy_distributed_environment,
)
from wllm.serving.distributed.communication_op import warmup_sequence_parallel_communication
from wllm.apps.worldplay.backend.cuda.runtime.dist_timeout import patch_pg_timeout
import wllm.serving.models.vae.wan_vae as wan_vae_mod
from wllm.apps.worldplay.backend.cuda.runtime import vendored_vae_plan as vplan

logger = init_logger(__name__)
set_torch_options()

_START, _TERMINATE, _RESET, _STEP, _IDLE = 1, 2, 3, 4, 5


def _setup_groups(dit_ranks: int, vae_ranks: int):
    """World group (all ranks) + custom SP groups: {0..dit-1} for the DiT,
    {dit..N-1} for the VAE. Registered into the shared parallel_state globals."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
    lr = get_world_group().local_rank
    backend = "nccl"
    dit_g = list(range(0, dit_ranks))
    vae_g = list(range(dit_ranks, dit_ranks + vae_ranks))
    # _SP: DiT ranks share one group, VAE ranks share another.
    ps._SP = init_model_parallel_group([dit_g, vae_g], lr, backend, group_name="sp")
    # trivial _TP (size 1 each) and _DP (world) so model_parallel_is_initialized() holds.
    ps._TP = init_model_parallel_group([[r] for r in range(world_size)], lr, backend, group_name="tp")
    ps._DP = init_model_parallel_group([list(range(world_size))], lr, backend, group_name="dp")
    return rank, world_size, local_rank


class StageSplitWorker:
    def __init__(self, cfg_path: str, dit_ranks: int, vae_ranks: int, stream_vae: bool = False):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)
        self.dit_ranks = dit_ranks
        self.vae_ranks = vae_ranks
        self.stream_vae = bool(stream_vae)

        self.rank, self.world_size, local_rank = _setup_groups(dit_ranks, vae_ranks)
        assert self.world_size == dit_ranks + vae_ranks
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        self.is_dit = self.rank < dit_ranks
        self.dit_leader = 0
        self.vae_leader = dit_ranks

        set_global_seed(self.cfg.seed)
        self.pipe = WorldPlayPipeline(cfg=self.cfg, device=self.device)

        # VAE decoder tiling. DiT ranks never produce output frames, so they must
        # NOT tile (otherwise their start_instance VAE warmup would issue an
        # N-way world-group collective that the VAE ranks — tiling over only the
        # VAE sub-group — never join, deadlocking init). VAE ranks point the
        # decoder's tiling at the VAE sub-group via the group-aware vendored
        # split_tile/gather_tile (monkeypatched onto the module-level names that
        # WanDecoder3d.forward calls).
        dec = self.pipe.vae_runner.vae.decoder
        if self.is_dit:
            dec.world_size = 1
            dec.rank = 0
        else:
            vae_world = self.vae_ranks
            vae_rank = get_sp_parallel_rank()  # rank within the VAE SP group
            dec.world_size = vae_world
            dec.rank = vae_rank
            if vae_world > 1:
                vae_group = get_sp_group().device_group
                wan_vae_mod.split_tile = lambda x, _r=vae_rank, _w=vae_world: vplan.split_tile(x, _r, _w)
                wan_vae_mod.gather_tile = lambda y, m, _r=vae_rank, _w=vae_world, _g=vae_group: vplan.gather_tile(y, m, _r, _w, _g)

        self.pipe.start_instance()  # rank 0 creates the video buffer
        warmup_sequence_parallel_communication(self.device)

        # VAE leader attaches to the video buffer rank 0 created (it does the writes).
        self.video_out = None
        if self.rank == self.vae_leader:
            self.video_out = SharedTensorBuffer(
                name=self.cfg.video_buffer_name, frame_shape=(self.cfg.height, self.cfg.width, 3),
                max_len=self.cfg.max_num_frames, dtype=np.uint8, create=False,
            )

        # camera accumulators (DiT side only needs them)
        self.T = torch.eye(4, dtype=torch.float32)
        self.C_inv = torch.zeros((4, 4), dtype=torch.float32)
        self.num_executed_actions = 0
        self.session_started = False

        if self.rank == self.dit_leader:
            self._init_ipc()

        # preallocated p2p buffers (fixed chunk latent shape [1, C, chunk, h, w])
        cs = self.cfg.chunk_size
        self._lat_shape = (1, self.cfg.dit_config.out_channels, cs,
                           self.cfg.latent_height, self.cfg.latent_width)
        self.warming = True   # suppress video-buffer writes during warmup
        self.warmup()
        self.warming = False
        if self.rank == self.dit_leader:
            logger.info("serving: stage-split worker dit=%d vae=%d stream_vae=%s",
                        dit_ranks, vae_ranks, self.stream_vae)
            logger.info("WorldPlay backend READY")

    # ------------------------------------------------------------------
    def _init_ipc(self):
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        self.action_buffer = SharedTensorBuffer(self.cfg.action_buffer_name, frame_shape=(1,),
                                                dtype=np.int64, max_len=int(self.cfg.max_num_actions), create=True)

    # ------------------------------------------------------------------
    # cross-stage p2p (DiT leader <-> VAE leader) over the world group
    # ------------------------------------------------------------------
    def _send_header(self, opcode, chunk_idx):
        h = torch.tensor([opcode, chunk_idx, 0], dtype=torch.int64, device=self.device)
        get_world_group().send(h, dst=self.vae_leader)

    def _recv_header(self):
        h = get_world_group().recv(torch.Size([3]), torch.int64, src=self.dit_leader)
        return int(h[0].item()), int(h[1].item())

    def _send_latents(self, latents):
        get_world_group().send(latents.contiguous(), dst=self.vae_leader)

    def _recv_latents(self):
        return get_world_group().recv(torch.Size(self._lat_shape), self.pipe.dtype, src=self.dit_leader)

    # broadcast within a stage's SP group (used when a stage has >1 rank)
    def _dit_bcast_obj(self, obj):
        if self.dit_ranks > 1:
            return get_sp_group().broadcast_object(obj, src=0)
        return obj

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------
    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=self.cfg.image_path)
        for ci in range(int(self.cfg.max_num_actions) // int(self.cfg.chunk_size)):
            dummy = np.zeros((int(self.cfg.chunk_size),), dtype=np.int64)
            self._run_chunk_dummy(dummy, ci)
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()

    def _run_chunk_dummy(self, codes, chunk_idx):
        # warm both stages without IPC/p2p: DiT ranks denoise, VAE ranks decode a
        # zero latent so all kernels (incl tiling collectives) get compiled.
        if self.is_dit:
            lat = self._dit_denoise(codes)
        else:
            lat = torch.zeros(self._lat_shape, device=self.device, dtype=self.pipe.dtype)
            self._vae_decode(lat, chunk_idx)

    def _do_start(self):
        set_global_seed(self.cfg.seed)
        self.num_executed_actions = 0
        self.session_started = True
        custom = os.path.join("/tmp", f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png")
        image_path = custom if os.path.exists(custom) else self.cfg.image_path
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=image_path)
        if self.rank == self.dit_leader:
            self.ctrl_buffer.commit()

    def _do_reset(self):
        self.session_started = False
        self.num_executed_actions = 0
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()
        if self.rank == self.dit_leader:
            self.action_buffer.clear()
            self.ctrl_buffer.commit()

    def _do_terminate(self):
        self.session_started = False
        self.pipe.terminate_instance()
        if self.rank == self.dit_leader:
            self.ctrl_buffer.unlink(); self.action_buffer.unlink()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # DiT denoise (runs on DiT ranks; returns the chunk's latents on the leader)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _dit_denoise(self, codes: np.ndarray):
        pipe, cfg = self.pipe, self.cfg
        tcodes, rcodes = wllm.kernels_t.camera_action.decode_combined_actions(codes)
        vm, ks, act = wllm.kernels_t.camera_action.motions_to_matrix_with_rotation(
            tcodes, rcodes, self.T, self.C_inv, first_chunk=(pipe._session_ctx["latent_chunk_idx"] == 0))
        act = wllm.kernels_t.camera_action.compute_worldplay_combined_label(act, rcodes)
        viewmats, Ks, action = vm.unsqueeze(0), ks.unsqueeze(0), act.unsqueeze(0)

        chunk_i = pipe._session_ctx["latent_chunk_idx"]
        first_image_condition = pipe._session_ctx["first_image_condition"]
        start_idx = chunk_i * cfg.chunk_size
        end_idx = start_idx + cfg.chunk_size
        pipe._viewmats[:, start_idx:end_idx] = viewmats
        pipe._Ks[:, start_idx:end_idx] = Ks
        pipe._action[:, start_idx:end_idx] = action

        sel = None
        if chunk_i == 0:
            already = cfg.first_chunk_size; gen = cfg.first_chunk_size
            pipe._latents[:, :, :1] = first_image_condition
            latents_curr = pipe._latents[:, :, :already].to(self.device, pipe.dtype)
        else:
            already = chunk_i * cfg.chunk_size + cfg.first_chunk_size
            latents_curr = pipe._latents[:, :, :already].to(self.device, pipe.dtype)
            gen = cfg.chunk_size
            cur = chunk_i * cfg.chunk_size
            if cfg.context_window_size <= cur < cfg.max_num_actions:
                sel = select_mem_frames_wan(pipe._viewmats[0], cur, memory_frames=cfg.context_window_size,
                                            temporal_context_size=(cfg.context_window_size - cfg.chunk_size),
                                            pred_latent_size=cfg.chunk_size, points_local=pipe.points_local, device=self.device)
            else:
                sel = list(range(0, cur))

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
            awn = latents_curr.shape[2]
            if chunk_i > 0 and i == 0:
                lc = latents_curr[:, :, sel].clone()
                kvend = len(sel) * cfg.kv_spatial
                pipe.dit_runner.run(latents=lc, timestep=timestep[:, sel].flatten(), is_cache=True,
                                    cache_start=0, cache_end=kvend, rope_start=0, rope_end=kvend,
                                    viewmats=pipe._viewmats[:, sel], Ks=pipe._Ks[:, sel],
                                    action=pipe._action[:, sel], i2v_condition=None)
            now_window = (len(sel) + cfg.chunk_size) if sel is not None else cfg.chunk_size
            lmi = latents_curr[:, :, -gen:].clone()
            gs = awn - gen; ge = awn
            rs = (now_window - gen) * cfg.kv_spatial; re = now_window * cfg.kv_spatial
            noise_pred = pipe.dit_runner.run(latents=lmi, timestep=timestep[:, -gen:].flatten(), is_cache=False,
                                             cache_start=rs, cache_end=re, rope_start=rs, rope_end=re,
                                             viewmats=pipe._viewmats[:, gs:ge], Ks=pipe._Ks[:, gs:ge],
                                             action=pipe._action[:, gs:ge], i2v_condition=None)
            dt = pipe._sigmas[i + 1] - pipe._sigmas[i]
            if chunk_i == 0:
                prev = lmi + dt * noise_pred
                latents_curr[:, :, -cfg.first_chunk_size + 1:] = prev[:, :, 1:]
            else:
                latents_curr[:, :, -cfg.chunk_size:] = lmi[:, :, -cfg.chunk_size:] + dt * noise_pred[:, :, -cfg.chunk_size:]
        pipe._latents[:, :, :already] = latents_curr
        pipe._session_ctx["latent_chunk_idx"] = chunk_i + 1
        # Hand the chunk's latents to the VAE stage in the worker dtype (bf16):
        # pipe._latents is float32, but the p2p recv buffer is pipe.dtype, so they
        # MUST match (a float32 send into a bf16 recv desyncs the NCCL p2p channel).
        # The VAE casts latents to bf16 internally anyway, so this is numerically the
        # reference path.
        return pipe._latents[:, :, start_idx:end_idx, :, :].to(self.pipe.dtype).contiguous()

    # ------------------------------------------------------------------
    # VAE decode (runs on VAE ranks; leader writes frames)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _vae_decode(self, chunk_latents, chunk_idx):
        pipe, cfg = self.pipe, self.cfg
        # VAE leader broadcasts the latents to the VAE group so all VAE ranks
        # can width-tile the decode.
        if self.vae_ranks > 1:
            chunk_latents = chunk_latents.contiguous()
            get_sp_group().broadcast(chunk_latents, src=0)
        start_idx = chunk_idx * cfg.chunk_size
        is_first = (start_idx == 0)
        publish = self.video_out is not None and not self.warming
        if self.stream_vae:
            for j in range(cfg.chunk_size):
                vi = pipe.vae_runner.run(chunk_latents[:, :, j:j + 1].contiguous(), (start_idx + j == 0))
                if publish:
                    self.video_out.write(vi[0].cpu().numpy())
        else:
            video = pipe.vae_runner.run(chunk_latents, is_first)
            if publish:
                self.video_out.write(video[0].cpu().numpy())

    # ------------------------------------------------------------------
    # rank0 reactive action read (mirror reference)
    # ------------------------------------------------------------------
    def _get_actions(self) -> Optional[np.ndarray]:
        actions: list = []
        for _ in range(int(self.cfg.chunk_size)):
            self.num_executed_actions, na = self.action_buffer.read(self.num_executed_actions, 1)
            if na is None:
                break
            actions.append(na.ravel())
        if not actions:
            return None
        actions = np.concatenate(actions).ravel()
        base, rem = divmod(int(self.cfg.chunk_size), len(actions))
        reps = np.full(len(actions), base, dtype=np.int64); reps[:rem] += 1
        actions = np.repeat(actions, reps).flatten()
        if self.num_executed_actions > int(self.cfg.max_num_actions):
            return None
        return actions

    def _poll(self):
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
        if self.rank == self.dit_leader:
            self._loop_dit_leader()
        elif self.is_dit:
            self._loop_dit_follower()
        else:
            self._loop_vae()
        destroy_model_parallel()
        destroy_distributed_environment()

    def _loop_dit_leader(self):
        chunk_idx = 0
        while True:
            op, payload = self._poll()
            self._dit_bcast_obj((op, payload))           # tell DiT followers
            if op == _STEP:
                # Denoise FIRST, then send the header+latents. Sending the header
                # before the denoise would block (the VAE leader only posts its
                # recv after finishing the previous decode), serialising the two
                # stages; sending after denoise lets the VAE decode chunk N while
                # we already compute chunk N+1.
                lat = self._dit_denoise(np.asarray(payload, dtype=np.int64))
                self._send_header(_STEP, chunk_idx)
                self._send_latents(lat)                  # hand chunk to VAE stage
                chunk_idx += 1
            elif op == _TERMINATE:
                self._send_header(_TERMINATE, chunk_idx)
                self._do_terminate(); break
            elif op == _START:
                self._send_header(_START, chunk_idx)
                self._do_start(); chunk_idx = 0
            elif op == _RESET:
                self._send_header(_RESET, chunk_idx)
                self._do_reset(); chunk_idx = 0

    def _loop_dit_follower(self):
        while True:
            op, payload = self._dit_bcast_obj(None)
            if op == _TERMINATE:
                self._do_terminate(); break
            elif op == _START:
                self._do_start()
            elif op == _RESET:
                self._do_reset()
            elif op == _STEP:
                self._dit_denoise(np.asarray(payload, dtype=np.int64))

    def _loop_vae(self):
        while True:
            if self.rank == self.vae_leader:
                op, chunk_idx = self._recv_header()
                meta = (op, chunk_idx)
            else:
                meta = None
            if self.vae_ranks > 1:
                meta = get_sp_group().broadcast_object(meta, src=0)
            op, chunk_idx = meta
            if op == _TERMINATE:
                self._do_terminate(); break
            elif op == _START:
                self._do_start()
            elif op == _RESET:
                self._do_reset()
            elif op == _STEP:
                if self.rank == self.vae_leader:
                    lat = self._recv_latents()
                else:
                    lat = torch.empty(self._lat_shape, device=self.device, dtype=self.pipe.dtype)
                self._vae_decode(lat, chunk_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--dit-ranks", type=int, required=True)
    ap.add_argument("--vae-ranks", type=int, required=True)
    ap.add_argument("--stream-vae", action="store_true")
    args = ap.parse_args()
    patch_pg_timeout()  # widen PG timeout so cold-cache rank desync isn't watchdog-killed
    StageSplitWorker(args.cfg, args.dit_ranks, args.vae_ranks, stream_vae=args.stream_vae).loop()


if __name__ == "__main__":
    main()
