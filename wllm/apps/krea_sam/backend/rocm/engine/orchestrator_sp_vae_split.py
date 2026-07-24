"""Three-way disaggregation: SAM || DiT(sequence-parallel) || VAE.

The 4-GPU `stream_sp3` runs DiT-SP and the VAE decode on the SAME sp ranks, so
the decode serializes behind the SP group. The 3-GPU `krea_pipeline` splits
DiT|VAE onto separate ranks but is SP=1. This composes both: an SP group denoises
while a dedicated rank decodes chunk N-1 in parallel, with SAM decoupled as usual.

    ranks 0..sp-1  DiT sequence-parallel group (encode + denoise)
      rank 0       also owns input/ctrl/SAM-link (write) and drives the session
    rank sp        VAE decode + SAM masks + composite + output buffer
    SAM            separate process on its own GPU (as in every variant)

  => sp_size + 1 Krea GPUs + 1 SAM GPU  (sp=3 -> 5 GPUs)

TOPOLOGY NOTE (why the bootstrap below exists): the stock
`initialize_model_parallel(sp_size=k)` slices the world into consecutive k-rank
SP groups, so a world of k+1 leaves the VAE rank in no SP group at all and
`GroupCoordinator` raises ("rank N group not found"). We therefore build the
groups directly from the shared primitive with a non-uniform layout --
`[[0..k-1], [k]]` -- so the DiT ranks see sp_world_size==k while the VAE rank
gets a singleton group (sp_world_size==1, no collectives it must join). Both are
registered into the shared `parallel_state` globals, so every shared consumer
(the DiT's SP sharding, the world group used for P2P) sees a normal world.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.distributed import parallel_state as ps
from wllm.serving.distributed.parallel_state import get_sp_group, get_world_group
from wllm.serving.logger import init_logger
from wllm.serving.platforms import current_platform
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.reference.pipeline import KreaSAMPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator import READY_MARKER
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_stream import denoise_chunk
from wllm.apps.krea_sam.backend.rocm.engine.sam_link import SamLink

logger = init_logger("orchestrator_sp_vae_split")
set_torch_options()

IDLE, START, TERMINATE, RESET, CHUNK = 0, 1, 2, 3, 4


def init_sp_vae_split_parallel_state(sp_size: int) -> None:
    """Bootstrap `parallel_state` with a DiT SP group + a standalone VAE rank.

    The `maybe_init_distributed_environment_and_model_parallel` counterpart for
    this topology: same env contract (RANK/WORLD_SIZE/LOCAL_RANK), composing the
    same `init_distributed_environment` / `init_model_parallel_group` primitives,
    but with the non-uniform SP layout this variant needs. The result lands in
    the same globals, so every shared consumer sees an ordinary world.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    assert world_size == sp_size + 1, (
        f"sp_vae_split needs world == sp_size + 1 (sp ranks + 1 VAE rank); "
        f"got world={world_size}, sp_size={sp_size}")
    device = ps.get_local_torch_device()
    logger.info("sp_vae_split: world=%d rank=%d sp=%d vae_rank=%d device=%s",
                world_size, rank, sp_size, sp_size, device)
    ps.init_distributed_environment(world_size=world_size, rank=rank,
                                    local_rank=local_rank,
                                    distributed_init_method="env://",
                                    device_id=device)
    wl = ps.get_world_group().local_rank
    backend = torch.distributed.get_backend(ps.get_world_group().device_group)
    singletons = [[i] for i in range(world_size)]
    # tp=1, dp=1: every rank alone in its group (neither is used by this app).
    ps._TP = ps.init_model_parallel_group(singletons, wl, backend,
                                          use_message_queue_broadcaster=True,
                                          group_name="tp")
    # DiT ranks share one SP group; the VAE rank gets a singleton so it belongs
    # to a group (GroupCoordinator requires every rank to) yet sees
    # sp_world_size==1 and never has to join an SP collective.
    ps._SP = ps.init_model_parallel_group([list(range(sp_size)), [sp_size]],
                                          wl, backend, group_name="sp")
    ps._DP = ps.init_model_parallel_group(singletons, wl, backend, group_name="dp")
    current_platform.get_torch_device().set_device(
        torch.device(f"{current_platform.device_type}:{local_rank}"))


class KreaSPVaeSplit:
    def __init__(self, cfg_path: str, sam_link_name: str, device, rank: int,
                 sp_size: int):
        self.cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True).to_runtime_config()
        self.device = device
        self.rank = rank
        self.sp_size = sp_size
        self.vae_rank = sp_size
        self.is_lead = (rank == 0)
        self.is_vae = (rank == self.vae_rank)
        self.is_dit = (rank < sp_size)
        torch.cuda.set_device(device)
        H, W = int(self.cfg.height), int(self.cfg.width)

        # shm ownership: lead writes input/ctrl + creates the SAM link; the VAE
        # rank owns the output buffer and reads masks; SP followers own nothing.
        if self.is_lead:
            self.video_input_buffer = SharedTensorBuffer(
                name=self.cfg.video_input_buffer_name, frame_shape=(H, W, 3),
                max_len=int(self.cfg.video_input_max_frames), dtype=np.uint8, create=True)
            self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
            self.sam_link = SamLink(sam_link_name, H, W,
                                    int(self.cfg.video_input_max_frames), create=True)
        elif self.is_vae:
            self.video_buffer = SharedTensorBuffer(
                name=self.cfg.video_buffer_name, frame_shape=(H, W, 3),
                max_len=int(self.cfg.max_num_frames), dtype=np.uint8, create=True)
            # attach to the lead's link (SamLink waits for the creator)
            self.sam_link = SamLink(sam_link_name, H, W,
                                    int(self.cfg.video_input_max_frames), create=False)

        self.pipe = KreaSAMPipeline(cfg=self.cfg, device=device)
        self.pipe.start_instance()
        # The WAN VAE decoder auto-width-tiles when get_world_size()>1 (a WORLD
        # collective), but decode runs on the VAE rank alone -- disable tiling so
        # it never waits on ranks that are busy denoising.
        self.pipe.vae_runner.vae.decoder.world_size = 1

        self.session_started = False
        self.num_consumed_input_frames = 0
        self._sam_push_count = 0
        self._skip = 0
        self.warmup()
        if self.is_lead:
            t0 = time.time()
            while self.sam_link.sam_ready_epoch < 0:
                if time.time() - t0 > 1800:
                    raise TimeoutError("SAM worker never ready")
                time.sleep(0.05)
            logger.info("sp_vae_split up (sp=%d, vae_rank=%d)", self.sp_size, self.vae_rank)
            print(READY_MARKER, flush=True)

    # ---- warmup ----
    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt,
                               negative_prompt=self.cfg.negative_prompt or None)
        if self.is_dit:
            T = int(self.pipe.input_frames_for_next_step())
            dummy = torch.zeros((T, 3, self.cfg.height, self.cfg.width),
                                device=self.device, dtype=self.pipe.dtype)
            for _ in range(8):
                if denoise_chunk(self.pipe, dummy)[0] is not None:
                    break
        else:
            lat = torch.zeros(1, self.cfg.vae_config.z_dim, 1, self.cfg.latent_height,
                              self.cfg.latent_width, device=self.device, dtype=self.pipe.dtype)
            self.pipe.vae_runner.run(lat, True)
        torch.cuda.synchronize(self.device)
        self.pipe.reset()

    # ---- transports: SP-group broadcast (DiT ranks) + world P2P (lead -> VAE) ----
    def _bcast_cmd(self, action: int, T: int):
        t = torch.tensor([action, T], device=self.device, dtype=torch.int64)
        get_sp_group().broadcast(t, src=0)      # src is the group-local rank
        return int(t[0].item()), int(t[1].item())

    def _bcast_input(self, T: int, krea_input):
        if krea_input is None:
            krea_input = torch.empty((T, 3, self.cfg.height, self.cfg.width),
                                     device=self.device, dtype=self.pipe.dtype)
        get_sp_group().broadcast(krea_input, src=0)
        return krea_input

    def _send_hdr(self, action, T, mask_start, block_idx):
        t = torch.tensor([action, T, mask_start, block_idx], device=self.device,
                         dtype=torch.int64)
        get_world_group().send(t, dst=self.vae_rank)

    def _recv_hdr(self):
        t = get_world_group().recv(torch.Size([4]), torch.int64, src=0)
        return [int(x) for x in t.tolist()]

    # ---- session lifecycle ----
    def _start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.num_consumed_input_frames = 0
        self._sam_push_count = 0
        self._skip = max(0, int(self.cfg.vae_config.scale_factor_temporal) - 1)
        self.pipe.init_session(prompt=self.cfg.prompt,
                               negative_prompt=self.cfg.negative_prompt or None)
        if self.is_lead:
            self.video_input_buffer.clear()
            self.sam_link.new_session()
            self.ctrl_buffer.commit()
        elif self.is_vae:
            self.video_buffer.clear()

    def _reset(self):
        self.session_started = False
        self.num_consumed_input_frames = 0
        self._sam_push_count = 0
        self.pipe.reset()
        if self.is_lead:
            self.video_input_buffer.clear()
            self.sam_link.new_session()
            self.ctrl_buffer.commit()
        elif self.is_vae:
            self.video_buffer.clear()

    def _terminate(self):
        self.session_started = False
        self.pipe.terminate_instance()
        if self.is_lead:
            self.sam_link.signal_terminate()
            self.ctrl_buffer.unlink()
            self.video_input_buffer.unlink()
            time.sleep(0.3)
            self.sam_link.unlink()
        elif self.is_vae:
            self.video_buffer.unlink()
        torch.cuda.empty_cache()

    def _poll_input(self):
        target = int(self.pipe.input_frames_for_next_step())
        available = max(0, int(self.video_input_buffer.num) - int(self.num_consumed_input_frames))
        if available < target:
            return None
        self.num_consumed_input_frames, frames = self.video_input_buffer.read(
            self.num_consumed_input_frames, available)
        if frames is None:
            return None
        if len(frames) != target:
            idx = np.round(np.linspace(0, len(frames) - 1, target)).astype(np.int64)
            frames = frames[idx]
        raw = np.ascontiguousarray(frames)
        ft = torch.from_numpy(raw).to(device=self.device, dtype=torch.uint8)
        krea_input = (ft.permute(0, 3, 1, 2).to(self.pipe.dtype)
                      .div_(127.5).sub_(1.0).contiguous())
        return krea_input, raw

    # ---- loops ----
    def loop(self):
        if self.is_lead:
            self._loop_lead()
        elif self.is_vae:
            self._loop_vae()
        else:
            self._loop_sp_follower()

    def _loop_lead(self):
        while True:
            op = int(self.ctrl_buffer.recv())
            if op == 2:
                self._bcast_cmd(TERMINATE, 0)
                self._send_hdr(TERMINATE, 0, 0, 0)
                self._terminate()
                break
            if op == 1 and not self.session_started:
                self._bcast_cmd(START, 0)
                self._send_hdr(START, 0, 0, 0)
                self._start()
                continue
            if op == 3 and self.session_started:
                self._bcast_cmd(RESET, 0)
                self._send_hdr(RESET, 0, 0, 0)
                self._reset()
                continue
            polled = self._poll_input() if self.session_started else None
            if polled is None:
                self._bcast_cmd(IDLE, 0)
                self._send_hdr(IDLE, 0, 0, 0)
                time.sleep(0.003)
                continue
            krea_input, raw = polled
            T = int(raw.shape[0])
            self.sam_link.push_frames(raw)          # SAM starts immediately
            mask_start = self._sam_push_count
            self._sam_push_count += T
            block_idx = self.pipe._block_idx
            self._bcast_cmd(CHUNK, T)
            krea_input = self._bcast_input(T, krea_input)
            denoised, _ = denoise_chunk(self.pipe, krea_input)   # SP over 0..sp-1
            if denoised is None:                    # not enough latents yet
                self._send_hdr(IDLE, 0, 0, 0)
                continue
            self._send_hdr(CHUNK, T, mask_start, block_idx)
            get_world_group().send(torch.from_numpy(raw).to(self.device), dst=self.vae_rank)
            get_world_group().send(denoised.contiguous(), dst=self.vae_rank)

    def _loop_sp_follower(self):
        while True:
            action, T = self._bcast_cmd(IDLE, 0)    # value comes from the lead
            if action == TERMINATE:
                self._terminate()
                break
            if action == START:
                self._start()
                continue
            if action == RESET:
                self._reset()
                continue
            if action != CHUNK:
                continue
            krea_input = self._bcast_input(T, None)
            denoise_chunk(self.pipe, krea_input)   # join the SP collective

    def _loop_vae(self):
        H, W = int(self.cfg.height), int(self.cfg.width)
        latent_shape = torch.Size([1, self.cfg.dit_config.out_channels,
                                   self.cfg.chunk_size, self.cfg.latent_height,
                                   self.cfg.latent_width])
        while True:
            action, T, mask_start, block_idx = self._recv_hdr()
            if action == TERMINATE:
                self._terminate()
                break
            if action == START:
                self._start()
                continue
            if action == RESET:
                self._reset()
                continue
            if action != CHUNK:
                continue
            raw = get_world_group().recv(torch.Size([T, H, W, 3]), torch.uint8,
                                         src=0).cpu().numpy()
            denoised = get_world_group().recv(latent_shape, self.pipe.dtype, src=0)
            self._decode_and_emit(denoised, raw, mask_start, block_idx)

    def _decode_and_emit(self, denoised, raw, mask_start, block_idx):
        p = 0
        for frame_i in range(int(denoised.shape[2])):
            latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
            is_first = (block_idx == 0 and frame_i == 0)
            pix = np.asarray(self.pipe.vae_runner.run(latent_i, is_first)[0].cpu().numpy())
            for j in range(int(pix.shape[0])):
                if p < self._skip:
                    p += 1
                    continue
                mask = self.sam_link.read_masks(mask_start + p, 1)[0]
                m = (mask > 0).astype(np.uint8)[:, :, None]
                self.video_buffer.write(raw[p] * m + pix[j] * (1 - m))
                p += 1
        self._skip = 0
