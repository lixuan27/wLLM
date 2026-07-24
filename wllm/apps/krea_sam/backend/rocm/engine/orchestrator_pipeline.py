"""Disaggregated 2-stage Krea PIPELINE (the IR find_pipeline_stages lever).

The IR splits Krea into stages with disjoint caches (encode ⟂ decode). This
variant puts encode+denoise on rank 0 and decode+composite on rank 1, connected
by P2P, so decode(chunk N) overlaps denoise(chunk N+1) across chunks:

  rank0: read input -> push raw to SAM -> encode+denoise -> P2P send (denoised, raw)
  rank1: P2P recv -> decode -> read SAM masks -> composite -> write output

Rank 0 owns the DiT clean-context recurrence + input/ctrl buffers + SAM-link
write side; rank 1 owns the VAE-decoder recurrence + output buffer + SAM-link
read side. Uses the shared world GroupCoordinator for P2P (not a hand-rolled
transport). SAM is a third process on its own GPU.

NB: throughput is still gated by the SAM stage (IR bottleneck), so this mainly
tests the intra-Krea pipelining lever; latency gains the P2P transfer on the
serial encode→denoise→decode path.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.distributed.parallel_state import get_world_group
from wllm.serving.logger import init_logger
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.reference.pipeline import KreaSAMPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options
from wllm.apps.krea_sam.backend.rocm.engine.sam_link import SamLink
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator import READY_MARKER
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_stream import denoise_chunk

logger = init_logger("orchestrator_pipeline")
set_torch_options()

IDLE, START, TERMINATE, RESET, CHUNK = 0, 1, 2, 3, 4


class KreaPipeline:
    def __init__(self, cfg_path, sam_link_name, device, rank):
        self.cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True).to_runtime_config()
        self.device = device
        self.rank = rank
        self.is_lead = (rank == 0)
        torch.cuda.set_device(device)
        H, W = int(self.cfg.height), int(self.cfg.width)

        # shm: rank0 owns input+ctrl+sam_link (write), rank1 owns output+sam_link (read)
        if self.is_lead:
            self.video_input_buffer = SharedTensorBuffer(
                name=self.cfg.video_input_buffer_name, frame_shape=(H, W, 3),
                max_len=int(self.cfg.video_input_max_frames), dtype=np.uint8, create=True)
            self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
            self.sam_link = SamLink(sam_link_name, H, W, int(self.cfg.video_input_max_frames), create=True)
        else:
            self.video_buffer = SharedTensorBuffer(
                name=self.cfg.video_buffer_name, frame_shape=(H, W, 3),
                max_len=int(self.cfg.max_num_frames), dtype=np.uint8, create=True)
            self.sam_link = SamLink(sam_link_name, H, W, int(self.cfg.video_input_max_frames), create=False)

        self.pipe = KreaSAMPipeline(cfg=self.cfg, device=device)
        self.pipe.start_instance()
        # world=2 is only for P2P here (SP=1). The WAN VAE decoder auto-width-tiles
        # when get_world_size()>1 (a world collective); but decode runs on rank 1
        # alone, so disable tiling to avoid a collective the other rank never joins.
        try:
            self.pipe.vae_runner.vae.decoder.world_size = 1
        except Exception:
            pass

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
            print(READY_MARKER, flush=True)

    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=self.cfg.negative_prompt or None)
        win = torch.zeros((int(self.pipe.input_frames_for_next_step()), 3, self.cfg.height, self.cfg.width),
                          device=self.device, dtype=self.pipe.dtype)
        for _ in range(8):
            if self.is_lead:
                if denoise_chunk(self.pipe, win)[0] is not None:
                    break
            else:
                # rank1 warms the decoder on a dummy latent
                lat = torch.zeros(1, self.cfg.vae_config.z_dim, 1, self.cfg.latent_height,
                                  self.cfg.latent_width, device=self.device, dtype=self.pipe.dtype)
                self.pipe.vae_runner.run(lat, True)
                break
        torch.cuda.synchronize(self.device)
        self.pipe.reset()

    # ---- P2P header helpers (world group; rank0<->rank1) ----
    def _send_hdr(self, action, T, mask_start, block_idx):
        t = torch.tensor([action, T, mask_start, block_idx], device=self.device, dtype=torch.int64)
        get_world_group().send(t, dst=1)

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
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=self.cfg.negative_prompt or None)
        if self.is_lead:
            self.video_input_buffer.clear()
            self.sam_link.new_session()
            self.ctrl_buffer.commit()
        else:
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
        else:
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
        else:
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
        krea_input = (ft.permute(0, 3, 1, 2).to(self.pipe.dtype).div_(127.5).sub_(1.0).contiguous())
        return krea_input, raw

    # ---- loops ----
    def loop(self):
        if self.is_lead:
            self._loop_lead()
        else:
            self._loop_follow()

    def _loop_lead(self):
        while True:
            op = int(self.ctrl_buffer.recv())
            if op == 2:
                self._send_hdr(TERMINATE, 0, 0, 0)
                self._terminate()
                break
            if op == 1 and not self.session_started:
                self._send_hdr(START, 0, 0, 0)
                self._start()
                continue
            if op == 3 and self.session_started:
                self._send_hdr(RESET, 0, 0, 0)
                self._reset()
                continue
            polled = self._poll_input() if self.session_started else None
            if polled is None:
                self._send_hdr(IDLE, 0, 0, 0)
                time.sleep(0.003)
                continue
            krea_input, raw = polled
            T = int(raw.shape[0])
            self.sam_link.push_frames(raw)
            mask_start = self._sam_push_count
            self._sam_push_count += T
            block_idx = self.pipe._block_idx
            denoised, _ = denoise_chunk(self.pipe, krea_input)
            if denoised is None:
                self._send_hdr(IDLE, 0, 0, 0)
                continue
            # hand off to decode stage
            self._send_hdr(CHUNK, T, mask_start, block_idx)
            get_world_group().send(torch.from_numpy(raw).to(self.device), dst=1)
            get_world_group().send(denoised.contiguous(), dst=1)

    def _loop_follow(self):
        H, W = int(self.cfg.height), int(self.cfg.width)
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
            raw = get_world_group().recv(torch.Size([T, H, W, 3]), torch.uint8, src=0).cpu().numpy()
            denoised = get_world_group().recv(
                torch.Size([1, self.cfg.dit_config.out_channels, self.cfg.chunk_size,
                            self.cfg.latent_height, self.cfg.latent_width]), self.pipe.dtype, src=0)
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
