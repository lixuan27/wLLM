"""Krea orchestrator: runs the Krea-Realtime v2v pipeline while a decoupled
SAM worker segments the same frames concurrently on another GPU.

Honors the exact reference adapter contract (same shm buffers, same 1/2/3
control opcodes, same input/output cadence). The only behavioral change vs the
reference is *scheduling*: raw frames are handed to the SAM worker up-front so
SAM runs concurrently with the Krea DiT/VAE (IR: sam_segment || all Krea ops),
instead of after it. Frame I/O and compositing are ports of KreaSAMWorker so
numerics match.

Krea itself can be sequence-parallel across `krea_world` ranks (DiT frame-SP +
VAE decoder width-tiling handled inside the shared models via the SP group);
rank 0 owns all I/O + SAM coordination, other ranks are pure compute followers.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.reference.pipeline import KreaSAMPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options
from wllm.apps.krea_sam.backend.rocm.engine.sam_link import SamLink

logger = init_logger("orchestrator")
set_torch_options()

READY_MARKER = "KreaSAM backend READY"


class KreaOrchestrator:
    def __init__(self, cfg_path: str, sam_link_name: str, device: torch.device,
                 rank: int = 0, world: int = 1, broadcast_input=None):
        self.reference_cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self.device = device
        self.rank = rank
        self.world = world
        self._broadcast_input = broadcast_input  # fn(tensor)->tensor for SP ranks
        torch.cuda.set_device(self.device)

        self.is_lead = (rank == 0)
        H, W = int(self.cfg.height), int(self.cfg.width)

        # Create all shm FIRST (before the ~110s Krea load) so the SAM worker
        # (which loads faster) can attach to the link right away.
        if self.is_lead:
            self.video_buffer = SharedTensorBuffer(
                name=self.cfg.video_buffer_name, frame_shape=(H, W, 3),
                max_len=int(self.cfg.max_num_frames), dtype=np.uint8, create=True)
            self.video_input_buffer = SharedTensorBuffer(
                name=self.cfg.video_input_buffer_name, frame_shape=(H, W, 3),
                max_len=int(self.cfg.video_input_max_frames), dtype=np.uint8, create=True)
            self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
            self.sam_link = SamLink(sam_link_name, H, W, int(self.cfg.video_input_max_frames), create=True)

        self.pipe = KreaSAMPipeline(cfg=self.cfg, device=self.device)
        self.pipe.start_instance()

        self.session_started = False
        self.num_consumed_input_frames = 0
        self._output_frame_skip_frames = 0
        self._sam_push_count = 0

        self.warmup()
        if self.is_lead:
            # wait for SAM worker to finish loading
            t0 = time.time()
            while self.sam_link.sam_ready_epoch < 0:
                if time.time() - t0 > 1800:
                    raise TimeoutError("SAM worker never became ready")
                time.sleep(0.05)
            logger.info("Krea orchestrator up (rank0); SAM loaded")
            print(READY_MARKER, flush=True)

    # ---- session lifecycle ----
    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=self.cfg.negative_prompt or None)
        warmup_input = torch.zeros((self._required_input_frames(), 3, self.cfg.height, self.cfg.width),
                                   device=self.device, dtype=self.pipe.dtype)
        for _ in range(8):
            if self._pipe_step(warmup_input) is not None:
                break
        torch.cuda.synchronize(self.device)
        self.pipe.reset()

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.num_consumed_input_frames = 0
        self._sam_push_count = 0
        self._output_frame_skip_frames = max(0, int(self.cfg.vae_config.scale_factor_temporal) - 1)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=self.cfg.negative_prompt or None)
        if self.is_lead:
            self.video_input_buffer.clear()
            self.video_buffer.clear()
            self.sam_link.new_session()
            self.ctrl_buffer.commit()
        logger.info("orchestrator session started")

    def reset(self):
        self.session_started = False
        self.num_consumed_input_frames = 0
        self._output_frame_skip_frames = 0
        self._sam_push_count = 0
        self.pipe.reset()
        if self.is_lead:
            self.video_input_buffer.clear()
            self.video_buffer.clear()
            self.sam_link.new_session()   # bump epoch -> SAM closes old session
            self.ctrl_buffer.commit()
        logger.info("orchestrator session reset")

    def terminate(self):
        self.session_started = False
        self.pipe.terminate_instance()
        if self.is_lead:
            self.sam_link.signal_terminate()
            self.ctrl_buffer.unlink()
            self.video_input_buffer.unlink()
            self.video_buffer.unlink()
            time.sleep(0.3)
            self.sam_link.unlink()
        torch.cuda.empty_cache()

    # ---- Krea step (SP-aware via broadcast hook) ----
    def _pipe_step(self, krea_input: torch.Tensor):
        if self._broadcast_input is not None:
            krea_input = self._broadcast_input(krea_input)
        return self.pipe.step(krea_input)

    # ---- frame I/O (ports of worker.py) ----
    def _required_input_frames(self) -> int:
        return int(self.pipe.input_frames_for_next_step())

    def _resample_frames(self, frames: np.ndarray, target_length: int) -> np.ndarray:
        if len(frames) == target_length:
            return frames
        idx = np.round(np.linspace(0, len(frames) - 1, target_length)).astype(np.int64)
        return frames[idx]

    def _poll_input_frames(self) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        target = self._required_input_frames()
        available = max(0, int(self.video_input_buffer.num) - int(self.num_consumed_input_frames))
        if available < target:
            return None
        self.num_consumed_input_frames, frames = self.video_input_buffer.read(
            self.num_consumed_input_frames, available)
        if frames is None:
            return None
        selected = self._resample_frames(frames, target)
        raw_frames_np = np.ascontiguousarray(selected)
        frame_tensor = torch.from_numpy(selected).to(device=self.device, dtype=torch.uint8)
        krea_input = (frame_tensor.permute(0, 3, 1, 2).to(dtype=self.pipe.dtype)
                      .div_(127.5).sub_(1.0).contiguous())
        return krea_input, raw_frames_np

    @staticmethod
    def _composite(krea_frames, original_frames, masks):
        if masks is None:
            return krea_frames
        m3 = (masks > 0).astype(np.uint8)[:, :, :, None]
        return original_frames * m3 + krea_frames * (1 - m3)

    # ---- control ----
    def is_start(self):
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self):
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self):
        return int(self.ctrl_buffer.recv()) == 3

    # ---- main loop ----
    def loop(self):
        while True:
            if self.is_lead:
                op = int(self.ctrl_buffer.recv())
            else:
                op = 0
            if self._should_terminate(op):
                self.terminate()
                break
            self._handle_session_control(op)
            if not self.session_started:
                time.sleep(0.005)
                continue
            self._run_one_chunk()

    def _should_terminate(self, op) -> bool:
        return op == 2

    def _handle_session_control(self, op):
        if op == 1 and not self.session_started:
            self.start()
        elif op == 3 and self.session_started:
            self.reset()

    def _run_one_chunk(self):
        polled = self._poll_input_frames() if self.is_lead else None
        # (single-rank path: only lead has input; SP path overrides this method)
        if polled is None:
            time.sleep(0.002)
            return
        krea_input, raw_frames_np = polled
        n_pushed = int(raw_frames_np.shape[0])

        # hand raw frames to SAM up-front so it runs concurrently with Krea
        self.sam_link.push_frames(raw_frames_np)
        mask_start = self._sam_push_count
        self._sam_push_count += n_pushed

        krea_frames = self._pipe_step(krea_input)
        if krea_frames is None or len(krea_frames) == 0:
            return

        masks = self.sam_link.read_masks(mask_start, n_pushed)   # [n_pushed, H, W]

        # first-chunk causal-VAE warmup-frame drop (worker.loop tail)
        if self._output_frame_skip_frames > 0:
            skip = min(self._output_frame_skip_frames, int(krea_frames.shape[0]))
            krea_frames = krea_frames[skip:]
            raw_frames_np = raw_frames_np[skip:]
            if masks is not None:
                masks = masks[skip:]
            self._output_frame_skip_frames -= skip
            if krea_frames.shape[0] == 0:
                return

        n_out = int(krea_frames.shape[0])
        originals = raw_frames_np[:n_out] if raw_frames_np.shape[0] >= n_out else None
        if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
            self.video_buffer.write(krea_frames)
            return
        if masks is not None and masks.shape[0] >= n_out:
            masks = masks[:n_out]
        else:
            masks = None
        composited = self._composite(krea_frames, originals, masks)
        self.video_buffer.write(composited)
