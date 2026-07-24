"""Reference single-GPU sequential Krea-Realtime + SAM 3 worker.

    poll input frames
        -> Krea v2v step (DiT denoise + VAE decode)
        -> SAM 3 segmentation on the same input frames
        -> composite (SAM-mask = body pixel: keep original; else: keep Krea)
        -> write composited frames to the output video buffer
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

logger = init_logger(__name__)
set_torch_options()


class KreaSAMWorker:
    """Sequential single-GPU reference worker for the Krea+SAM app."""

    def __init__(self, cfg_path: str):
        self.reference_cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self._init_worker()
        self.warmup()
        logger.info("KreaSAM backend READY")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _create_pipeline(self):
        return KreaSAMPipeline(cfg=self.cfg, device=self.device)

    def _init_worker(self):
        if self.cfg.device != "cuda":
            raise ValueError(
                f"The reference worker expects cfg.device='cuda', got {self.cfg.device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The reference worker requires a visible CUDA GPU.")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.pipe = self._create_pipeline()
        self.pipe.start_instance()

        # Output buffer the adapter reads from (composited frames).
        self.video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=int(self.cfg.max_num_frames),
            dtype=np.uint8,
            create=True,
        )

        # Webcam input buffer the adapter writes to.
        self.video_input_buffer = SharedTensorBuffer(
            name=self.cfg.video_input_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=int(self.cfg.video_input_max_frames),
            dtype=np.uint8,
            create=True,
        )

        # Control channel (start / terminate / reset opcodes).
        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=True,
        )

        self.session_started = False
        self.num_consumed_input_frames = 0
        # Number of leading decoded frames to drop at session start (the
        # causal VAE's warmup frames). Set when a session starts.
        self._output_frame_skip_frames = 0

        self._init_sam3()

    def _init_sam3(self):
        self.sam_predictor = None
        self.sam_session_id: Optional[str] = None
        self._sam_prompt_set = False
        self._sam_frame_index = 0

        if bool(self.cfg.sam_disable):
            logger.info("SAM 3 disabled by config; running plain Krea passthrough")
            return

        from sam3.model_builder import build_sam3_stream_predictor  # type: ignore[import-not-found]

        logger.info(
            "Loading SAM 3 stream predictor on %s (text prompt=%r)",
            self.device, self.cfg.sam_text_prompt,
        )
        self.sam_predictor = build_sam3_stream_predictor(device=str(self.device))

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt or None,
        )

        # Push several full chunks of zero frames through Krea so EVERY per-chunk
        # path is compiled before the first live frame, not just the first chunk:
        # the streaming encoder switches stream=False -> stream=True after chunk 0,
        # the DiT clean-context cache grows over the first context_window_size
        # chunks (each a new autotune shape), and the VAE decode uses is_first
        # True then False. Recompute the frame count per chunk (it changes after
        # block 0).
        for _ in range(int(self.cfg.context_window_size) + 3):
            warmup_input = torch.zeros(
                (self._required_input_frames(), 3, self.cfg.height, self.cfg.width),
                device=self.device,
                dtype=self.pipe.dtype,
            )
            self.pipe.step(warmup_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.pipe.reset()

        # Warm SAM 3 too, in a throwaway session, so the first live frame pays no
        # SAM cold start (cudnn.benchmark tuning on first use). The throwaway
        # session is closed here; the real session opens fresh in start().
        if self.sam_predictor is not None:
            self._start_sam_session()
            n_frames = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
            for _ in range(max(1, n_frames)):
                try:
                    self._run_sam(np.zeros(
                        (1, int(self.cfg.height), int(self.cfg.width), 3), dtype=np.uint8))
                except Exception:
                    logger.warning("SAM 3 warmup frame failed (continuing)", exc_info=True)
                    break
            self._close_sam_session()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.num_consumed_input_frames = 0
        self._output_frame_skip_frames = max(
            0, int(self.cfg.vae_config.scale_factor_temporal) - 1,
        )
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt or None,
        )
        self.video_input_buffer.clear()
        self.video_buffer.clear()
        self._start_sam_session()
        self.ctrl_buffer.commit()
        logger.info("Krea+SAM3 session started")

    def reset(self):
        self.session_started = False
        self.num_consumed_input_frames = 0
        self._output_frame_skip_frames = 0
        self.pipe.reset()
        self.video_input_buffer.clear()
        self.video_buffer.clear()
        self._close_sam_session()
        self.ctrl_buffer.commit()
        logger.info("Krea+SAM3 session reset")

    def terminate(self):
        # Idempotent: loop() calls this on a graceful ctrl-buffer TERM, and the
        # launcher's finally calls it on Ctrl+C / exit -- both must be safe.
        if getattr(self, "_terminated", False):
            return
        self._terminated = True
        self.session_started = False
        self.pipe.terminate_instance()
        self._close_sam_session()
        self.ctrl_buffer.unlink()
        self.video_input_buffer.unlink()
        self.video_buffer.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SAM 3 session + per-chunk inference
    # ------------------------------------------------------------------

    def _start_sam_session(self):
        if self.sam_predictor is None:
            return
        self._close_sam_session()
        resp = self.sam_predictor.handle_request({"type": "start_session"})
        self.sam_session_id = resp["session_id"]
        self._sam_prompt_set = False
        self._sam_frame_index = 0

    def _close_sam_session(self):
        if self.sam_predictor is None or self.sam_session_id is None:
            return
        try:
            self.sam_predictor.handle_request(
                {"type": "close_session", "session_id": self.sam_session_id}
            )
        except Exception:
            logger.warning("SAM 3 close_session failed", exc_info=True)
        self.sam_session_id = None
        self._sam_prompt_set = False
        self._sam_frame_index = 0

    def _run_sam(self, frames_np: np.ndarray) -> Optional[np.ndarray]:
        """Run SAM 3 over a batch of webcam frames sequentially.

        Args:
            frames_np: ``[T, H, W, 3]`` uint8 RGB.

        Returns:
            ``[T, H, W]`` uint8 mask, ``255`` = body pixel, ``0`` =
            background, or ``None`` if SAM is disabled / no session.
        """
        if self.sam_predictor is None or self.sam_session_id is None:
            return None

        T, H, W, _ = frames_np.shape
        out = np.zeros((T, H, W), dtype=np.uint8)
        score_thresh = float(self.cfg.sam_min_score)
        mask_thresh = float(self.cfg.sam_mask_threshold)
        dilate_px = int(self.cfg.sam_dilate_pixels)
        cv2 = None  # imported lazily if we need resize / dilate

        for i in range(T):
            self.sam_predictor.handle_request(
                {
                    "type": "add_frame",
                    "session_id": self.sam_session_id,
                    "frame": frames_np[i],
                }
            )
            if not self._sam_prompt_set:
                resp = self.sam_predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": self.sam_session_id,
                        "frame_index": self._sam_frame_index,
                        "text": self.cfg.sam_text_prompt,
                    }
                )
                self._sam_prompt_set = True
            else:
                resp = self.sam_predictor.handle_request(
                    {
                        "type": "run_inference",
                        "session_id": self.sam_session_id,
                        "frame_index": self._sam_frame_index,
                    }
                )
            self._sam_frame_index += 1

            outputs = (resp or {}).get("outputs") or {}
            raw_masks = outputs.get("out_binary_masks")
            raw_probs = outputs.get("out_probs")
            # SAM returns numpy arrays for these; ``raw or []`` raises on
            # multi-element numpy arrays, so use explicit None / length checks.
            masks = list(raw_masks) if raw_masks is not None and len(raw_masks) > 0 else []
            probs = list(raw_probs) if raw_probs is not None and len(raw_probs) > 0 else []
            if not masks or not probs:
                continue

            mask_union = np.zeros((H, W), dtype=bool)
            for m, s in zip(masks, probs):
                try:
                    score = float(s)
                except (TypeError, ValueError):
                    score = 0.0
                if score < score_thresh:
                    continue
                m_np = np.asarray(m)
                if m_np.ndim == 3 and m_np.shape[0] == 1:
                    m_np = m_np[0]
                if m_np.shape != (H, W):
                    if cv2 is None:
                        import cv2 as _cv2  # type: ignore[import-not-found]
                        cv2 = _cv2
                    m_np = cv2.resize(
                        m_np.astype(np.float32),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_union |= (m_np > mask_thresh)

            if dilate_px > 0:
                if cv2 is None:
                    import cv2 as _cv2  # type: ignore[import-not-found]
                    cv2 = _cv2
                k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=np.uint8)
                mask_union = cv2.dilate(mask_union.astype(np.uint8), k) > 0

            out[i] = mask_union.astype(np.uint8) * 255

        return out

    @staticmethod
    def _composite(
        krea_frames: np.ndarray,
        original_frames: np.ndarray,
        masks: Optional[np.ndarray],
    ) -> np.ndarray:
        """Replace body pixels in ``krea_frames`` with the corresponding
        pixels from ``original_frames``."""
        if masks is None:
            return krea_frames
        m3 = (masks > 0).astype(np.uint8)[:, :, :, None]
        return original_frames * m3 + krea_frames * (1 - m3)

    # ------------------------------------------------------------------
    # frame I/O
    # ------------------------------------------------------------------

    def _required_input_frames(self) -> int:
        # The pipeline's streaming encoder needs a special first-chunk frame
        # count (1 + (chunk_size - 1) * scale_t), then chunk_size * scale_t.
        return int(self.pipe.input_frames_for_next_step())

    def _resample_frames(self, frames: np.ndarray, target_length: int) -> np.ndarray:
        if len(frames) == target_length:
            return frames
        indices = np.round(np.linspace(0, len(frames) - 1, target_length)).astype(np.int64)
        return frames[indices]

    def _poll_input_frames(self) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        target = self._required_input_frames()
        available = max(
            0,
            int(self.video_input_buffer.num) - int(self.num_consumed_input_frames),
        )
        if available < target:
            return None

        self.num_consumed_input_frames, frames = self.video_input_buffer.read(
            self.num_consumed_input_frames, available,
        )
        if frames is None:
            return None

        selected = self._resample_frames(frames, target)
        raw_frames_np = np.ascontiguousarray(selected)
        frame_tensor = torch.from_numpy(selected).to(device=self.device, dtype=torch.uint8)
        krea_input = (
            frame_tensor.permute(0, 3, 1, 2)
            .to(dtype=self.pipe.dtype)
            .div_(127.5)
            .sub_(1.0)
            .contiguous()
        )
        return krea_input, raw_frames_np

    # ------------------------------------------------------------------
    # control buffer + main loop
    # ------------------------------------------------------------------

    def is_start(self):
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self):
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self):
        return int(self.ctrl_buffer.recv()) == 3

    def loop(self):
        while True:
            if self.is_terminate():
                self.terminate()
                break
            if self.is_start() and not self.session_started:
                self.start()
            elif self.is_reset() and self.session_started:
                self.reset()

            if not self.session_started:
                time.sleep(0.005)
                continue

            polled = self._poll_input_frames()
            if polled is None:
                time.sleep(0.002)
                continue

            krea_input, raw_frames_np = polled

            # 1. Krea v2v: decode this chunk into uint8 frames.
            krea_frames = self.pipe.step(krea_input)
            if krea_frames is None or len(krea_frames) == 0:
                # Streaming encoder didn't yield a full latent chunk yet;
                # nothing to emit this round.
                continue

            # 2. SAM 3: segment "person" on the same input frames. Run it on
            #    every input frame (so SAM's tracking memory advances over all
            #    of them) even if some output frames are dropped below.
            try:
                masks = self._run_sam(raw_frames_np)
            except Exception:
                logger.warning(
                    "SAM 3 inference failed; falling back to raw Krea output",
                    exc_info=True,
                )
                masks = None

            # 2b. Drop the causal VAE's session-warmup frames so the emitted
            #     video lines up 1:1 with the input frames (and their SAM
            #     masks). The WAN VAE's first decoded latent of a session
            #     yields a single pixel frame, so the first chunk decodes
            #     `scale_factor_temporal - 1` fewer real frames than later
            #     chunks; the official Krea server drops exactly these
            #     (`if idx == 0: pixels = pixels[:, 3:]`). After the drop, the
            #     j-th krea frame, raw frame, and mask still share an index.
            if self._output_frame_skip_frames > 0:
                skip = min(self._output_frame_skip_frames, int(krea_frames.shape[0]))
                krea_frames = krea_frames[skip:]
                raw_frames_np = raw_frames_np[skip:]
                if masks is not None:
                    masks = masks[skip:]
                self._output_frame_skip_frames -= skip
                if krea_frames.shape[0] == 0:
                    continue

            # 3. Composite: Krea-stylised background + original body.
            n_out = int(krea_frames.shape[0])
            originals = (
                raw_frames_np[:n_out]
                if raw_frames_np.shape[0] >= n_out
                else None
            )
            if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
                self.video_buffer.write(krea_frames)
                continue
            if masks is not None and masks.shape[0] >= n_out:
                masks = masks[:n_out]
            else:
                masks = None

            composited = self._composite(krea_frames, originals, masks)
            self.video_buffer.write(composited)
