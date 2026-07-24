"""Variant `sam_colocate` — in-process *threaded* SAM ‖ Krea on ONE GPU.

This is the single-GPU "parallel co-location" control: it runs SAM 3 and the
Krea-Realtime v2v stage concurrently on one GPU to isolate the "separate-GPU"
lever from mere "async overlap". The earlier implementation ran them as two
*processes* sharing the GPU; on B200 that reproducibly crashed Krea with a
``CUDA error: unspecified launch failure`` once both ran concurrently — two
independent CUDA contexts + two independent caching allocators on one GPU,
which (under cudnn.benchmark + the cold-cache autotune on the first chunk)
over-subscribes and faults. (It worked on H200; not on B200.)

This version keeps the *single GPU, concurrent SAM ‖ Krea* intent but runs both
in **one process** (one CUDA context, one allocator — the reference's working
model). Krea runs in a worker thread on its own CUDA stream; SAM runs
concurrently in the main thread on the default stream (SAM's internal autocast
trips a bf16/fp32 conv mismatch when driven from a worker thread, so it must run
in the main thread — its warmup proves it works there). The Python GIL
serializes the Python halves, so the overlap is partial (GPU-bound work overlaps
while CUDA calls release the GIL); as a single-GPU compute-bound control, the
expected result is ≈ the 1-GPU reference — the point of the variant.

Vendored from the reference worker (wllm/apps/krea_sam/reference/worker.py): the
adapter contract (shm buffers + 3-state control opcode), the SAM 3 per-chunk
inference, the warmup-frame drop, and the background-swap composite are
byte-identical; only the per-chunk *scheduling* (threaded vs sequential) and an
added SAM warmup differ. Launched as a single-process backend by
the launcher exactly like the reference (CUDA_VISIBLE_DEVICES pins the one
GPU; the worker sees it as cuda:0).
"""

from __future__ import annotations

import threading
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


class Worker:
    """In-process threaded SAM ‖ Krea co-location worker (one GPU)."""

    VARIANT = "sam_colocate"

    def __init__(self, cfg_path: str):
        self.reference_cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self._init_worker()
        # Krea runs on its own stream (in a worker thread) so its GPU work can
        # overlap SAM, which runs on the default stream in the main thread.
        self.krea_stream = torch.cuda.Stream(device=self.device)
        self.warmup()
        # Readiness marker users (and the frontend docs) wait on for a
        # non-reference backend; the READY token is what users wait for.
        logger.info("Krea+SAM3 threaded co-locate worker started (variant=sam_colocate, one GPU)")
        print("KreaSAM backend READY", flush=True)

    # ------------------------------------------------------------------
    # construction  (vendored from the reference worker)
    # ------------------------------------------------------------------
    def _create_pipeline(self):
        return KreaSAMPipeline(cfg=self.cfg, device=self.device)

    def _init_worker(self):
        if self.cfg.device != "cuda":
            raise ValueError(f"expected cfg.device='cuda', got {self.cfg.device!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("threaded co-locate worker requires a visible CUDA GPU.")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.pipe = self._create_pipeline()
        self.pipe.start_instance()

        self.video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=int(self.cfg.max_num_frames), dtype=np.uint8, create=True)
        self.video_input_buffer = SharedTensorBuffer(
            name=self.cfg.video_input_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=int(self.cfg.video_input_max_frames), dtype=np.uint8, create=True)
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)

        self.session_started = False
        self.num_consumed_input_frames = 0
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
        logger.info("Loading SAM 3 stream predictor on %s (text prompt=%r)",
                    self.device, self.cfg.sam_text_prompt)
        self.sam_predictor = build_sam3_stream_predictor(device=str(self.device))

    # ------------------------------------------------------------------
    # session lifecycle  (vendored; warmup additionally warms SAM)
    # ------------------------------------------------------------------
    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt,
                               negative_prompt=self.cfg.negative_prompt or None)
        # Run several full chunks so every per-chunk path is compiled before the
        # first live chunk (stream=False->True encode, growing DiT context, VAE
        # is_first True->False decode), not just block 0. Recompute the frame
        # count per chunk since it changes after block 0.
        for _ in range(int(self.cfg.context_window_size) + 3):
            warmup_input = torch.zeros(
                (self._required_input_frames(), 3, self.cfg.height, self.cfg.width),
                device=self.device, dtype=self.pipe.dtype)
            self.pipe.step(warmup_input)
        torch.cuda.synchronize(self.device)
        self.pipe.reset()

        # Warm SAM sequentially in a throwaway session so its kernels are
        # compiled/cudnn-tuned BEFORE the concurrent phase — otherwise the first
        # live chunk would tune SAM and Krea at the same time on the same GPU.
        if self.sam_predictor is not None:
            H, W = int(self.cfg.height), int(self.cfg.width)
            n_warm = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
            self._start_sam_session()
            for _ in range(max(1, n_warm)):
                try:
                    self._run_sam(np.zeros((1, H, W, 3), dtype=np.uint8))
                except Exception:
                    logger.warning("SAM warmup frame failed (continuing)", exc_info=True)
                    break
            self._close_sam_session()
            torch.cuda.synchronize(self.device)

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.num_consumed_input_frames = 0
        self._output_frame_skip_frames = max(0, int(self.cfg.vae_config.scale_factor_temporal) - 1)
        self.pipe.init_session(prompt=self.cfg.prompt,
                               negative_prompt=self.cfg.negative_prompt or None)
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
    # SAM 3 session + per-chunk inference  (vendored verbatim)
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
                {"type": "close_session", "session_id": self.sam_session_id})
        except Exception:
            logger.warning("SAM 3 close_session failed", exc_info=True)
        self.sam_session_id = None
        self._sam_prompt_set = False
        self._sam_frame_index = 0

    def _run_sam(self, frames_np: np.ndarray) -> Optional[np.ndarray]:
        if self.sam_predictor is None or self.sam_session_id is None:
            return None
        T, H, W, _ = frames_np.shape
        out = np.zeros((T, H, W), dtype=np.uint8)
        score_thresh = float(self.cfg.sam_min_score)
        mask_thresh = float(self.cfg.sam_mask_threshold)
        dilate_px = int(self.cfg.sam_dilate_pixels)
        cv2 = None
        for i in range(T):
            self.sam_predictor.handle_request(
                {"type": "add_frame", "session_id": self.sam_session_id, "frame": frames_np[i]})
            if not self._sam_prompt_set:
                resp = self.sam_predictor.handle_request(
                    {"type": "add_prompt", "session_id": self.sam_session_id,
                     "frame_index": self._sam_frame_index, "text": self.cfg.sam_text_prompt})
                self._sam_prompt_set = True
            else:
                resp = self.sam_predictor.handle_request(
                    {"type": "run_inference", "session_id": self.sam_session_id,
                     "frame_index": self._sam_frame_index})
            self._sam_frame_index += 1
            outputs = (resp or {}).get("outputs") or {}
            raw_masks = outputs.get("out_binary_masks")
            raw_probs = outputs.get("out_probs")
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
                    m_np = cv2.resize(m_np.astype(np.float32), (W, H),
                                      interpolation=cv2.INTER_NEAREST)
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
    def _composite(krea_frames: np.ndarray, original_frames: np.ndarray,
                   masks: Optional[np.ndarray]) -> np.ndarray:
        if masks is None:
            return krea_frames
        m3 = (masks > 0).astype(np.uint8)[:, :, :, None]
        return original_frames * m3 + krea_frames * (1 - m3)

    # ------------------------------------------------------------------
    # frame I/O  (vendored verbatim)
    # ------------------------------------------------------------------
    def _required_input_frames(self) -> int:
        return int(self.pipe.input_frames_for_next_step())

    def _resample_frames(self, frames: np.ndarray, target_length: int) -> np.ndarray:
        if len(frames) == target_length:
            return frames
        indices = np.round(np.linspace(0, len(frames) - 1, target_length)).astype(np.int64)
        return frames[indices]

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

    # ------------------------------------------------------------------
    # control buffer + main loop  (threaded SAM ‖ Krea)
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

            # --- run Krea v2v and SAM segmentation CONCURRENTLY, sharing one
            #     CUDA context + allocator. Krea runs in a worker thread (its own
            #     stream); SAM runs in THIS (main) thread on purpose: SAM's
            #     internal autocast trips a bf16/fp32 conv mismatch when it is
            #     driven from a worker thread, but works in the main thread (as
            #     its warmup does). Krea has no such issue and runs fine threaded.
            res = {"krea": None, "krea_err": None}

            def _run_krea():
                try:
                    with torch.cuda.stream(self.krea_stream):
                        res["krea"] = self.pipe.step(krea_input)
                except Exception as e:  # surface a Krea/CUDA failure to the harness
                    res["krea_err"] = e

            tk = threading.Thread(target=_run_krea, name="krea")
            tk.start()
            try:
                masks = self._run_sam(raw_frames_np)  # main thread, concurrent with Krea
            except Exception:
                logger.warning("SAM 3 inference failed; raw Krea output", exc_info=True)
                masks = None
            tk.join()
            torch.cuda.synchronize(self.device)  # Krea stream done before compositing
            if res["krea_err"] is not None:
                raise res["krea_err"]

            krea_frames = res["krea"]
            if krea_frames is None or len(krea_frames) == 0:
                # Krea still priming; nothing to emit (SAM tracking already
                # advanced over these frames, matching the prior co-locate engine).
                continue

            # drop the causal VAE's session-warmup frames (1:1 align output↔input)
            if self._output_frame_skip_frames > 0:
                skip = min(self._output_frame_skip_frames, int(krea_frames.shape[0]))
                krea_frames = krea_frames[skip:]
                raw_frames_np = raw_frames_np[skip:]
                if masks is not None:
                    masks = masks[skip:]
                self._output_frame_skip_frames -= skip
                if krea_frames.shape[0] == 0:
                    continue

            # composite: Krea-stylised background + original body
            n_out = int(krea_frames.shape[0])
            originals = raw_frames_np[:n_out] if raw_frames_np.shape[0] >= n_out else None
            if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
                self.video_buffer.write(krea_frames)
                continue
            if masks is not None and masks.shape[0] >= n_out:
                masks = masks[:n_out]
            else:
                masks = None
            self.video_buffer.write(self._composite(krea_frames, originals, masks))
