"""Execution context for the Krea+SAM IR.

The IR operators (``ops.py``) are thin nodes; the heavy lifting — the
DiT / VAE runners and the SAM stream predictor — lives here on a single
shared context object that the ``SequentialExecutor`` threads through
every ``execute`` call. This mirrors how the reference worker
(``wllm/apps/krea_sam/reference/worker.py``) and pipeline
(``wllm/apps/krea_sam/reference/pipeline.py``) hold their model handles.

Building the context constructs the *real* models (so the IR executes
the real computation, not a mock), runs the same warmup the reference
worker runs, and opens a SAM tracking session. The per-frame SAM logic
and the compositing logic are vendored verbatim from the reference
worker so the IR is byte-faithful to the reference's behavior.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from wllm.serving.rt_config import RTConfig
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.reference.pipeline import KreaSAMPipeline
from wllm.serving.utils.rand import set_global_seed


class KreaSamIRContext:
    """Holds the model runners + SAM predictor + per-chunk scalars.

    Constructed once per IR-executor session. ``init_session`` reseeds
    exactly like the reference worker's ``start()`` so noise draws line
    up chunk-for-chunk with the reference.
    """

    def __init__(self, cfg_path: str, device: str = "cuda:0", warmup: bool = True):
        self.reference_cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg: RTConfig = self.reference_cfg.to_runtime_config()
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        self.pipe = KreaSAMPipeline(cfg=self.cfg, device=self.device)
        self.pipe.start_instance()

        # SAM 3 (focus is the sam_disable=false case)
        self.sam_predictor = None
        self.sam_session_id: Optional[str] = None
        self._sam_prompt_set = False
        self._sam_frame_index = 0
        if not bool(self.cfg.sam_disable):
            from sam3.model_builder import build_sam3_stream_predictor
            self.sam_predictor = build_sam3_stream_predictor(device=str(self.device))

        self._output_frame_skip_frames = 0
        # chunk counter (drives the streaming encode `stream` flag and the
        # decoder's first-frame flag) and the nested Krea model executor,
        # both set by the harness / worker op.
        self.block_idx = 0
        self.krea_executor = None
        if warmup:
            self._warmup()

    def begin_chunk(self, idx: int) -> None:
        """Advance to chunk ``idx`` (keeps pipe._block_idx in sync so
        ``input_frames_for_next_step`` is correct)."""
        self.block_idx = idx
        self.pipe._block_idx = idx

    # ------------------------------------------------------------------
    # lifecycle (mirrors worker.warmup / worker.start)
    # ------------------------------------------------------------------

    def _required_input_frames(self) -> int:
        return int(self.pipe.input_frames_for_next_step())

    def _warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt or None,
        )
        warmup_input = torch.zeros(
            (self._required_input_frames(), 3, self.cfg.height, self.cfg.width),
            device=self.device, dtype=self.pipe.dtype,
        )
        for _ in range(8):
            if self.pipe.step(warmup_input) is not None:
                break
        torch.cuda.synchronize(self.device)
        self.pipe.reset()

    def start_session(self):
        """Begin a fresh inference session (reseed + SAM session)."""
        set_global_seed(self.cfg.seed)
        self._output_frame_skip_frames = max(
            0, int(self.cfg.vae_config.scale_factor_temporal) - 1,
        )
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt or None,
        )
        self._start_sam_session()

    def close(self):
        self._close_sam_session()
        if self.pipe.dit_runner is not None:
            self.pipe.dit_runner.clear()
        if self.pipe.vae_runner is not None:
            self.pipe.vae_runner.clear()

    # ------------------------------------------------------------------
    # SAM session + per-chunk inference (vendored from worker.py)
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
            pass
        self.sam_session_id = None
        self._sam_prompt_set = False
        self._sam_frame_index = 0

    def run_sam(self, frames_np: np.ndarray) -> Optional[np.ndarray]:
        """[T,H,W,3] uint8 RGB -> [T,H,W] uint8 mask (255=body). Vendored."""
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
                {"type": "add_frame", "session_id": self.sam_session_id,
                 "frame": frames_np[i]}
            )
            if not self._sam_prompt_set:
                resp = self.sam_predictor.handle_request(
                    {"type": "add_prompt", "session_id": self.sam_session_id,
                     "frame_index": self._sam_frame_index,
                     "text": self.cfg.sam_text_prompt}
                )
                self._sam_prompt_set = True
            else:
                resp = self.sam_predictor.handle_request(
                    {"type": "run_inference", "session_id": self.sam_session_id,
                     "frame_index": self._sam_frame_index}
                )
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
                        import cv2 as _cv2
                        cv2 = _cv2
                    m_np = cv2.resize(m_np.astype(np.float32), (W, H),
                                      interpolation=cv2.INTER_NEAREST)
                mask_union |= (m_np > mask_thresh)

            if dilate_px > 0:
                if cv2 is None:
                    import cv2 as _cv2
                    cv2 = _cv2
                k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=np.uint8)
                mask_union = cv2.dilate(mask_union.astype(np.uint8), k) > 0

            out[i] = mask_union.astype(np.uint8) * 255

        return out

    @staticmethod
    def composite(krea_frames: np.ndarray, original_frames: np.ndarray,
                  masks: Optional[np.ndarray]) -> np.ndarray:
        """Body pixel (mask>0) -> original; else -> krea. Vendored."""
        if masks is None:
            return krea_frames
        m3 = (masks > 0).astype(np.uint8)[:, :, :, None]
        return original_frames * m3 + krea_frames * (1 - m3)
