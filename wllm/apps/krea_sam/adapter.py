"""Shared backend-facing IPC contract for the Krea+SAM application.

The frontend writes raw webcam frames to ``video_input_buffer`` and
reads composited (Krea-stylised + SAM-preserved-body) frames back out
of ``video_buffer``. Control is a 3-state opcode buffer:

    1 = start session, 2 = terminate, 3 = reset session.

This adapter is shared by the user reference backend
(``wllm/apps/krea_sam/reference/``) and any agent-written optimised backend
under ``wllm/apps/krea_sam/backend/``; both must produce / consume the
same buffers under the names declared in the runtime config.
"""

import numpy as np
from PIL import Image

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.rt_config import RTConfig


def _resize_center_crop_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Scale-up + center-crop ``frame`` to ``(target_height, target_width)``.

    Webcams typically deliver 720p / 1080p RGB; the Krea pipeline runs
    at the resolution declared in the config (e.g. 480x832), so we
    have to normalise frames before they hit the input shm buffer."""
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    orig_w, orig_h = image.size
    scale = max(target_width / orig_w, target_height / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)

    left = max(0, (new_w - target_width) // 2)
    top = max(0, (new_h - target_height) // 2)
    image = image.crop((left, top, left + target_width, top + target_height))
    return np.asarray(image, dtype=np.uint8)


class KreaSAMAdapter:
    def __init__(self, cfg_path: str):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)

        self._height: int = self.cfg.height
        self._width: int = self.cfg.width

        # Output: the composited video frames (Krea background + original
        # body where SAM segmented "person").
        self.video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self._height, self._width, 3),
            max_len=int(self.cfg.max_num_frames),
            dtype=np.uint8,
            create=False,
        )

        # Input: raw webcam frames the frontend feeds in.
        self.video_input_buffer = SharedTensorBuffer(
            name=self.cfg.video_input_buffer_name,
            frame_shape=(self._height, self._width, 3),
            max_len=int(self.cfg.video_input_max_frames),
            dtype=np.uint8,
            create=False,
        )

        # Control opcodes (1 = start, 2 = terminate, 3 = reset).
        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=False,
        )

        self.num_played_frames = 0
        self.num_pushed_input_frames = 0

    # ------------------------------------------------------------------
    # session control
    # ------------------------------------------------------------------

    def start(self, t: float = 0.005) -> None:
        _ = t
        ack = self.ctrl_buffer.send(1, timeout_s=1800.0)
        if not ack:
            raise TimeoutError("Timed out waiting for Krea+SAM worker start ack")
        self.num_played_frames = 0
        self.num_pushed_input_frames = 0

    def terminate(self) -> None:
        self.ctrl_buffer.send(2, timeout_s=10.0)

    def reset(self, t: float = 0.005) -> None:
        _ = t
        ack = self.ctrl_buffer.send(3, timeout_s=1800.0)
        if not ack:
            raise TimeoutError("Timed out waiting for Krea+SAM worker reset ack")
        self.num_played_frames = 0
        self.num_pushed_input_frames = 0

    # ------------------------------------------------------------------
    # video I/O
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a single uint8 RGB frame into the input buffer. Frames
        whose ``(H, W)`` doesn't match the configured pipeline resolution
        are scale-up + center-cropped to fit, so frontends can pipe raw
        camera frames in directly."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"push_frame expects HWC RGB, got shape={tuple(frame.shape)}"
            )
        resized = _resize_center_crop_frame(frame, self._width, self._height)
        self.video_input_buffer.write(np.ascontiguousarray(resized))
        self.num_pushed_input_frames += 1

    def get_frames(self):
        """Non-blocking: return the next composited frame, or ``None``
        if the worker hasn't produced anything new yet."""
        self.num_played_frames, new_frame = self.video_buffer.read(
            self.num_played_frames, 1,
        )
        if new_frame is not None:
            return new_frame[0]
        return None

    def get_frames_blocked(self):
        """Peek the next frame without advancing the read cursor.
        Returns ``(next_index, frame_or_None)``."""
        idx, frame = self.video_buffer.read(self.num_played_frames, 1)
        return idx, frame[0] if frame is not None else None

    def commit(self, frame_next_index: int = None) -> bool:
        """Advance the read cursor to ``frame_next_index``. Used by
        publishers that peek-then-emit so playback stays in sync with
        what was actually sent over the wire."""
        if frame_next_index is None:
            return False
        self.num_played_frames = frame_next_index
        return True
