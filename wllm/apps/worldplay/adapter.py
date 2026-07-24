"""Shared backend-facing IPC contract for the WorldPlay application.

The frontend pushes WASD / arrow-key actions into the action buffer and
reads generated video frames back out of ``video_buffer``. Control is a
3-state opcode buffer:

    1 = start session, 2 = terminate, 3 = reset session.

This adapter is shared by the user reference backend
(``wllm/apps/worldplay/reference/``) and any agent-written optimised backend
under ``wllm/apps/worldplay/backend/``; both must produce / consume the
same buffers under the names declared in the runtime config.
"""

import os
import numpy as np
from PIL import Image

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.utils.image_process import resize_center_crop
from wllm.serving.rt_config import RTConfig


class WorldPlayAdapter:
    def __init__(self, cfg_path: str):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)

        self._height: int = self.cfg.height
        self._width: int = self.cfg.width
        self._num_frames: int = self.cfg.max_num_frames

        # Output: generated video frames the worker writes here.
        self.video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self._height, self._width, 3),
            max_len=int(self.cfg.max_num_frames),
            dtype=np.uint8,
            create=False,
        )

        # Control opcodes (1 = start, 2 = terminate, 3 = reset).
        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=False,
        )

        # Input: discrete WASD/arrow-key action codes (0..80) the
        # frontend pushes one-per-frame.
        self.action_buffer = SharedTensorBuffer(
            self.cfg.action_buffer_name,
            frame_shape=(1,),
            dtype=np.int64,
            max_len=int(self.cfg.max_num_actions),
            create=False,
        )

        self.num_played_frames = 0

        self.waiting_img = resize_center_crop(
            self.cfg.image_path,
            (self._width, self._height),
            normalize=False, CHW=False,
        )
        self._custom_image_path = os.path.join(
            "/tmp", f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png",
        )

    # ------------------------------------------------------------------
    # session control
    # ------------------------------------------------------------------

    def set_custom_image(self, image_data: bytes) -> np.ndarray:
        with open(self._custom_image_path, "wb") as f:
            f.write(image_data)
        self.waiting_img = resize_center_crop(
            self._custom_image_path,
            (self._width, self._height),
            normalize=False, CHW=False,
        )
        Image.fromarray(self.waiting_img).save(self._custom_image_path)
        return self.waiting_img

    def get_num_played_frames(self):
        return self.num_played_frames

    def get_waiting_image(self):
        return self.waiting_img

    def start(self, t: float = 0.005) -> None:
        _ = t
        ack = self.ctrl_buffer.send(1, timeout_s=1800.0)
        if not ack:
            raise TimeoutError("Timed out waiting for WorldPlay worker start ack")
        self.num_played_frames = 0

    def terminate(self) -> None:
        self.ctrl_buffer.send(2, timeout_s=10.0)
        try:
            os.remove(self._custom_image_path)
        except OSError:
            pass

    def reset(self, t: float = 0.005) -> None:
        _ = t
        ack = self.ctrl_buffer.send(3, timeout_s=1800.0)
        if not ack:
            raise TimeoutError("Timed out waiting for WorldPlay worker reset ack")
        self.num_played_frames = 0

    # ------------------------------------------------------------------
    # action I/O
    # ------------------------------------------------------------------

    def push_action(self, curr_action: int) -> None:
        if 0 <= curr_action <= 80:
            self.action_buffer.write(np.array([curr_action], dtype=np.int64))

    # ------------------------------------------------------------------
    # video output
    # ------------------------------------------------------------------

    def get_frames(self):
        """Non-blocking: return the next generated frame, or ``None``
        if the worker hasn't produced anything new yet."""
        self.num_played_frames, new_frame = self.video_buffer.read(
            self.num_played_frames, 1,
        )
        if new_frame is not None:
            return new_frame[0]
        return None
