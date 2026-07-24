"""IPC contract between the <app> frontend and any <app> backend.

The backend (reference or optimized) creates the shared-memory buffers named
in the app config; this adapter attaches to them (``create=False``) and is
the only way the frontend, or a test harness, drives a backend. Control is
an opcode buffer: 1 = start session, 2 = terminate, 3 = reset.

Replace the input/output buffers below with your app's actual streams and
shapes. ``wllm/apps/worldplay/adapter.py`` is a worked example.
"""

import numpy as np

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.rt_config import RTConfig


class AppAdapter:  # TODO rename to <App>Adapter
    def __init__(self, cfg_path: str):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)

        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=False)

        # TODO one SharedTensorBuffer per input stream the frontend pushes,
        # and one per output stream it reads, e.g.:
        # self.video_buffer = SharedTensorBuffer(
        #     name=self.cfg.video_buffer_name,
        #     frame_shape=(self.cfg.height, self.cfg.width, 3),
        #     max_len=int(self.cfg.max_num_frames),
        #     dtype=np.uint8,
        #     create=False,
        # )
        self.num_read = 0

    def start(self) -> None:
        if not self.ctrl_buffer.send(1, timeout_s=1800.0):
            raise TimeoutError("Timed out waiting for backend start ack")
        self.num_read = 0

    def terminate(self) -> None:
        self.ctrl_buffer.send(2, timeout_s=10.0)

    def reset(self) -> None:
        if not self.ctrl_buffer.send(3, timeout_s=1800.0):
            raise TimeoutError("Timed out waiting for backend reset ack")
        self.num_read = 0

    # TODO push_* methods for inputs and get_* methods for outputs, e.g.:
    # def get_frames(self):
    #     self.num_read, frame = self.video_buffer.read(self.num_read, 1)
    #     return frame[0] if frame is not None else None
