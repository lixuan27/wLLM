import numpy as np

from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.rt_config import RTConfig


class LongLiveAdapter:
    def __init__(self, cfg_path: str):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)

        self._height: int = self.cfg.height
        self._width: int = self.cfg.width

        self.video_buffer = SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self._height, self._width, 3),
            max_len=self.cfg.max_num_frames,
            dtype=np.uint8,
            create=False,
        )

        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=False,
        )

        self.audio_buffer = SharedTensorBuffer(
            self.cfg.audio_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=False,
        )

        if self.cfg.signal_buffer_name:
            self.signal_buffer = SharedControlBuffer(
                self.cfg.signal_buffer_name, create=False,
            )
        else:
            self.signal_buffer = None

        self.num_played_frames = 0

    def start(self, t: float = 0.005):
        _ = t
        ack = self.ctrl_buffer.send(1, timeout_s=1800.0)
        if not ack:
            raise TimeoutError("Timed out waiting for LongLive worker start ack")
        self.num_played_frames = 0

    def terminate(self):
        self.ctrl_buffer.send(2, timeout_s=10.0)

    def reset(self, t: float = 0.005):
        _ = t
        ack = self.ctrl_buffer.send(3, timeout_s=1800.0)
        if not ack:
            raise TimeoutError("Timed out waiting for LongLive worker reset ack")
        self.num_played_frames = 0

    def push_audio(self, audio_chunk: np.ndarray):
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        target = int(self.cfg.audio_frame_samples)
        if chunk.shape[0] < target:
            padded = np.zeros((target,), dtype=np.float32)
            padded[: chunk.shape[0]] = chunk
            chunk = padded
        elif chunk.shape[0] > target:
            chunk = chunk[:target]
        self.audio_buffer.write(chunk)

    def enable_microphone(self):
        if self.signal_buffer is not None:
            self.signal_buffer.send(1)

    def disable_microphone(self):
        if self.signal_buffer is not None:
            self.audio_buffer.clear()
            self.signal_buffer.send(2)

    def get_frames(self):
        self.num_played_frames, new_frame = self.video_buffer.read(
            self.num_played_frames, 1,
        )
        if new_frame is not None:
            return new_frame[0]
        return None

    def get_frames_blocked(self):
        idx, frame = self.video_buffer.read(self.num_played_frames, 1)
        return idx, frame[0] if frame is not None else None

    def commit(self, frame_next_index: int) -> bool:
        self.num_played_frames = frame_next_index
        return True
