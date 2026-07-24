import os
import numpy as np
from PIL import Image
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.utils.image_process import resize_center_crop
from wllm.serving.rt_config import RTConfig


# start()/reset() ack timeout. The worker commits the ack only AFTER init_session
# (which encodes the reference image -> a first-time VAE-encoder torch.compile).
# The combined_stream_pp_vae cluster variant warms only the decode path, so its
# start() pays a cold encoder compile that can far exceed the old hardcoded 10s.
# Separate from BENCH_READY_TIMEOUT (which only governs the worker-started marker).
_ACK_TIMEOUT_S = float(os.environ.get("LA_START_ACK_TIMEOUT", "1800.0"))


class LiveAvatarAdapter:
    def __init__(self, cfg_path: str):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)

        self._height: int = self.cfg.height
        self._width: int = self.cfg.width
        self._num_frames: int = self.cfg.max_num_frames

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

        self.audio_output_buffer = SharedTensorBuffer(
            self.cfg.audio_output_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=False,
        )

        # Signal buffer (microphone enable/disable)
        if self.cfg.signal_buffer_name:
            self.signal_buffer = SharedControlBuffer(
                self.cfg.signal_buffer_name, create=False,
            )
        else:
            self.signal_buffer = None

        self.num_played_frames = 0
        self.num_played_audio_chunks = 0

        self.waiting_img = resize_center_crop(
            self.cfg.image_path,
            (self._width, self._height),
            normalize=False, CHW=False,
        )
        self._custom_image_path = os.path.join(
            "/tmp", f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png",
        )

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

    def get_waiting_image(self):
        return self.waiting_img

    def start(self, t=0.005):
        _ = t
        ack = self.ctrl_buffer.send(1, timeout_s=_ACK_TIMEOUT_S)
        if not ack:
            raise TimeoutError("Timed out waiting for LiveAvatar worker start ack")
        self.num_played_frames = 0
        self.num_played_audio_chunks = 0

    def terminate(self):
        self.ctrl_buffer.send(2, timeout_s=10.0)
        try:
            os.remove(self._custom_image_path)
        except OSError:
            pass

    def reset(self, t=0.005):
        _ = t
        ack = self.ctrl_buffer.send(3, timeout_s=_ACK_TIMEOUT_S)
        if not ack:
            raise TimeoutError("Timed out waiting for LiveAvatar worker reset ack")
        self.num_played_frames = 0
        self.num_played_audio_chunks = 0

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

    def get_audio_chunks(self):
        self.num_played_audio_chunks, chunk = self.audio_output_buffer.read(
            self.num_played_audio_chunks, 1,
        )
        if chunk is not None:
            return chunk[0]
        return None

    def get_audio_chunks_blocked(self):
        idx, chunk = self.audio_output_buffer.read(
            self.num_played_audio_chunks, 1,
        )
        return idx, chunk[0] if chunk is not None else None

    def get_first_image(self) -> np.ndarray:
        return self.waiting_img

    def get_first_audio_chunk(self) -> np.ndarray:
        return np.zeros(int(self.cfg.audio_frame_samples), dtype=np.float32)

    def commit(self, frame_next_index: int = None, audio_next_index: int = None) -> bool:
        if frame_next_index is None or audio_next_index is None:
            return False

        if frame_next_index != audio_next_index:
            return False

        self.num_played_frames = frame_next_index
        self.num_played_audio_chunks = audio_next_index
        return True
