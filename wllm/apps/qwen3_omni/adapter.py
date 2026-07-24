"""Shared backend-facing IPC contract for the Qwen3-Omni application.

The frontend pushes UTF-8 text prompts into ``text_input_buffer`` and
reads streamed float32 audio frames out of ``audio_output_buffer``.
A small control buffer (``audio_meta_buffer``) publishes the active
sample rate so the frontend doesn't have to hard-code it. Session
control (start / terminate / reset) is a 3-state opcode buffer:

    1 = start session, 2 = terminate, 3 = reset session.

This adapter is shared by the user reference backend
(``wllm/apps/qwen3_omni/reference/``) and any agent-written optimised backend
under ``wllm/apps/qwen3_omni/backend/``; both must produce / consume the
same buffers under the names declared in the runtime config.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import yaml

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer


# Bytes per text-prompt frame. Anything longer is truncated.
TEXT_FRAME_BYTES = 8192


def encode_text_frame(text: str) -> np.ndarray:
    """Encode ``text`` into a fixed-size, NUL-padded byte frame."""
    raw = text.encode("utf-8")[:TEXT_FRAME_BYTES]
    out = np.zeros(TEXT_FRAME_BYTES, dtype=np.uint8)
    out[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def decode_text_frame(frame: np.ndarray) -> str:
    """Decode a NUL-padded UTF-8 byte frame back to a Python string."""
    raw = bytes(frame.astype(np.uint8).tobytes())
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    return raw.decode("utf-8", errors="replace")


class Qwen3OmniAdapter:
    """Lightweight client-side handle for the Qwen3-Omni worker."""

    # Control commands (kept in sync with Qwen3OmniWorker).
    _CMD_START = 1
    _CMD_TERMINATE = 2
    _CMD_RESET = 3

    def __init__(self, cfg_path: str):
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"qwen3-omni adapter config not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cfg = data

        self._audio_frame_samples: int = int(data.get("audio_frame_samples", 960))
        self._audio_max_chunks: int = int(data.get("audio_max_chunks", 8192))
        self._text_max_pending: int = int(data.get("text_max_pending", 16))
        self._default_sample_rate: int = int(data.get("audio_sample_rate", 24000))

        self.ctrl_buffer = SharedControlBuffer(
            data["ctrl_buffer_name"], create=False,
        )
        self.text_input_buffer = SharedTensorBuffer(
            name=data["text_input_buffer_name"],
            frame_shape=(TEXT_FRAME_BYTES,),
            max_len=self._text_max_pending,
            dtype=np.uint8,
            create=False,
        )
        self.audio_output_buffer = SharedTensorBuffer(
            name=data["audio_output_buffer_name"],
            frame_shape=(self._audio_frame_samples,),
            max_len=self._audio_max_chunks,
            dtype=np.float32,
            create=False,
        )
        self.audio_meta_buffer = SharedControlBuffer(
            data["audio_meta_buffer_name"], create=False,
        )

        self.num_played_audio_chunks: int = 0

    # ------------------------------------------------------------------
    # session control
    # ------------------------------------------------------------------

    def start(self, timeout_s: float = 1800.0) -> None:
        """Signal the worker to start a session and wait for ack.

        Pipeline startup includes loading three large engines, so the
        ack timeout is generous.
        """
        ack = self.ctrl_buffer.send(self._CMD_START, timeout_s=timeout_s)
        if not ack:
            raise TimeoutError("Timed out waiting for Qwen3-Omni worker start ack")
        self.num_played_audio_chunks = 0

    def reset(self, timeout_s: float = 1800.0) -> None:
        ack = self.ctrl_buffer.send(self._CMD_RESET, timeout_s=timeout_s)
        if not ack:
            raise TimeoutError("Timed out waiting for Qwen3-Omni worker reset ack")
        self.num_played_audio_chunks = 0

    def terminate(self, timeout_s: float = 10.0) -> None:
        self.ctrl_buffer.send(self._CMD_TERMINATE, timeout_s=timeout_s)

    # ------------------------------------------------------------------
    # text input
    # ------------------------------------------------------------------

    def push_text(self, text: str) -> None:
        """Submit a text prompt for the worker to respond to."""
        if not isinstance(text, str):
            raise TypeError(f"push_text expects str, got {type(text)}")
        if not text.strip():
            return
        self.text_input_buffer.write(encode_text_frame(text))

    # ------------------------------------------------------------------
    # audio output
    # ------------------------------------------------------------------

    def get_sample_rate(self) -> int:
        """Read the active output sample rate (Hz).

        Returns the worker's last-published rate if available, else the
        config default.
        """
        sr = int(self.audio_meta_buffer.recv())
        return sr if sr > 0 else self._default_sample_rate

    @property
    def audio_frame_samples(self) -> int:
        return self._audio_frame_samples

    def get_audio_chunks(self) -> Optional[np.ndarray]:
        """Non-blocking: return the next audio frame or None if not ready."""
        self.num_played_audio_chunks, chunk = self.audio_output_buffer.read(
            self.num_played_audio_chunks, 1,
        )
        if chunk is None:
            return None
        return chunk[0]

    def get_audio_chunks_blocked(self):
        """Return ``(next_index, frame_or_None)`` for the next available frame."""
        idx, chunk = self.audio_output_buffer.read(self.num_played_audio_chunks, 1)
        return idx, chunk[0] if chunk is not None else None

    def wait_for_audio_chunk(
        self,
        poll_interval: float = 0.005,
        timeout_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Block until a new audio frame is available (or timeout)."""
        deadline = None if timeout_s is None else time.time() + timeout_s
        while True:
            chunk = self.get_audio_chunks()
            if chunk is not None:
                return chunk
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(poll_interval)
