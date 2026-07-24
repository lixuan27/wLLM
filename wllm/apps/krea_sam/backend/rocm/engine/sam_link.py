"""Shared-memory link between the Krea orchestrator and a decoupled SAM
worker process.

The orchestrator streams raw webcam frames into `frames` and reads per-frame
body masks back out of `masks`; `state` carries session lifecycle (epoch /
terminate / ready) so the two processes stay in sync without a torch.distributed
world (the env-var-collision pitfall: SAM does its own device setup, so it must
NOT share the Krea process group).

Built on the shared runtime's SharedTensorBuffer ring buffers + a tiny raw-int
state segment. Masks are [H,W] uint8 (255=body, 0=background); "no detection"
is an all-zero mask (equivalent to the reference's None -> keep-Krea path).
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer

# state layout (int64 x4): [epoch, terminate, sam_ready_epoch, sam_done_count]
_EPOCH, _TERM, _READY, _DONE = 0, 1, 2, 3


class SamLink:
    def __init__(self, name: str, height: int, width: int, max_len: int, create: bool):
        self.name = name
        self.create = create
        self.frames = SharedTensorBuffer(
            name=f"{name}_frames", frame_shape=(height, width, 3),
            max_len=max_len, dtype=np.uint8, create=create)
        self.masks = SharedTensorBuffer(
            name=f"{name}_masks", frame_shape=(height, width),
            max_len=max_len, dtype=np.uint8, create=create)
        self._state_name = f"{name}_state"
        if create:
            self._shm = shared_memory.SharedMemory(name=self._state_name, create=True, size=8 * 4)
            self._state = np.ndarray((4,), dtype=np.int64, buffer=self._shm.buf)
            self._state[:] = 0
            self._state[_READY] = -1  # -1 = SAM not loaded yet; 0 = loaded/no session; N = session N ready
        else:
            self._shm = _attach_wait(self._state_name)
            self._state = np.ndarray((4,), dtype=np.int64, buffer=self._shm.buf)

    # ---- state accessors ----
    @property
    def epoch(self) -> int:
        return int(self._state[_EPOCH])

    @property
    def terminate_flag(self) -> int:
        return int(self._state[_TERM])

    @property
    def sam_ready_epoch(self) -> int:
        return int(self._state[_READY])

    @property
    def sam_done(self) -> int:
        return int(self._state[_DONE])

    # ---- orchestrator side ----
    def new_session(self, timeout_s: float = 1800.0) -> None:
        """Clear buffers, bump epoch, wait until SAM acks the new session."""
        self.frames.clear()
        self.masks.clear()
        self._state[_DONE] = 0
        self._state[_EPOCH] = self.epoch + 1
        target = self.epoch
        t0 = time.time()
        while self.sam_ready_epoch < target:
            if time.time() - t0 > timeout_s:
                raise TimeoutError("SAM worker did not ack new session")
            time.sleep(0.002)

    def push_frames(self, frames_np: np.ndarray) -> None:
        """frames_np: [T,H,W,3] uint8 (or single [H,W,3])."""
        self.frames.write(frames_np)

    def read_masks(self, start_idx: int, n: int, timeout_s: float = 1800.0) -> np.ndarray:
        """Block until masks [start_idx:start_idx+n] are available, return them."""
        t0 = time.time()
        while True:
            nxt, m = self.masks.read(start_idx, n)
            if m is not None:
                return m
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"SAM masks [{start_idx}:{start_idx+n}] not ready "
                                   f"(masks.num={self.masks.num})")
            time.sleep(0.001)

    def signal_terminate(self) -> None:
        self._state[_TERM] = 1

    # ---- SAM worker side ----
    def set_ready(self, epoch: int) -> None:
        self._state[_READY] = epoch

    def set_done(self, n: int) -> None:
        self._state[_DONE] = n

    def close(self) -> None:
        for b in (self.frames, self.masks):
            try:
                b.close()
            except Exception:
                pass
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        for b in (self.frames, self.masks):
            try:
                b.unlink()
            except Exception:
                pass
        try:
            self._shm.unlink()
        except Exception:
            pass


def _attach_wait(name: str, timeout_s: float = 1800.0):
    t0 = time.time()
    while True:
        try:
            return shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(0.002)
