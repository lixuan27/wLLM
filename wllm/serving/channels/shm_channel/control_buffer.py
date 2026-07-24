import time
import numpy as np
from multiprocessing import shared_memory


class SharedControlBuffer:
    """
    Shared-memory control/synchronization primitive.

    Layout (two int64, total 16 bytes):
      - value: payload written by sender
      - ack:   incremented by receiver on commit()

    Methods:
      - send(v): write value and wait until ack increases by 1
      - recv():  read value only (no ack change)
      - commit(): ack += 1

    Attach behavior (create=False):
      - Will wait until the creator has created the shared memory block.
    """

    def __init__(
        self,
        name: str,
        create: bool = True,
        wait: bool = True,
        timeout_s: float = 10.0,
        sleep_s: float = 0.001,
    ):
        self.name = name
        self.shm_name = f"{name}__ctrl"
        self.is_creator = bool(create)
        self._shm = None

        if create:
            self._shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=16
            )
            arr = np.ndarray((2,), dtype=np.int64, buffer=self._shm.buf)
            arr[0] = 0  # value
            arr[1] = 0  # ack
        else:
            self._shm = self._attach_with_wait(
                self.shm_name, wait=wait, timeout_s=timeout_s, sleep_s=sleep_s
            )

        self._arr = np.ndarray((2,), dtype=np.int64, buffer=self._shm.buf)

    @staticmethod
    def _attach_with_wait(name: str, wait: bool, timeout_s: float, sleep_s: float):
        """Try to attach to an existing SharedMemory, optionally waiting."""
        if not wait:
            return shared_memory.SharedMemory(name=name, create=False)

        t0 = time.time()
        last_err = None
        while True:
            try:
                return shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError as e:
                last_err = e
                if (time.time() - t0) >= timeout_s:
                    raise TimeoutError(
                        f"Timed out waiting for shared memory '{name}' to be created."
                    ) from last_err
                time.sleep(sleep_s)

    @property
    def value(self) -> int:
        """Current payload value."""
        return int(self._arr[0])

    @property
    def ack(self) -> int:
        """Current ack counter."""
        return int(self._arr[1])

    def send(
        self,
        v: int,
        timeout_s: float | None = None,
        sleep_s: float = 0.0005,
    ) -> bool:
        """
        Send a value and block until receiver acknowledges once via commit().

        Returns:
            True if acknowledged, False if timed out.
        """
        v = int(v)
        ack0 = self.ack

        self._arr[0] = v
        target = ack0 + 1

        t0 = time.time()
        while True:
            if self.ack >= target:
                return True
            if timeout_s is not None and (time.time() - t0) >= timeout_s:
                return False
            if sleep_s > 0:
                time.sleep(sleep_s)

    def recv(self) -> int:
        """Read the current value without acknowledging."""
        return self.value

    def commit(self) -> int:
        """Acknowledge one received value by incrementing ack."""
        new_ack = self.ack + 1
        self._arr[1] = new_ack
        return int(new_ack)

    def close(self):
        """Detach from shared memory (does NOT destroy it)."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def unlink(self):
        """Destroy the shared memory block (creator only)."""
        if self._shm is not None:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass