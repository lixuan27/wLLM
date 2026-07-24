import time
import numpy as np
from multiprocessing import shared_memory


class SharedTensorBuffer:
    """
    Shared ring buffer using multiprocessing.shared_memory.

    Meta stored in shared memory:
      - num (int64): total number of frames written so far (monotonic)
        * Empty buffer: num == 0
        * Latest global index: latest = num - 1

    Tensor storage:
      - tensor: shape = (max_len, *frame_shape)
        Physical position for global index g is: g % max_len

    Attach behavior (create=False):
      - Will wait until the creator has created the shared memory blocks,
        to avoid FileNotFoundError on early attach.
    """

    def __init__(
        self,
        name: str,
        frame_shape,
        max_len: int,
        dtype=np.float32,
        create: bool = True,
        wait: bool = True,
        timeout_s: float = 10.0,
        sleep_s: float = 0.001,
    ):
        self.name = name
        self.frame_shape = tuple(frame_shape)
        self.max_len = int(max_len)
        self.dtype = np.dtype(dtype)
        self.is_creator = bool(create)

        self.meta_name = f"{name}__meta"
        self.tensor_name = f"{name}__tensor"

        self._meta_shm = None
        self._tensor_shm = None

        # ---- Meta (num: int64) ----
        if create:
            self._meta_shm = shared_memory.SharedMemory(
                name=self.meta_name, create=True, size=8
            )
            np.ndarray((1,), dtype=np.int64, buffer=self._meta_shm.buf)[0] = 0
        else:
            self._meta_shm = self._attach_with_wait(
                self.meta_name, wait=wait, timeout_s=timeout_s, sleep_s=sleep_s
            )

        self.num_arr = np.ndarray((1,), dtype=np.int64, buffer=self._meta_shm.buf)

        # ---- Tensor ----
        n_elem = self.max_len * int(np.prod(self.frame_shape, dtype=np.int64))
        n_bytes = n_elem * self.dtype.itemsize

        if create:
            self._tensor_shm = shared_memory.SharedMemory(
                name=self.tensor_name, create=True, size=n_bytes
            )
        else:
            self._tensor_shm = self._attach_with_wait(
                self.tensor_name, wait=wait, timeout_s=timeout_s, sleep_s=sleep_s
            )

        self.tensor = np.ndarray(
            (self.max_len, *self.frame_shape),
            dtype=self.dtype,
            buffer=self._tensor_shm.buf,
        )

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
                # The creator hasn't shm_open'd the segment yet.
                last_err = e
            except ValueError as e:
                # The segment exists but the creator hasn't sized it yet
                # (shm_open + ftruncate are not atomic), so it mmaps empty.
                if "empty file" not in str(e):
                    raise
                last_err = e
            if (time.time() - t0) >= timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for shared memory '{name}' to be created."
                ) from last_err
            time.sleep(sleep_s)

    @property
    def num(self) -> int:
        """Total number of frames written so far."""
        return int(self.num_arr[0])

    def clear(self):
        """Reset the buffer to empty state (num = 0)."""
        self.num_arr[0] = 0

    def write(self, x: np.ndarray) -> int:
        """
        Write one or multiple frames.

        Acceptable shapes:
          - frame_shape -> writes one frame
          - (k, *frame_shape) -> writes k frames

        Returns:
            Updated num (total frames written).
        """
        x = np.asarray(x, dtype=self.dtype)

        if x.shape == self.frame_shape:
            x = x.reshape((1, *self.frame_shape))
        else:
            if (
                len(x.shape) != 1 + len(self.frame_shape)
                or tuple(x.shape[1:]) != self.frame_shape
            ):
                raise ValueError(
                    f"Shape mismatch. Expected {self.frame_shape} "
                    f"or (k, *{self.frame_shape}), got {x.shape}"
                )

        k = int(x.shape[0])
        if k <= 0:
            return self.num

        old_num = self.num
        start_g = old_num

        # Write data first
        for i in range(k):
            g = start_g + i
            pos = g % self.max_len
            self.tensor[pos, ...] = x[i]

        # Publish num last
        self.num_arr[0] = old_num + k
        return int(self.num_arr[0])

    def read(self, x: int, n: int):
        """
        Read n frames starting from global index x.

        Here x means "num-consumed":
          - you have already consumed x frames in total
          - next frame to read has global index = x

        Returns:
            (next_x, frames_or_None)

          - If insufficient frames:
                frames_or_None = None
                next_x = corrected x (may jump forward if overwritten)

          - If sufficient frames:
                frames_or_None = ndarray (n, *frame_shape), copied
                next_x = x + n (or oldest + n if overwritten and corrected)
        """
        if n <= 0:
            raise ValueError("n must be positive")

        x = int(x)
        n = int(n)

        num = self.num
        if num <= 0:
            return 0, None

        latest = num - 1
        oldest = max(0, num - self.max_len)

        if x < oldest:
            x = oldest

        available = latest - x + 1
        if available < n:
            return x, None

        start = x
        end = x + n - 1

        start_pos = start % self.max_len
        end_pos_exclusive = (end % self.max_len) + 1

        if start_pos < end_pos_exclusive:
            frames = self.tensor[start_pos:end_pos_exclusive]
        else:
            frames = np.concatenate(
                [self.tensor[start_pos:], self.tensor[:end_pos_exclusive]],
                axis=0,
            )

        return x + n, frames.copy()

    def get(self):
        """
        Return all currently valid frames in chronological order (oldest -> newest).
        Always returns a copied array.
        """
        num = self.num
        if num <= 0:
            return np.empty((0, *self.frame_shape), dtype=self.dtype)

        n_valid = min(num, self.max_len)
        oldest = num - n_valid
        newest = num - 1

        start_pos = oldest % self.max_len
        end_pos_exclusive = (newest % self.max_len) + 1

        if n_valid < self.max_len or start_pos < end_pos_exclusive:
            frames = self.tensor[start_pos:end_pos_exclusive]
        else:
            frames = np.concatenate(
                [self.tensor[start_pos:], self.tensor[:end_pos_exclusive]],
                axis=0,
            )

        return frames.copy()

    def close(self):
        """Detach from shared memory (does NOT destroy it)."""
        if self._meta_shm is not None:
            self._meta_shm.close()
            self._meta_shm = None
        if self._tensor_shm is not None:
            self._tensor_shm.close()
            self._tensor_shm = None

    def unlink(self):
        """Destroy shared memory blocks (creator only)."""
        if self._meta_shm is not None:
            try:
                self._meta_shm.unlink()
            except FileNotFoundError:
                pass
        if self._tensor_shm is not None:
            try:
                self._tensor_shm.unlink()
            except FileNotFoundError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass