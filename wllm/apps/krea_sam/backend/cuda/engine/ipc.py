"""Lightweight duplex IPC between the coordinator and its model services.

Uses ``multiprocessing.connection`` over AF_UNIX sockets so the services
can be launched as fully independent subprocesses (each with its own
CUDA_VISIBLE_DEVICES and a scrubbed torch.distributed env), rather than
``mp.spawn`` children that would inherit the parent's CUDA/dist state.

Messages are plain dicts; numpy frame batches ride along as pickled
arrays. A 12x480x832x3 uint8 chunk (~14 MB) pickles over the socket in
~1 ms, negligible against the ~100 ms-scale per-chunk model compute.
"""

from __future__ import annotations

from multiprocessing.connection import Client, Listener
from typing import Any

_AUTHKEY = b"krea_sam_backend"


class CoordinatorLink:
    """Coordinator side: listen for one service to connect, then talk."""

    def __init__(self, address: str):
        self.address = address
        self._listener = Listener(address, family="AF_UNIX", authkey=_AUTHKEY)
        self._conn = None

    def accept(self, timeout_s: float | None = None) -> None:
        # a timeout lets the caller poll for dead service processes
        sock = self._listener._listener._socket
        if timeout_s is not None:
            sock.settimeout(timeout_s)
        try:
            self._conn = self._listener.accept()
        finally:
            if timeout_s is not None:
                sock.settimeout(None)

    def send(self, obj: Any) -> None:
        self._conn.send(obj)

    def recv(self) -> Any:
        return self._conn.recv()

    def poll(self, timeout: float = 0.0) -> bool:
        return self._conn.poll(timeout)

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        try:
            self._listener.close()
        except Exception:
            pass


def connect_to_coordinator(address: str):
    """Service side: connect back to the coordinator's listener."""
    return Client(address, family="AF_UNIX", authkey=_AUTHKEY)
