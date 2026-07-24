"""Runtime helpers shared by the LiveAvatar backend variants."""

from __future__ import annotations

import socket


def free_port() -> int:
    """Pick an unused localhost port for a variant's torch.distributed rendezvous.

    Binding to port 0 and reading back what the kernel assigned beats deriving
    the port from the pid: a derived port can already be held by an unrelated
    service, and the collision surfaces as a garbled TCPStore handshake rather
    than a clean address-in-use error.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
