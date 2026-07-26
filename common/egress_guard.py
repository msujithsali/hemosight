"""Runtime network-egress guard for the inference path.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Hard boundary #4: no cloud calls in the inference path. Federated
aggregation is the ONLY sanctioned network path and it lives outside this
guard. We enforce the boundary at runtime by monkey-patching
``socket.socket`` so that any attempt to open an outbound connection while
inside ``no_egress()`` raises :class:`EgressViolation`.

Loopback (127.0.0.0/8, ::1) is allowed so a local MLflow/SQLite sidecar on
the Pi is not blocked; anything else is refused.
"""
from __future__ import annotations

import socket
from contextlib import contextmanager
from typing import Iterator


class EgressViolation(RuntimeError):
    """Raised when code inside ``no_egress()`` tries to reach the network."""


_LOOPBACK_PREFIXES = ("127.", "::1", "localhost")


def _is_loopback(host: object) -> bool:
    if not isinstance(host, str):
        return False
    return host == "localhost" or host.startswith("127.") or host == "::1"


class _GuardedSocket(socket.socket):
    def connect(self, address):  # type: ignore[override]
        host = address[0] if isinstance(address, (tuple, list)) else address
        if not _is_loopback(host):
            raise EgressViolation(
                f"Outbound network call to {host!r} blocked inside inference "
                "path. Only weight-delta federation (outside no_egress) may "
                "touch the network."
            )
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[override]
        host = address[0] if isinstance(address, (tuple, list)) else address
        if not _is_loopback(host):
            raise EgressViolation(
                f"Outbound network call to {host!r} blocked inside inference path."
            )
        return super().connect_ex(address)


@contextmanager
def no_egress() -> Iterator[None]:
    """Block all non-loopback outbound sockets for the duration of the block."""
    original = socket.socket
    socket.socket = _GuardedSocket  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[assignment]
