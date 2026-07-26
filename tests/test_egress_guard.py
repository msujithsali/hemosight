"""Inference path must refuse outbound network calls."""
import socket

import pytest

from common.egress_guard import EgressViolation, no_egress


def test_blocks_outbound():
    with no_egress():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(EgressViolation):
            s.connect(("8.8.8.8", 53))


def test_allows_loopback():
    with no_egress():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.01)
        try:
            s.connect_ex(("127.0.0.1", 9))  # should not raise EgressViolation
        except EgressViolation:
            raise AssertionError("loopback must be allowed")
        finally:
            s.close()


def test_restores_socket_after_block():
    original = socket.socket
    with no_egress():
        pass
    assert socket.socket is original
