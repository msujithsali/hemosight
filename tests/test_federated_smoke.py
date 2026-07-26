"""Federated strategy factory smoke test (no full simulation in CI unit tier)."""
import pytest


def _flwr() -> bool:
    try:
        import flwr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _flwr(), reason="flwr absent")
def test_strategy_factory():
    from federated.strategies import make_strategy
    assert make_strategy("fedavg") is not None
    assert make_strategy("fedprox") is not None
    with pytest.raises(ValueError):
        make_strategy("nope")
