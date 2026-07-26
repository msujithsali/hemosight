"""ONNX and PyTorch outputs must match within 1e-4.
Skips if onnxruntime/timm are unavailable in the runner."""
import pytest
import torch
import torch.nn as nn

from common.seed_everything import seed_everything


def _deps() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _deps(), reason="onnxruntime absent")
def test_onnx_pytorch_parity(tmp_path):
    from ml.onnx_export import export_onnx, verify_parity
    seed_everything(1729)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(4, 5))
    path = export_onnx(model, tmp_path / "m.onnx")
    assert verify_parity(model, path, tol=1e-4) <= 1e-4
