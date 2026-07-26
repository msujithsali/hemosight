"""Export trained PyTorch classifiers to ONNX with a dynamic batch axis.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Edge inference on the Pi 5 uses ONNX Runtime + XNNPACK, not PyTorch. This
script exports and then verifies that ONNX and PyTorch outputs agree within
1e-4 (see tests/test_onnx_parity.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, out_path: str | Path, image_size: int = 224) -> Path:
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return out_path


def verify_parity(model: torch.nn.Module, onnx_path: str | Path, tol: float = 1e-4) -> float:
    import onnxruntime as ort

    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > tol:
        raise AssertionError(f"ONNX/PyTorch mismatch {max_diff} > {tol}")
    return max_diff
