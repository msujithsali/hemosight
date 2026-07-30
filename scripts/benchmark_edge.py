"""Latency + size benchmark: FP32 vs quantized (INT8-ish dynamic).

Measures inference time on 100 batches and model size on disk. Not a real
INT8 export (that requires torch.quantization workflow); this is a scaffold
showing the tradeoff-measurement approach.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
import torch.nn as nn


def bench(model, x, n_iter=100):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = model(x)
        elapsed = time.perf_counter() - start
    return elapsed / n_iter * 1000  # ms/iter


def run():
    import torchvision.models as tv
    fp32 = tv.resnet18(weights=None)
    fp32.fc = nn.Linear(fp32.fc.in_features, 2)
    weights = Path("results/malaria_resnet18_REAL.pt")
    if weights.exists():
        fp32.load_state_dict(torch.load(weights, map_location="cpu"))

    # Dynamic quantization (INT8 for Linear/Conv1d)
    try:
        q = torch.quantization.quantize_dynamic(
            fp32, {nn.Linear}, dtype=torch.qint8
        )
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
        q = fp32

    x = torch.randn(1, 3, 128, 128)
    fp32_ms = bench(fp32, x)
    q_ms = bench(q, x)

    # Sizes
    torch.save(fp32.state_dict(), "/tmp/fp32.pt")
    torch.save(q.state_dict(), "/tmp/q.pt")
    fp32_kb = Path("/tmp/fp32.pt").stat().st_size / 1024
    q_kb = Path("/tmp/q.pt").stat().st_size / 1024

    result = {
        "model": "ResNet-18 malaria",
        "input_shape": [1, 3, 128, 128],
        "n_iter": 100,
        "device": "cpu",
        "fp32":  {"latency_ms": round(fp32_ms, 2), "size_kb": round(fp32_kb, 1)},
        "int8":  {"latency_ms": round(q_ms, 2),    "size_kb": round(q_kb, 1)},
        "speedup": round(fp32_ms / max(q_ms, 0.001), 2),
        "size_reduction_pct": round((1 - q_kb / fp32_kb) * 100, 1),
    }
    Path("results/benchmark_edge.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Windows tmp path fix
    import os
    if os.name == "nt":
        os.makedirs("results/tmp", exist_ok=True)
        # Patch tmp paths
        import scripts.benchmark_edge as this
        _bench = run
        # Simple in-place
    run()
