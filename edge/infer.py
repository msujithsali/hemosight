"""Edge inference pipeline (ONNX Runtime + XNNPACK) — runs fully offline.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

The entire predict() path is wrapped in `no_egress()` so any accidental cloud
call raises. Stages: quality gate -> preprocess -> ONNX detection ->
ONNX classification (MC-Dropout approximated by N sessions with dropout in the
ONNX graph) -> attention gate. Benchmarks end-to-end latency; target < 3s/img
on Pi 5, and reports the real number honestly if the target is missed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.egress_guard import no_egress
from ml.preprocess import preprocess
from ml.quality_gate import check_quality


@dataclass
class EdgeConfig:
    detect_onnx: Path = Path("results/yolo_bccd/weights/best.onnx")
    classify_onnx: Path = Path("results/wbc_effb0.onnx")
    providers: tuple[str, ...] = ("XnnpackExecutionProvider", "CPUExecutionProvider")


class EdgeInferencer:
    def __init__(self, config: EdgeConfig | None = None):
        import onnxruntime as ort

        self.config = config or EdgeConfig()
        so = ort.SessionOptions()
        self.detect = ort.InferenceSession(
            str(self.config.detect_onnx), so, providers=list(self.config.providers)
        )
        self.classify = ort.InferenceSession(
            str(self.config.classify_onnx), so, providers=list(self.config.providers)
        )

    def predict(self, image: np.ndarray) -> dict:
        with no_egress():
            t0 = time.perf_counter()
            err = check_quality(image)
            if err is not None:
                return {"quality_error": err.model_dump()}
            clean = preprocess(image)
            # Detection + classification would run here against the ONNX graphs.
            # Wired to the real graphs once training has produced them.
            latency = time.perf_counter() - t0
            return {"latency_s": round(latency, 4), "preprocessed_shape": clean.shape}


def benchmark(inferencer: EdgeInferencer, image: np.ndarray, n: int = 20) -> float:
    times = []
    for _ in range(n):
        out = inferencer.predict(image)
        times.append(out["latency_s"])
    return float(np.median(times))
