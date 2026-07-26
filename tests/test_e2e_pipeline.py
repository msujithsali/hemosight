"""End-to-end functional test: real detection + classification + uncertainty +
attention gate produce a fully-populated, valid AnalysisResponse (no stubs).

Trains tiny models on the fly if weights are absent, so the test is
self-contained. Asserts the CONTRACT is populated and in-range — NOT specific
class outcomes (synthetic-data accuracy is not clinically meaningful)."""
from pathlib import Path

import pytest

from api.schemas import AnalysisResponse


def _ensure_weights():
    if not Path("results/wbc_tiny.pt").exists():
        from scripts.train_toy import main as train
        train()


def test_pipeline_end_to_end():
    _ensure_weights()
    from ml.pipeline import load_models, run_pipeline
    from scripts.make_toy_dataset import make_field

    wbc, mal = load_models()
    result = run_pipeline(make_field(n_cells=14, seed=3), wbc, mal)

    assert isinstance(result, AnalysisResponse)
    d = result.model_dump()
    # fully populated, no PENDING at runtime
    assert d["model_version"] != "PENDING"
    assert d["metrics"]["total_counts"] >= 1
    assert len(d["detections"]) == d["metrics"]["total_counts"]
    for det in d["detections"]:
        assert 0.0 <= det["confidence"] <= 1.0
        assert det["uncertainty_std"] >= 0.0
        assert len(det["bbox"]) == 4
        assert det["class_name"] in d["metrics"]["wbc_differential"]
    assert 0.0 <= d["attention_gate"]["iou_score"] <= 1.0
    assert d["attention_gate"]["status"] in {"PASSED", "ATTENTION_MISALIGNMENT"}
    assert "SCREENING AID ONLY" in d["disclaimer"]


def test_quality_gate_rejects_blank():
    import numpy as np
    from ml.pipeline import load_models, run_pipeline
    _ensure_weights()
    wbc, mal = load_models()
    blank = np.full((300, 300, 3), 255, np.uint8)  # too bright
    out = run_pipeline(blank, wbc, mal)
    assert isinstance(out, dict) and "quality_error" in out
