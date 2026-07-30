"""Ablation study: measure impact of calibration, MC-Dropout on toy inputs.

Reports before/after metrics as a table. This is honest reporting scaffolding
for a real dataset run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ml.calibrate import (
    TemperatureScaler, expected_calibration_error, brier_score
)


def run_ablation():
    """Simulate: calibration on vs off, with toy logits."""
    torch.manual_seed(42)
    np.random.seed(42)
    # Simulate 200 samples, 2 classes, poorly-calibrated (overconfident) logits
    n = 200
    labels = np.random.randint(0, 2, n)
    # Overconfident: multiply by 5 to shrink softmax entropy
    logits = torch.randn(n, 2) * 5.0
    # Bias towards correct label for ~85% of samples
    for i in range(n):
        if np.random.rand() < 0.85:
            logits[i, labels[i]] += 3.0

    probs_uncal = F.softmax(logits, dim=-1).numpy()
    ece_uncal = expected_calibration_error(probs_uncal, labels)
    brier_uncal = brier_score(probs_uncal, labels)

    # Fit temperature scaling on same data (in practice: use held-out val)
    class Fixed(torch.nn.Module):
        def __init__(s, l): super().__init__(); s.l = l
        def forward(s, x): return s.l

    dummy_loader = [(torch.randn(n, 3, 4, 4), torch.tensor(labels))]
    scaler = TemperatureScaler.fit(Fixed(logits), dummy_loader, device="cpu")
    probs_cal = F.softmax(scaler(logits), dim=-1).numpy()
    ece_cal = expected_calibration_error(probs_cal, labels)
    brier_cal = brier_score(probs_cal, labels)

    result = {
        "ablation": "temperature_scaling",
        "n_samples": n,
        "before": {"ECE": round(ece_uncal, 4), "Brier": round(brier_uncal, 4), "T": 1.0},
        "after":  {"ECE": round(ece_cal, 4),   "Brier": round(brier_cal, 4),
                   "T": round(float(scaler.temperature.item()), 3)},
        "ece_reduction": round(ece_uncal - ece_cal, 4),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/ablation_calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_ablation()
