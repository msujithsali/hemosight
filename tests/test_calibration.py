"""Tests for calibration module."""
import torch
import numpy as np
import pytest
from ml.calibrate import (
    expected_calibration_error, brier_score,
    TemperatureScaler, CalibrationMetrics,
)


def test_perfect_calibration_has_zero_ece():
    # 100 samples, 2 classes, perfectly calibrated
    probs = np.array([[0.9, 0.1]] * 90 + [[0.1, 0.9]] * 10)
    labels = np.array([0] * 90 + [1] * 10)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.15, f"ECE should be low for well-calibrated data, got {ece}"


def test_brier_score_perfect():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_brier_score_worst():
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == pytest.approx(2.0, abs=1e-6)


def test_temperature_scaler_reduces_confidence():
    scaler = TemperatureScaler()
    scaler.temperature = torch.nn.Parameter(torch.tensor([2.0]))
    logits = torch.tensor([[5.0, 1.0]])
    scaled = scaler(logits)
    assert scaled[0, 0].item() == pytest.approx(2.5, abs=1e-4)
    assert scaled[0, 1].item() == pytest.approx(0.5, abs=1e-4)


def test_temperature_scaler_fit_on_toy_data():
    # Toy model: just returns fixed logits
    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[3.0, 0.1]] * x.shape[0])

    # Toy dataset
    data = [(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))]
    model = ToyModel()
    scaler = TemperatureScaler.fit(model, data, device="cpu")
    assert scaler.temperature.item() > 0, "Temperature must be positive"


def test_calibration_metrics_repr():
    m = CalibrationMetrics(ece=0.05, brier=0.12, temperature=1.5)
    s = repr(m)
    assert "ECE" in s and "Brier" in s and "T=" in s
