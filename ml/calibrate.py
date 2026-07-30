"""Post-hoc temperature-scaling calibration with ECE and Brier score.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

A confidently-wrong prediction is the dangerous failure mode in medical
screening. Temperature scaling (Guo et al., 2017) learns a single scalar T
that rescales logits so that softmax outputs approximate true class
probabilities. ECE and Brier score measure calibration before/after.

Usage:
    calibrator = TemperatureScaler.fit(model, val_loader, device)
    calibrated_probs = calibrator(logits)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    ece: float          # Expected Calibration Error (lower = better)
    brier: float        # Brier score (lower = better)
    temperature: float  # learned T (1.0 = uncalibrated)

    def __repr__(self) -> str:
        return (f"CalibrationMetrics(ECE={self.ece:.4f}, "
                f"Brier={self.brier:.4f}, T={self.temperature:.3f})")


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute ECE with equal-width bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score."""
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


class TemperatureScaler(nn.Module):
    """Learns a single temperature T on validation logits."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    @classmethod
    @torch.no_grad()
    def _collect_logits(cls, model, loader, device):
        model.eval()
        all_logits, all_labels = [], []
        for x, y in loader:
            logits = model(x.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(y)
        return torch.cat(all_logits), torch.cat(all_labels)

    @classmethod
    def fit(cls, model, val_loader, device="cpu", lr=0.01, max_iter=200):
        """Fit temperature on validation set, return fitted scaler."""
        logits, labels = cls._collect_logits(model, val_loader, device)
        scaler = cls()
        optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(scaler(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return scaler

    def calibration_report(
        self, model, val_loader, device="cpu"
    ) -> tuple[CalibrationMetrics, CalibrationMetrics]:
        """Return (before, after) calibration metrics."""
        logits, labels = self._collect_logits(model, val_loader, device)
        labels_np = labels.numpy()

        # Before calibration
        probs_before = F.softmax(logits, dim=-1).numpy()
        before = CalibrationMetrics(
            ece=expected_calibration_error(probs_before, labels_np),
            brier=brier_score(probs_before, labels_np),
            temperature=1.0,
        )

        # After calibration
        probs_after = F.softmax(self(logits), dim=-1).numpy()
        after = CalibrationMetrics(
            ece=expected_calibration_error(probs_after, labels_np),
            brier=brier_score(probs_after, labels_np),
            temperature=float(self.temperature.item()),
        )
        return before, after
