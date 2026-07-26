"""Calibration: temperature scaling + ECE / Brier score.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Temperature scaling (Guo et al. 2017): fit a single scalar T on the
validation split by minimising NLL, then divide logits by T at inference.
Reported alongside ECE (Expected Calibration Error) and Brier score before
and after calibration, all logged to MLflow by the training scripts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TemperatureScaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        nll = torch.nn.CrossEntropyLoss()

        def _closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return float(self.temperature.detach().item())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = accuracies[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    num_classes = probs.shape[1]
    onehot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
