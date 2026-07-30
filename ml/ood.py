"""Out-of-distribution detection via energy scoring.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

A screening tool that confidently classifies a photo of a wall is dangerous.
Energy-based OOD detection (Liu et al., 2020) uses the LogSumExp of logits
as a score: in-distribution inputs produce lower energy (higher confidence),
OOD inputs produce higher energy. We threshold this to reject invalid inputs
before they reach the classification head.

    E(x) = -T * log(sum(exp(f_i(x) / T)))

Lower energy = more likely in-distribution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score for a batch of logits. Lower = more in-distribution."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def is_ood(
    logits: torch.Tensor,
    threshold: float = -5.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return boolean mask: True if sample is out-of-distribution."""
    scores = energy_score(logits, temperature)
    return scores > threshold  # high energy = OOD


class OODGate:
    """Wraps a classifier to reject OOD inputs before classification."""

    def __init__(self, model: torch.nn.Module, threshold: float = -5.0,
                 temperature: float = 1.0):
        self.model = model
        self.threshold = threshold
        self.temperature = temperature

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        """Returns dict with keys: is_ood, energy, logits, probs."""
        self.model.eval()
        logits = self.model(x)
        scores = energy_score(logits, self.temperature)
        ood_mask = scores > self.threshold
        probs = F.softmax(logits, dim=-1)
        return {
            "is_ood": ood_mask,
            "energy": scores,
            "logits": logits,
            "probs": probs,
        }

    def classify_or_reject(self, x: torch.Tensor) -> list[dict]:
        """Per-sample: returns class+confidence or rejection notice."""
        result = self(x)
        outputs = []
        for i in range(x.shape[0]):
            if result["is_ood"][i]:
                outputs.append({
                    "status": "rejected",
                    "reason": "out-of-distribution",
                    "energy": float(result["energy"][i]),
                })
            else:
                pred = int(result["probs"][i].argmax())
                conf = float(result["probs"][i].max())
                outputs.append({
                    "status": "accepted",
                    "predicted_class": pred,
                    "confidence": round(conf, 4),
                    "energy": float(result["energy"][i]),
                })
        return outputs
