"""MC-Dropout uncertainty quantification.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Runs N stochastic forward passes with dropout active, returns the mean
softmax probability and the per-class standard deviation across passes.
The std is the epistemic-uncertainty signal that feeds ``uncertainty_std``
in each Detection and drives the needs-review flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ml.models import enable_mc_dropout

N_PASSES = 10


@dataclass
class MCResult:
    mean_probs: torch.Tensor  # (B, C)
    std_probs: torch.Tensor   # (B, C)
    predicted: torch.Tensor   # (B,) argmax of mean
    confidence: torch.Tensor  # (B,) max mean prob
    uncertainty: torch.Tensor  # (B,) std of the predicted class


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module, x: torch.Tensor, n_passes: int = N_PASSES
) -> MCResult:
    enable_mc_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=-1) for _ in range(n_passes)], dim=0)
    mean_probs = probs.mean(dim=0)
    std_probs = probs.std(dim=0)
    predicted = mean_probs.argmax(dim=-1)
    confidence = mean_probs.max(dim=-1).values
    uncertainty = std_probs.gather(1, predicted.unsqueeze(1)).squeeze(1)
    return MCResult(mean_probs, std_probs, predicted, confidence, uncertainty)
