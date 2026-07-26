"""Lightweight from-scratch CNN for the fully-functional offline path.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

timm/EfficientNet needs a pretrained-weight download (network). For a
self-contained, fully-runnable-offline build we use this small CNN so the
whole pipeline trains and infers with zero external downloads. The timm
backbones in ml/models.py remain the path for real-dataset training.

Named `features[-1]` conv is exposed for Grad-CAM; the dropout stays active
at inference for MC-Dropout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MC_DROPOUT_P = 0.3


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int, p: float = MC_DROPOUT_P):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        z = self.pool(f).flatten(1)
        return self.fc(self.dropout(z))

    @property
    def last_conv(self) -> nn.Module:
        return self.features[-3]  # the final Conv2d


def build_tiny(num_classes: int) -> TinyCNN:
    return TinyCNN(num_classes)
