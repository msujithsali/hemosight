"""Model definitions for HemoSight classifiers.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

- WBC 5-class: EfficientNet-B0 backbone (timm) + MC-Dropout head (p=0.3).
- Malaria binary: MobileNetV3-Small (timm) + MC-Dropout head.

MC-Dropout: the dropout layer in the head is kept ACTIVE at inference (via
``enable_mc_dropout``) so N stochastic forward passes give an epistemic
uncertainty estimate. timm is optional at import time so unit tests that
only need the head can run without downloading pretrained weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

WBC_CLASSES = ["Neutrophil", "Eosinophil", "Basophil", "Lymphocyte", "Monocyte"]
MALARIA_CLASSES = ["Uninfected", "Parasitized"]
MC_DROPOUT_P = 0.3


class MCDropoutHead(nn.Module):
    """Linear classification head whose dropout stays on for MC sampling."""

    def __init__(self, in_features: int, num_classes: int, p: float = MC_DROPOUT_P):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


def _build(backbone_name: str, num_classes: int) -> nn.Module:
    import timm

    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    in_features = backbone.num_features
    return nn.Sequential(backbone, MCDropoutHead(in_features, num_classes))


def build_wbc_model() -> nn.Module:
    return _build("efficientnet_b0", len(WBC_CLASSES))


def build_malaria_model() -> nn.Module:
    return _build("mobilenetv3_small_100", len(MALARIA_CLASSES))


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-activate every Dropout layer."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
