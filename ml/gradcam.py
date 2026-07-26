"""Grad-CAM heatmap generation for the WBC/malaria classifiers.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Thin wrapper over pytorch-grad-cam. We target the last conv block of the
timm backbone. The returned CAM feeds the attention gate (ml/attention_gate).
"""
from __future__ import annotations

import numpy as np
import torch


def generate_gradcam(model: torch.nn.Module, x: torch.Tensor, target_layer) -> np.ndarray:
    """Return a HxW float CAM in [0,1] for a single image tensor x (1,C,H,W)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        logits = model(x)
        cls = int(logits.argmax(dim=-1).item())
        grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])
    return grayscale[0]


def last_conv_layer(model: torch.nn.Module):
    """Best-effort: return the last Conv2d module in the backbone."""
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found for Grad-CAM target")
    return last
