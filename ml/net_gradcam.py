"""Dependency-free Grad-CAM for TinyCNN (functional path).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.
"""
from __future__ import annotations

import numpy as np
import torch


def gradcam_tiny(model, x: torch.Tensor) -> np.ndarray:
    """Return an HxW CAM in [0,1] for input x (1,3,H,W) using the last conv."""
    model.eval()
    activations, gradients = {}, {}
    layer = model.last_conv

    def fwd_hook(_m, _i, o):
        activations["v"] = o.detach()

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    logits = model(x)
    cls = int(logits.argmax(1).item())
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()

    act = activations["v"][0]                       # (C,h,w)
    grad = gradients["v"][0]                         # (C,h,w)
    weights = grad.mean(dim=(1, 2))                  # (C,)
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam.cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    # upscale to input size
    import cv2
    return cv2.resize(cam, (x.shape[3], x.shape[2]))
