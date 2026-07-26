"""Deterministic blood-smear preprocessing pipeline.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Order (fixed): flat-field background subtraction -> non-local-means denoise
-> CLAHE (clip=2.0) -> gray-world white balance. All parameters are fixed
constants so the pipeline is reproducible for a given input.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a runtime dep
    cv2 = None  # type: ignore

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
NLM_H = 10.0
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv-python is required for preprocessing")


def flat_field_subtract(img: np.ndarray, sigma: int = 51) -> np.ndarray:
    """Estimate illumination via a large Gaussian blur and normalise it out.

    Removes vignetting / uneven microscope illumination. Works per channel.
    """
    _require_cv2()
    img_f = img.astype(np.float32) + 1e-6
    background = cv2.GaussianBlur(img_f, (sigma | 1, sigma | 1), 0)
    corrected = img_f / (background + 1e-6)
    corrected = corrected / corrected.max() * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def nlm_denoise(img: np.ndarray) -> np.ndarray:
    _require_cv2()
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            img, None, NLM_H, NLM_H, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW
        )
    return cv2.fastNlMeansDenoising(
        img, None, NLM_H, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW
    )


def clahe_equalize(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space (colour-safe contrast)."""
    _require_cv2()
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Gray-world assumption: force per-channel means to the global mean."""
    if img.ndim != 3:
        return img
    result = img.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)
    global_mean = means.mean()
    scale = global_mean / (means + 1e-6)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Full fixed-order pipeline. Input/output are BGR uint8 arrays."""
    x = flat_field_subtract(img)
    x = nlm_denoise(x)
    x = clahe_equalize(x)
    x = gray_world_white_balance(x)
    return x
