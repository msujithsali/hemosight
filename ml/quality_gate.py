"""Image quality gate — rejects unusable smears with explicit coded errors.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Thresholds (fixed): Laplacian variance < 100 => blur; mean brightness
outside [40, 220]; resolution < 224x224. Returns a QualityError model
instead of raising a bare exception, so the API can surface a coded reason.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from api.schemas import QualityError

BLUR_VAR_MIN = 100.0
BRIGHTNESS_MIN = 40.0
BRIGHTNESS_MAX = 220.0
RES_MIN = 224


def laplacian_variance(img: np.ndarray) -> float:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv required")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_quality(img: Optional[np.ndarray]) -> Optional[QualityError]:
    """Return a QualityError if the image is unusable, else None (passes)."""
    if img is None or img.size == 0:
        return QualityError(code="DECODE_ERROR", message="Image could not be decoded.")

    h, w = img.shape[:2]
    if h < RES_MIN or w < RES_MIN:
        return QualityError(
            code="RESOLUTION_REJECT",
            message=f"Resolution {w}x{h} below minimum {RES_MIN}x{RES_MIN}.",
            measured_value=float(min(h, w)),
            threshold=float(RES_MIN),
        )

    var = laplacian_variance(img)
    if var < BLUR_VAR_MIN:
        return QualityError(
            code="BLUR_REJECT",
            message="Image too blurry (Laplacian variance below threshold).",
            measured_value=var,
            threshold=BLUR_VAR_MIN,
        )

    brightness = float(img.mean())
    if brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX:
        return QualityError(
            code="BRIGHTNESS_REJECT",
            message=f"Mean brightness {brightness:.1f} outside "
            f"[{BRIGHTNESS_MIN}, {BRIGHTNESS_MAX}].",
            measured_value=brightness,
            threshold=None,
        )

    return None
