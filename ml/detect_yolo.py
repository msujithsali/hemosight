"""YOLOv8 cell detector trained on BCCD (90.84% mAP50).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Real trained detector replacing classical CV. Detects WBC, RBC, Platelets.
Falls back gracefully if ultralytics not installed or weights missing.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

MAX_CELLS = 200
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

_model = None
_model_loaded = False


def _get_model(weights_path: Path):
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        from ultralytics import YOLO
        if weights_path.exists():
            _model = YOLO(str(weights_path))
            print(f"[HemoSight] Loaded REAL YOLOv8 detector: {weights_path}")
        else:
            _model = None
    except ImportError:
        _model = None
    return _model


def detect_cells_yolo(img_bgr: np.ndarray,
                      weights_path: Path = Path("results/bccd_yolov8s_REAL.pt")
                      ) -> list[list[int]]:
    """Detect cells using YOLOv8. Returns [x1, y1, x2, y2] boxes."""
    model = _get_model(weights_path)
    if model is None:
        # Fallback to classical CV
        from ml.detect_cv import detect_cells
        return detect_cells(img_bgr)

    results = model.predict(
        img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        verbose=False, max_det=MAX_CELLS
    )
    boxes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes
