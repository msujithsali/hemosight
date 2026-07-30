"""Tests for YOLOv8 cell detector."""
import numpy as np
import pytest
from pathlib import Path


def test_yolo_detector_import():
    from ml.detect_yolo import detect_cells_yolo
    assert detect_cells_yolo is not None


def test_yolo_falls_back_when_weights_missing():
    from ml.detect_yolo import detect_cells_yolo
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 200
    # Use bogus path -> should fall back to classical CV
    boxes = detect_cells_yolo(img, weights_path=Path("does_not_exist.pt"))
    assert isinstance(boxes, list)


def test_yolo_returns_bbox_format():
    from ml.detect_yolo import detect_cells_yolo
    img = np.ones((200, 200, 3), dtype=np.uint8) * 100
    boxes = detect_cells_yolo(img)
    assert isinstance(boxes, list)
    for box in boxes:
        assert len(box) == 4
        assert all(isinstance(x, int) for x in box)


def test_yolo_with_real_weights():
    """Only runs if the real .pt is present."""
    weights = Path("results/bccd_yolov8s_REAL.pt")
    if not weights.exists():
        pytest.skip("Real YOLO weights not present")
    from ml.detect_yolo import detect_cells_yolo
    img = np.random.randint(100, 200, (416, 416, 3), dtype=np.uint8)
    boxes = detect_cells_yolo(img, weights_path=weights)
    assert isinstance(boxes, list)
