"""Functional classical-CV cell detector (runs now, no training needed).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

A real, deterministic contour-based detector so the end-to-end pipeline is
fully functional offline. When a trained YOLOv8 ONNX is available, edge/infer
switches to it; this remains the always-works fallback. Returns bboxes as
[x1,y1,x2,y2].
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_CELL_AREA = 150
MAX_CELLS = 200


def detect_cells(img_bgr: np.ndarray) -> list[list[int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CELLS]:
        if cv2.contourArea(c) < MIN_CELL_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes
