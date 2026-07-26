"""Attention gate — cross-checks model attention against detected cells.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Computes IoU between the Grad-CAM region (activation > 0.7) and the union of
YOLO cell masks. If IoU < 0.3 the sample is flagged ATTENTION_MISALIGNMENT
and routed to the needs-review queue: the classifier is "looking" somewhere
other than the cells, so its prediction is untrustworthy.
"""
from __future__ import annotations

import numpy as np

from api.schemas import AttentionGate, AttentionStatus

CAM_ACTIVATION_THRESHOLD = 0.7
IOU_MISALIGN_THRESHOLD = 0.3


def _binarize_cam(cam: np.ndarray, thresh: float = CAM_ACTIVATION_THRESHOLD) -> np.ndarray:
    cam = cam.astype(np.float32)
    rng = cam.max() - cam.min()
    norm = (cam - cam.min()) / (rng + 1e-8)
    return (norm >= thresh).astype(np.uint8)


def masks_iou(cam: np.ndarray, cell_mask: np.ndarray) -> float:
    """IoU between thresholded Grad-CAM and the binary cell mask."""
    cam_bin = _binarize_cam(cam)
    cell_bin = (cell_mask > 0).astype(np.uint8)
    intersection = np.logical_and(cam_bin, cell_bin).sum()
    union = np.logical_or(cam_bin, cell_bin).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def evaluate_gate(cam: np.ndarray, cell_mask: np.ndarray) -> AttentionGate:
    iou = masks_iou(cam, cell_mask)
    status = (
        AttentionStatus.PASSED
        if iou >= IOU_MISALIGN_THRESHOLD
        else AttentionStatus.ATTENTION_MISALIGNMENT
    )
    return AttentionGate(iou_score=round(iou, 4), status=status)


def cell_mask_from_bboxes(shape: tuple[int, int], bboxes: list[list[int]]) -> np.ndarray:
    """Rasterise YOLO bboxes [x1,y1,x2,y2] into a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    for x1, y1, x2, y2 in bboxes:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        mask[y1:y2, x1:x2] = 1
    return mask
