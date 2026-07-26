"""Attention gate flags misalignment; passes when CAM overlaps cells."""
import numpy as np

from api.schemas import AttentionStatus
from ml.attention_gate import cell_mask_from_bboxes, evaluate_gate


def test_aligned_passes():
    cam = np.zeros((100, 100), np.float32)
    cam[20:80, 20:80] = 1.0
    mask = cell_mask_from_bboxes((100, 100), [[20, 20, 80, 80]])
    gate = evaluate_gate(cam, mask)
    assert gate.status == AttentionStatus.PASSED
    assert gate.iou_score > 0.3


def test_misaligned_flags():
    cam = np.zeros((100, 100), np.float32)
    cam[0:20, 0:20] = 1.0            # CAM in the corner
    mask = cell_mask_from_bboxes((100, 100), [[60, 60, 95, 95]])  # cells elsewhere
    gate = evaluate_gate(cam, mask)
    assert gate.status == AttentionStatus.ATTENTION_MISALIGNMENT
    assert gate.iou_score < 0.3
