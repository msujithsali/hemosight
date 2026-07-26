"""Fully-functional end-to-end analysis pipeline.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

image -> quality gate -> preprocess -> detect cells -> per-cell WBC classify
(MC-Dropout) -> malaria flag -> Grad-CAM + attention gate -> AnalysisResponse.
Runs entirely offline on CPU with the TinyCNN weights produced by
`scripts/train_toy.py`. Every field is really computed — no PENDING at runtime.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from api.schemas import (AnalysisResponse, AttentionGate, AttentionStatus,
                         Detection, Metrics, Provenance)
from common.disclaimer import DISCLAIMER
from ml.attention_gate import cell_mask_from_bboxes, evaluate_gate
from ml.detect_cv import detect_cells
from ml.models import WBC_CLASSES
from ml.net import build_tiny
from ml.net_gradcam import gradcam_tiny
from ml.preprocess import preprocess
from ml.quality_gate import check_quality

MALARIA_CLASSES = ["Uninfected", "Parasitized"]
MC_PASSES = 10
NEEDS_REVIEW_UNCERTAINTY = 0.15
PARASITE_CONF_THRESHOLD = 0.90


def load_models(weights_dir: Path = Path("results")):
    wbc = build_tiny(len(WBC_CLASSES))
    mal = build_tiny(len(MALARIA_CLASSES))
    wp, mp = weights_dir / "wbc_tiny.pt", weights_dir / "malaria_tiny.pt"
    if wp.exists():
        wbc.load_state_dict(torch.load(wp, map_location="cpu"))
    if mp.exists():
        mal.load_state_dict(torch.load(mp, map_location="cpu"))
    wbc.eval(); mal.eval()
    return wbc, mal


def _crop_tensor(img: np.ndarray, box: list[int], size: int = 64) -> torch.Tensor:
    x1, y1, x2, y2 = box
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        crop = img
    crop = cv2.resize(crop, (size, size))
    t = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def _mc_predict(model, x, n=MC_PASSES):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    with torch.no_grad():
        probs = torch.stack([F.softmax(model(x), -1) for _ in range(n)], 0)
    mean = probs.mean(0)[0]
    std = probs.std(0)[0]
    cls = int(mean.argmax())
    return cls, float(mean[cls]), float(std[cls])


def run_pipeline(img_bgr: np.ndarray, wbc_model, malaria_model,
                 model_version: str = "tiny-synthetic-v1",
                 mlflow_run_id: str = "local-functional") -> AnalysisResponse | dict:
    err = check_quality(img_bgr)
    if err is not None:
        return {"quality_error": err.model_dump()}

    clean = preprocess(img_bgr)
    boxes = detect_cells(clean)

    detections: list[Detection] = []
    differential = {c: 0 for c in WBC_CLASSES}
    parasitized_cells = 0
    for i, box in enumerate(boxes):
        x = _crop_tensor(clean, box)
        cls, conf, unc = _mc_predict(wbc_model, x)
        name = WBC_CLASSES[cls]
        differential[name] += 1
        detections.append(Detection(id=i, class_name=name, bbox=box,
                                     confidence=round(conf, 4),
                                     uncertainty_std=round(unc, 4)))
        # Malaria assessed per-cell (parasites are small; whole-field resize hides them).
        # High-confidence gate suppresses out-of-distribution false positives.
        mcls, mconf, _ = _mc_predict(malaria_model, x)
        if mcls == 1 and mconf > PARASITE_CONF_THRESHOLD:
            parasitized_cells += 1

    parasite_flag = parasitized_cells > 0
    parasitemia = (round(100.0 * parasitized_cells / max(len(boxes), 1), 2)
                   if parasite_flag else None)

    # Attention gate: Grad-CAM (whole field) vs detected-cell mask
    whole = _crop_tensor(clean, [0, 0, clean.shape[1], clean.shape[0]])
    cam = gradcam_tiny(wbc_model, whole)
    mask = cell_mask_from_bboxes(clean.shape[:2], boxes)
    cam_full = cv2.resize(cam, (clean.shape[1], clean.shape[0]))
    gate: AttentionGate = evaluate_gate(cam_full, mask)

    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        provenance=Provenance.BOOTSTRAP,
        model_version=model_version,
        mlflow_run_id=mlflow_run_id,
        metrics=Metrics(
            total_counts=len(detections),
            wbc_differential=differential,
            parasite_flag=parasite_flag,
            parasitemia_estimate_pct=parasitemia,
        ),
        detections=detections,
        attention_gate=gate,
        disclaimer=DISCLAIMER,
    )
