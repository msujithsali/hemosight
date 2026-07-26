# HemoSight — Module Status

Author: M Sujith Sali, ISE Dept, VTU Karnataka.
Legend: [WORKING] tests pass in CI · [PROTOTYPE] code complete, needs dataset/HW run · [PLANNED]

| Module | Description | Status | Notes |
|--------|-------------|--------|-------|
| 0 | Scaffold + determinism | **[WORKING]** | `test_determinism` passes; seed/egress/disclaimer utils done |
| 1 | Data contracts + preprocess + quality gate | **[WORKING]** | schema/quality/preprocess tests pass; download script + manifests + toy set |
| 2 | YOLOv8 detection (BCCD) | [PROTOTYPE] | trainer + ONNX export done; ONNX↔PyTorch parity test passes; mAP PENDING (`make train-yolo`) |
| 3 | WBC 5-class (Raabin) | [PROTOTYPE] | trainer + calibration + bootstrap CIs done; F1/ECE PENDING (`make train-wbc`) |
| 4 | Malaria binary (NIH) | [PROTOTYPE] | trainer + ROC/PR + calibration done; PENDING (`make train-malaria`) |
| 5 | Uncertainty + attention gate | **[WORKING]** | MC-Dropout + attention-gate tests pass; Grad-CAM wired |
| 6 | Federated (Flower) | [PROTOTYPE] | FedAvg/FedProx + non-IID sim done; strategy smoke test passes; curves PENDING (`make federated-simulate`) |
| 7 | Edge deploy (Pi 5) | [PROTOTYPE] | ONNX/XNNPACK infer + camera adapter + systemd done; latency PENDING (run on Pi) |
| 8 | FastAPI + React UI | [PROTOTYPE] | SSE stream + schemas + kn/hi/en UI + gauge + canvas done |
| 9 | Signed PDF + audit | **[WORKING]** | signed-PDF verify + audit-chain tests pass |
| 10 | Tests + docs bundle | **[WORKING]** | 20 tests pass; README/MODEL_CARD/INTERVIEW_HANDBOOK/CONTRIBUTING done |

## Functional software status

The software runs **fully end-to-end today** on real (synthetic-data-trained)
weights — no PENDING in the runtime path:
`python -m scripts.train_toy` -> `pytest` (22 pass) -> `uvicorn api.main:app`.
POST an image to `/analyze` and get populated cell counts, WBC differential,
malaria flag, MC-Dropout uncertainty, attention-gate status, and a signed PDF.

Synthetic weights are for wiring/demo only; real accuracy needs Raabin-WBC /
NIH Malaria (`make train-*`). Clinical metrics remain `[BOOTSTRAP]`/PENDING
until those runs execute — by design, never fabricated.

Test summary: **22 passed** (determinism, schema, egress, quality, preprocess,
attention, MC-dropout, audit, ONNX parity, signed PDF, federated smoke,
**end-to-end pipeline**, **quality-gate rejection**).
