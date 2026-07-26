> ## ⚠️ SCREENING AID ONLY — NOT A DIAGNOSTIC DEVICE
> HemoSight produces **abnormality flags with calibrated confidence and a
> needs-review flag**. It does **not** diagnose disease and never emits a
> disease name. Every result must be confirmed by a qualified pathologist or
> physician. All benchmark numbers are on public datasets (Raabin-WBC, NIH
> Malaria) that are **off-domain** from Indian PHC settings and require
> `[TARGET]`-tagged clinical validation before any real-world use.

# HemoSight

**Open-source federated-learning framework for on-device blood microscopy
screening in Indian PHC settings, evaluated on Raabin-WBC and NIH Malaria
public benchmarks.**

Author: **M Sujith Sali**, ISE Dept, VTU Karnataka · License: Apache-2.0

HemoSight runs on a Raspberry Pi 5 with a USB microscope. It (1) detects blood
cells with YOLOv8, (2) classifies WBCs into 5 clinical classes, (3) flags
malaria parasites, (4) quantifies uncertainty with MC-Dropout, (5) federates
model updates across simulated PHCs with Flower (FedAvg/FedProx) **without
sharing patient images**, (6) offers a Kannada/Hindi/English UI with voice
prompts, and (7) generates cryptographically signed PDF reports.

## Integrity guarantees (enforced in code)
- **No fabricated metrics.** Numbers come only from logged MLflow runs with a
  fixed seed and dataset hash. Un-run metrics read `PENDING` with the exact
  `make` target — never a placeholder value.
- **Provenance tagging.** Every metric is `[BOOTSTRAP]` (public, off-domain) or
  `[TARGET]` (real PHC samples, not yet collected). Never blended silently.
- **No cloud in the inference path.** `common/egress_guard.no_egress()`
  monkey-patches `socket.socket` and raises on any outbound connection during
  `predict()`. Federated aggregation is the only sanctioned network path and it
  transmits weight deltas only.
- **Reproducibility.** `common/seed_everything.py` seeds Python/NumPy/PyTorch
  (CPU+CUDA), sets `cudnn.deterministic`, and pins `PYTHONHASHSEED`.

## Benchmark numbers

All values below are **PENDING** until the corresponding training run is
executed; the code logs real results to MLflow at that point.

| Task | Dataset | Metric | Value | Provenance | Run target |
|------|---------|--------|-------|-----------|-----------|
| Cell detection | BCCD | mAP@50 / mAP@50-95 | PENDING | `[BOOTSTRAP]` | `make train-yolo` |
| WBC 5-class | Raabin-WBC | macro-F1 (95% CI) | PENDING | `[BOOTSTRAP]` | `make train-wbc` |
| WBC calibration | Raabin-WBC | ECE / Brier (pre/post) | PENDING | `[BOOTSTRAP]` | `make train-wbc` |
| Malaria binary | NIH Malaria | ROC-AUC / PR-AUC | PENDING | `[BOOTSTRAP]` | `make train-malaria` |
| Federated | Raabin (non-IID) | global acc/round | PENDING | `[BOOTSTRAP]` | `make federated-simulate` |
| Edge latency | Pi 5 8GB | s/image (target <3s) | PENDING | `[TARGET]` | run `edge/infer.py` on Pi |

## Quickstart
```bash
make bootstrap          # install deps + pre-commit
make download-data      # BCCD/NIH via script; Raabin needs manual registration
make test               # 20 tests, 80% coverage gate on ml/ and api/
make train-wbc          # populates the WBC row above (needs the dataset)
make federated-simulate # 5 non-IID PHC clients, FedAvg + FedProx
docker compose up --build
```

## Hardware bill of materials (indicative INR, 2026 — verify before buying)
- Raspberry Pi 5 8GB — genuine boards from Robocraze / Silverline / ThinkRobotics / The Engineer Store (~₹7,500–9,000).
- Raspberry Pi 27W USB-C PD power supply (~₹1,500).
- Raspberry Pi Active Cooler (~₹600).
- 64/128GB Class-10 U3 microSD — Samsung EVO Plus / SanDisk Ultra (~₹700–1,300).
- Official Pi 5 case (~₹900).
- USB digital microscope 40x–1000x, 8 LEDs, metal stand, **UVC-compatible** — e.g. Microware / iBell IBL-ML1000X / Ruhza / Techie&Trendy on Amazon India (~₹1,200–3,500). **Verify UVC before purchase.**
- Prepared blood-smear teaching slides from a medical supply house (demo only — no patient data, no ethics approval needed for commercial training slides).
- *Optional:* Raspberry Pi AI Kit (Hailo-8L, 13 TOPS); 7-inch touchscreen for kiosk mode.

## Citations
- Kouzehkanan et al. (2022), *Scientific Reports* — Raabin-WBC. https://raabindata.com · Mendeley `snkd93bnjr`.
- Rajaraman et al. (2018), *PeerJ* — NIH Malaria Cell Images. https://lhncbc.nlm.nih.gov
- BCCD dataset — https://github.com/Shenggan/BCCD_Dataset
- Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics
- Flower (flwr) — https://flower.ai
- AI4Bharat IndicTrans2 / IndicTTS — https://ai4bharat.org

See `MODEL_CARD.md` for intended use, limitations, and out-of-scope uses, and
`INTERVIEW_HANDBOOK.md` for the placement-defence walkthrough.
