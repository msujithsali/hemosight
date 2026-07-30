# HemoSight — Federated Edge-AI Blood Smear Screening

> **Screening aid, not a diagnostic device.** All predictions require clinical review.

A federated blood smear screening system designed for rural Indian primary health centres. Runs offline on Raspberry Pi 5 with a UVC microscope. Multiple centres train a shared model without patient images ever leaving the premises.

## Real Model Metrics (NIH Malaria Cell Images Dataset)

Trained on **27,558 real microscope images** from the NIH Malaria Cell Images Dataset (Parasitized / Uninfected). ResNet-18 fine-tuned for 5 epochs on Kaggle GPU.

| Metric | Value |
|---|---|
| **Accuracy** | 96.83% |
| **Precision** | 97.05% |
| **Recall** | 96.74% |
| **F1 Score** | 96.89% |
| **ROC-AUC** | 99.40% |

**Confusion Matrix (validation set, n = 5,512):**

|  | Predicted Parasitized | Predicted Uninfected |
|---|---|---|
| **Actual Parasitized** | 2,610 | 83 |
| **Actual Uninfected** | 92 | 2,727 |

**Training provenance:** Kaggle notebook, GPU T4, seed 42, 80/20 train-val split. Model checkpoint: `malaria_resnet18_REAL.pt` (44.7 MB). No synthetic data, no fabricated metrics.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Flower FL Server                    │
│         FedAvg / FedProx Aggregation                │
│         Non-IID Dirichlet Sharding (α=0.5)          │
└────────────┬──────────────┬──────────────┬──────────┘
             │              │              │
     ┌───────▼──┐    ┌──────▼───┐   ┌─────▼────┐
     │ Client 1 │    │ Client 2 │   │ Client N │
     │ PHC Site │    │ PHC Site │   │ PHC Site │
     │ RPi 5    │    │ RPi 5    │   │ RPi 5    │
     └───────┬──┘    └──────┬───┘   └─────┬────┘
             │              │              │
     ┌───────▼──────────────▼──────────────▼──────┐
     │          Edge Inference Pipeline            │
     │  UVC Camera → Preprocess → ONNX Runtime    │
     │  → MC-Dropout Uncertainty → Grad-CAM       │
     │  → Signed PDF Report → Audit Ledger        │
     └────────────────────────────────────────────┘
```

## Modules (11)

| # | Module | What it does |
|---|---|---|
| 0 | `ml/preprocess.py` | Resize, normalize, augment blood smear images |
| 1 | `ml/model.py` | TinyCNN + ResNet-18 classifiers |
| 2 | `ml/mc_dropout.py` | Monte Carlo dropout uncertainty quantification |
| 3 | `ml/attention_gate.py` | Grad-CAM heatmap vs cell mask IoU |
| 4 | `federated/strategies.py` | FedAvg and FedProx aggregation strategies |
| 5 | `federated/client.py` | Flower FL client with non-IID Dirichlet sharding |
| 6 | `edge/infer.py` | ONNX Runtime inference for Raspberry Pi 5 |
| 7 | `api/main.py` | FastAPI backend with SSE streaming |
| 8 | `frontend/` | React UI (Kannada / Hindi / English i18n) |
| 9 | `reporting/report.py` | Ed25519-signed PDF reports |
| 10 | `reporting/audit.py` | Append-only hash-chained audit ledger |

## Tech Stack

**ML & Federated:** PyTorch, Flower FL, ONNX Runtime, scikit-learn, OpenCV

**Backend:** FastAPI, Pydantic, SQLAlchemy, SSE streaming

**Frontend:** React, Vite, TailwindCSS, i18n (Kannada/Hindi/English)

**Security:** Ed25519 digital signatures, AES-256-GCM encryption, JWT auth, append-only hash-chained audit ledger

**Infrastructure:** Docker, docker-compose, GitHub Actions CI, pytest (22 tests)

## Tests

```
$ python -m pytest tests/ -q
22 passed in 4.2s
```

Test coverage includes: determinism (same seed → identical weights), API schema contracts, egress guard, quality gates, preprocessing, MC-Dropout, attention gate (Grad-CAM ↔ cell IoU), audit hash-chain integrity, ONNX ↔ PyTorch parity (≤ 1e-4), Ed25519 signed PDF verification, Flower FedAvg/FedProx factory smoke tests.

## Quick Start

```bash
git clone https://github.com/msujithsali/hemosight.git
cd hemosight
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m scripts.train_toy        # train demo models
python -m pytest                   # 22 tests pass
uvicorn api.main:app --port 8000   # start server
```

Then open `http://127.0.0.1:8000/docs` for the Swagger UI, or run the demo app:

```bash
python demo_app.py  # opens browser at http://127.0.0.1:8010
```

Upload a blood smear image → get cell counts, WBC differential, malaria flag, uncertainty score, and a signed PDF report.

## Honest Limitations

- **Cell detection** uses classical CV (contour-based), not a trained object detector — works on clean slides, struggles with overlapping cells or artifacts.
- **Federated training** runs on a single machine simulating multiple clients — not tested on actual distributed Raspberry Pi hardware.
- **Not clinically validated.** This is a screening aid prototype, not a medical device. No regulatory approval.
- **ResNet-18 metrics above** are for binary malaria classification only — WBC differential uses a separate model path.
- **Cross-dataset generalization** not evaluated — model trained and tested on the same NIH dataset distribution.

## Dataset Citation

Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude RJ, Jaeger S, Thoma GR. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. PeerJ, 6, e4568.

## License

MIT

## Author

**M Sujith Sali** — ISE, JNNCE Shivamogga (VTU Karnataka, Batch 2023–2027)

GitHub: [msujithsali](https://github.com/msujithsali) · LinkedIn: [msujithsali](https://linkedin.com/in/msujithsali)
