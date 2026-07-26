# HemoSight — Interview Defence Handbook

Author: **M Sujith Sali**, ISE Dept, VTU Karnataka. For 2027 placements
(TCS, Infosys, Wipro, Cognizant, Accenture, JSW, product companies) and
SIH / IEEE / KSCST tracks.

> One-line pitch: *"HemoSight is an offline-first federated edge-AI screening
> aid for blood smears in rural PHCs — it flags abnormalities with calibrated
> confidence, never diagnoses, keeps patient images on-device, and shares only
> model weight deltas across clinics."*

## 1. System-design walkthrough (whiteboard order)
1. **Capture** — Pi 5 + UVC microscope; camera adapter is hot-swappable
   (Picamera2 or OpenCV VideoCapture) via config.
2. **Quality gate** — Laplacian-variance blur check, brightness window,
   resolution floor. Returns a *coded* error, never a bare exception.
3. **Preprocess** — fixed order: flat-field → NLM denoise → CLAHE(2.0) →
   gray-world WB. Deterministic for a given input.
4. **Detect** — YOLOv8 (ONNX + XNNPACK on Pi) → cell bboxes.
5. **Classify** — EfficientNet-B0 (WBC) / MobileNetV3-Small (malaria), with
   MC-Dropout for uncertainty.
6. **Calibrate** — temperature scaling; report ECE + Brier.
7. **Attention gate** — Grad-CAM vs cell-mask IoU; misalignment ⇒ needs-review.
8. **Serve** — FastAPI, SSE stream of the 7 pipeline stages; React UI.
9. **Report** — WeasyPrint PDF, Ed25519-signed, hash-chained audit ledger.
10. **Federate** — Flower FedAvg/FedProx over weight deltas; images never leave
    the device.

## 2. STAR answers (compressed)
**Situation** Rural PHCs lack pathologists; smears wait days or travel far.
**Task** Build an offline screening aid that respects patient privacy and never
overclaims. **Action** Designed the 10-stage pipeline above; enforced integrity
in code (egress guard, provenance tags, PENDING-not-fake metrics, determinism);
made it federated so clinics improve a shared model without sharing images.
**Result** A reproducible, test-covered (20 tests, 80% gate), Dockerised system
with signed reports and a clear screening-aid boundary — ready for `[TARGET]`
validation.

## 3. Decision rationale (the questions they will ask)
**Why FedAvg/FedProx, not centralised?** PHC images are sensitive and
bandwidth is scarce. Federation keeps images local; only weight deltas move.
**FedProx** adds a proximal term so heavily **non-IID** per-PHC caseloads don't
pull the global model apart — which centralised training never has to handle
because it sees the pooled distribution.

**Why MC-Dropout, not deep ensembles?** On a Pi 5 with ~4GB usable RAM, storing
and running N independent models is too costly. MC-Dropout gives an epistemic
uncertainty estimate from a **single** model via N stochastic passes — cheap,
and good enough to drive a needs-review flag.

**Why temperature scaling?** Neural nets are typically over-confident. A single
scalar T fit on validation NLL fixes calibration **without changing accuracy**
(it's monotonic in the logits), and we prove it by reporting ECE/Brier pre/post.

**Why ONNX Runtime + XNNPACK on Pi, not PyTorch?** ONNX Runtime with the
XNNPACK provider is lighter and faster for ARM CPU inference; PyTorch's runtime
footprint is heavy for an edge device. We verify ONNX↔PyTorch parity within
1e-4 so accuracy is unchanged.

**Why the attention gate?** A confident-but-wrong classifier is the dangerous
failure. If Grad-CAM attention doesn't overlap the detected cells (IoU < 0.3),
the model is "looking" at the wrong region, so we route to human review instead
of trusting the score.

## 4. Threat model (federated setting)
- **Honest-but-curious server** — sees weight deltas, not images; optional
  Opacus DP (report ε, δ) bounds what a delta leaks.
- **Malicious client / poisoning** — mitigated by robust aggregation and
  per-client divergence monitoring (future: Krum/trimmed-mean).
- **Transport** — mutual TLS on the gRPC channel between clients and aggregator.
- **At rest** — AES-256-GCM for local artifacts; argon2id password hashing;
  Ed25519-signed reports; append-only hash-chained audit ledger.

## 5. Calibration & ECE — the crisp explanation
ECE bins predictions by confidence and measures the gap between average
confidence and average accuracy per bin; a well-calibrated 80%-confidence
bucket should be right ~80% of the time. Brier score is the mean squared error
between predicted probabilities and one-hot truth. We report both before and
after temperature scaling to prove the calibration actually improved.

## 6. DPDP alignment notes
Data minimisation (sample ID only, no PII by default), purpose limitation
(screening triage), storage limitation (local, encrypted), and integrity
(signed reports + audit chain). HemoSight is a research/portfolio project, not a
certified device; real deployment needs ethics approval and clinician oversight.

## 7. Honest limitations to state proactively
"The numbers are on public datasets that don't match Indian PHC staining, so I
report them as `[BOOTSTRAP]` and mark PENDING until trained. Real deployment
needs `[TARGET]` validation. That honesty is a feature of the design, not a
gap." Interviewers reward the candidate who names the boundary before they do.
