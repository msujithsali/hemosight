# Model Card — HemoSight

> **SCREENING AID ONLY — NOT A DIAGNOSTIC DEVICE.** Output is always an
> abnormality flag + calibrated confidence + needs-review flag, never a disease
> name. Confirm every result with a qualified clinician.

Author: **M Sujith Sali**, ISE Dept, VTU Karnataka.

## Intended use
Assist ASHA / PHC workers in **triaging** peripheral blood smears in
resource-limited rural Indian primary health centres, flagging smears that
warrant review by a qualified pathologist. It is a pre-screening aid to
prioritise referrals — not a substitute for laboratory diagnosis.

## Models
| Component | Architecture | Dataset | Provenance |
|-----------|-------------|---------|-----------|
| Cell detection | YOLOv8n/m | BCCD | `[BOOTSTRAP]` |
| WBC 5-class | EfficientNet-B0 + MC-Dropout | Raabin-WBC | `[BOOTSTRAP]` |
| Malaria binary | MobileNetV3-Small + MC-Dropout | NIH Malaria | `[BOOTSTRAP]` |

Uncertainty via MC-Dropout (10 passes). Calibration via temperature scaling,
reported as ECE + Brier pre/post. An **attention gate** cross-checks Grad-CAM
against detected cell masks (IoU < 0.3 ⇒ `ATTENTION_MISALIGNMENT` ⇒ needs
review).

## Metrics
All metrics are **PENDING** until the training runs complete; each is logged to
MLflow with seed, git SHA, and dataset hash, and tagged `[BOOTSTRAP]`. No number
appears here until it exists in an MLflow run.

## Training-data limitations
Raabin-WBC and NIH Malaria are collected in **specific geographies with
specific stains and microscopes**. Colour, staining, and optics in an Indian
PHC will differ. Reported `[BOOTSTRAP]` performance is therefore an **upper
bound in-distribution** and does **not** transfer to PHC conditions without
`[TARGET]`-tagged validation on locally captured, ethically consented samples.

## Known failure modes
- Out-of-distribution staining / illumination (mitigated partly by flat-field +
  gray-world preprocessing and the quality gate, but not eliminated).
- Overlapping / clumped cells degrade detection and counts.
- Low parasitemia may fall below the flagging threshold.
- MC-Dropout is an approximation of epistemic uncertainty, not a guarantee.
- The attention gate reduces, but does not remove, silent misclassification.

## Out-of-scope uses
- Any autonomous diagnosis or treatment decision.
- Any use on real patients without local `[TARGET]` validation, ethics approval,
  and clinician sign-off.
- Detecting conditions outside the trained WBC classes / malaria flag.

## Ethical & regulatory notes
No PII is stored by default (sample ID only). Local artifacts are encrypted
(AES-256-GCM); the audit ledger is append-only and hash-chained. Design intent
aligns with India's DPDP Act 2023 data-minimisation principles. HemoSight is a
research/education portfolio project and is **not** a certified medical device.
