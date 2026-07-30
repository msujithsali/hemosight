# HemoSight: A Federated Edge-AI Blood Smear Screening System

**Author:** M Sujith Sali · ISE Dept, JNNCE Shivamogga, VTU Karnataka  
**Date:** 2026

## Abstract

We present HemoSight, an open-source federated-learning framework for on-device blood microscopy screening in Indian primary health centre (PHC) settings. HemoSight integrates three trained models — a ResNet-18 malaria classifier (96.83% accuracy on NIH), an EfficientNet-B0 white-blood-cell 5-class classifier (97.70% on Kaggle WBC), and a YOLOv8s cell detector (90.84% mAP50 on BCCD) — with a trust layer that combines temperature-scaling calibration, energy-based out-of-distribution rejection, MC-Dropout uncertainty quantification, and differentially-private federated learning (Opacus DP-SGD). The system runs offline on Raspberry Pi 5 with a UVC USB microscope, and produces Ed25519-signed reports with a hash-chained audit ledger.

## 1. Problem

Rural Indian PHCs face two constraints for AI-assisted diagnostics: (a) intermittent connectivity, and (b) patient-data privacy regulations. A screening tool that requires uploading images to a central server fails both. Meanwhile, aggregated learning is required to reach diagnostic-grade accuracy, since any single PHC's caseload is too small and biased.

## 2. Method

**Models.** Three classifiers trained on public benchmarks:
- Malaria: ResNet-18, 5 epochs, Adam(1e-4), 80/20 split, NIH 27,558 images.
- WBC: EfficientNet-B0, 5 epochs, Adam(1e-4), Kaggle WBC 4,339 images.
- Detection: YOLOv8s, 25 epochs, imgsz=416, BCCD 273 images.

**Trust layer.**
1. *Temperature scaling* (Guo et al., 2017): learn T on validation logits; compute ECE and Brier score before/after.
2. *Energy-based OOD rejection* (Liu et al., 2020): threshold `-T·log∑exp(f_i(x)/T)`; reject invalid inputs.
3. *MC-Dropout*: N=10 stochastic passes; per-class std as epistemic uncertainty; triggers `needs-review` flag.

**Federated learning.** Flower FL with FedAvg / FedProx (proximal μ=0.1) strategies over non-IID Dirichlet-sharded clients (α=0.5). For patient privacy, we integrate Opacus PrivacyEngine on each client: per-sample gradient clipping (C=1.0), calibrated noise, and (ε, δ)-tracking per round.

**Trustworthy reporting.** Ed25519-signed PDF reports; append-only hash-chained audit ledger; signed model card.

## 3. Results

| Model | Metric | Value | Dataset |
|---|---|---|---|
| Malaria ResNet-18 | Accuracy | 96.83% | NIH (27,558 imgs) |
| Malaria ResNet-18 | ROC-AUC | 99.40% | NIH |
| WBC EfficientNet-B0 | Accuracy | 97.70% | Kaggle WBC (5-class) |
| WBC EfficientNet-B0 | Macro-F1 | 0.95 | Kaggle WBC |
| YOLOv8s | mAP50 | 90.84% | BCCD |
| YOLOv8s | mAP50-95 | 65.51% | BCCD |

Ablation and federated E2E results are in `results/ablation_calibration.json` and `results/federated_e2e.json`.

## 4. Limitations

- Trained and validated on public datasets; **not clinically validated** on Indian PHC blood smears.
- Cell detection is trained on 218 BCCD images — sufficient for a benchmark but limited for diverse real-world staining.
- Federated experiments are simulated on one machine, not deployed to real distributed Raspberry Pis.
- DP privacy budget ε is per-round; total budget over training must be composed carefully.
- OOD threshold is data-dependent and requires tuning on deployment-domain samples.

## 5. Ethical and Regulatory Notes

HemoSight is a **screening aid**, not a diagnostic device. All outputs require confirmation by a qualified pathologist. Real-world deployment requires IRB/ethics approval and patient consent. No patient data was collected for this project.

## 6. References

1. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
2. Liu, W., et al. (2020). Energy-based Out-of-distribution Detection. *NeurIPS*.
3. Rajaraman, S., et al. (2018). Pre-trained CNNs for malaria parasite detection. *PeerJ*, 6, e4568.
4. Kouzehkanan, Z. M., et al. (2022). Raabin-WBC dataset. *Scientific Reports*.
5. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks (FedAvg). *AISTATS*.
6. Li, T., et al. (2020). FedProx: Federated Optimization in Heterogeneous Networks. *MLSys*.
7. Abadi, M., et al. (2016). Deep Learning with Differential Privacy (DP-SGD). *CCS*.
