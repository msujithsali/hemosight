# HemoSight Reproducibility Manifest

## Environment

- **python**: 3.14.3
- **platform**: Windows-11-10.0.26200-SP0

## Trained Models

| Model | SHA-256 | Size |
|---|---|---|
| `malaria_resnet18_REAL.pt` | `fd1354a2f9005e28...` | 42.72 MB |
| `wbc_efficientnet_b0_REAL.pt` | `adc8e5bb313f5ce1...` | 15.6 MB |
| `bccd_yolov8s_REAL.pt` | `f15a324a4aba2491...` | 21.33 MB |

## Datasets

- **NIH Malaria Cell Images** (27558 samples): Rajaraman et al., PeerJ 2018 (6:e4568)  
  Source: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- **Kaggle WBC 5-class** (4339 samples): Public Kaggle dataset  
  Source: https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset
- **BCCD** (273 samples): cosmicad/akshaylambda BCCD  
  Source: https://www.kaggle.com/datasets/reighns/bccd-object-detection

## Deterministic Seed: 42

## Training Platform: Kaggle GPU T4 / P100
