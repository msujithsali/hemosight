"""Reproducibility manifest: model hashes, sizes, dataset citations, env.

Writes REPRODUCIBILITY.md so anyone can verify what was trained on what.
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    manifest = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "models": {},
        "datasets_used": [
            {"name": "NIH Malaria Cell Images", "n": 27558,
             "url": "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
             "citation": "Rajaraman et al., PeerJ 2018 (6:e4568)"},
            {"name": "Kaggle WBC 5-class", "n": 4339,
             "url": "https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset",
             "citation": "Public Kaggle dataset"},
            {"name": "BCCD", "n": 273,
             "url": "https://www.kaggle.com/datasets/reighns/bccd-object-detection",
             "citation": "cosmicad/akshaylambda BCCD"},
        ],
        "seed": 42,
        "training_platform": "Kaggle GPU T4 / P100",
    }
    for name in ["malaria_resnet18_REAL.pt", "wbc_efficientnet_b0_REAL.pt",
                 "bccd_yolov8s_REAL.pt"]:
        p = Path("results") / name
        if p.exists():
            manifest["models"][name] = {
                "sha256": sha256(p),
                "size_mb": round(p.stat().st_size / (1024*1024), 2),
            }

    lines = ["# HemoSight Reproducibility Manifest\n"]
    lines.append("## Environment\n")
    for k, v in manifest["environment"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Trained Models\n")
    lines.append("| Model | SHA-256 | Size |")
    lines.append("|---|---|---|")
    for name, meta in manifest["models"].items():
        lines.append(f"| `{name}` | `{meta['sha256'][:16]}...` | {meta['size_mb']} MB |")
    lines.append("\n## Datasets\n")
    for d in manifest["datasets_used"]:
        lines.append(f"- **{d['name']}** ({d['n']} samples): {d['citation']}  ")
        lines.append(f"  Source: {d['url']}")
    lines.append(f"\n## Deterministic Seed: {manifest['seed']}\n")
    lines.append(f"## Training Platform: {manifest['training_platform']}\n")

    Path("REPRODUCIBILITY.md").write_text("\n".join(lines))
    Path("results/manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote REPRODUCIBILITY.md and results/manifest.json")


if __name__ == "__main__":
    run()
