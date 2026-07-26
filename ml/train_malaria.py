"""Train the malaria binary classifier on NIH Malaria Cell Images.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Run: `make train-malaria`. MobileNetV3-Small (edge-friendly). Logs P/R/F1,
ROC-AUC, PR-AUC with bootstrap CIs, plus calibration (ECE/Brier pre/post
temperature scaling) to MLflow. Results carry the [BOOTSTRAP] tag.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from common.seed_everything import seed_everything
from ml.calibration import TemperatureScaler, brier_score, expected_calibration_error
from ml.models import build_malaria_model


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "UNKNOWN"


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=1729):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    scores, n = [], len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/malaria"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    state = seed_everything(args.seed)
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hemosight-malaria")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    with mlflow.start_run() as run:
        mlflow.log_params({
            "seed": state.seed, "git_sha": git_sha(), "epochs": args.epochs,
            "backbone": "mobilenetv3_small_100", "provenance": "BOOTSTRAP",
        })
        train_ds = ImageFolder(args.data / "train", tf)
        val_ds = ImageFolder(args.data / "val", tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch)

        model = build_malaria_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            mlflow.log_metric("train_loss", float(loss.item()), step=epoch)

        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits_all.append(model(xb))
                labels_all.append(yb)
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)
        probs = torch.softmax(logits, dim=-1).numpy()
        y_true = labels.numpy()
        y_score = probs[:, 1]

        from sklearn.metrics import (average_precision_score, f1_score,
                                     precision_score, recall_score, roc_auc_score)

        mlflow.log_metric("precision", precision_score(y_true, y_score > 0.5))
        mlflow.log_metric("recall", recall_score(y_true, y_score > 0.5))
        mlflow.log_metric("f1", f1_score(y_true, y_score > 0.5))
        mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_score))
        mlflow.log_metric("pr_auc", average_precision_score(y_true, y_score))
        lo, hi = bootstrap_auc_ci(y_true, y_score)
        mlflow.log_metric("roc_auc_ci_low", lo)
        mlflow.log_metric("roc_auc_ci_high", hi)
        mlflow.log_metric("ece_pre", expected_calibration_error(probs, y_true))
        mlflow.log_metric("brier_pre", brier_score(probs, y_true))

        scaler = TemperatureScaler()
        mlflow.log_metric("temperature", scaler.fit(logits, labels))
        cal = torch.softmax(scaler(logits), dim=-1).detach().numpy()
        mlflow.log_metric("ece_post", expected_calibration_error(cal, y_true))
        mlflow.log_metric("brier_post", brier_score(cal, y_true))

        Path("results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "results/malaria_mnv3.pt")
        mlflow.log_artifact("results/malaria_mnv3.pt")
        print(f"Malaria training done. MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
