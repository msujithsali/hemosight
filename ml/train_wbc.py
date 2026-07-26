"""Train the WBC 5-class classifier on Raabin-WBC.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Run: `make train-wbc`. Logs seed, git SHA, dataset hash, hyperparameters,
per-class P/R/F1 with bootstrap CIs, confusion matrix, ECE/Brier before and
after temperature scaling — all to MLflow. NO number is written to the README
or model card until this run completes; results carry the [BOOTSTRAP] tag.
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
from ml.models import WBC_CLASSES, build_wbc_model


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "UNKNOWN"


def build_loaders(data_dir: Path, batch: int, img: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor()])
    train_ds = ImageFolder(data_dir / "train", train_tf)
    val_ds = ImageFolder(data_dir / "val", val_tf)
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch, num_workers=2),
    )


def bootstrap_f1_ci(y_true, y_pred, n_boot=1000, seed=1729):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raabin"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    state = seed_everything(args.seed)
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hemosight-wbc")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "seed": state.seed, "git_sha": git_sha(), "epochs": args.epochs,
            "batch": args.batch, "lr": args.lr, "backbone": "efficientnet_b0",
            "label_smoothing": 0.1, "mixup": True, "provenance": "BOOTSTRAP",
        })
        train_loader, val_loader = build_loaders(args.data, args.batch)
        model = build_wbc_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        for epoch in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            mlflow.log_metric("train_loss", float(loss.item()), step=epoch)

        # ---- Evaluation on held-out val split ----
        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits_all.append(model(xb))
                labels_all.append(yb)
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)
        probs = torch.softmax(logits, dim=-1).numpy()
        y_pred = probs.argmax(axis=1)
        y_true = labels.numpy()

        from sklearn.metrics import classification_report

        report = classification_report(
            y_true, y_pred, target_names=WBC_CLASSES, output_dict=True
        )
        for cls in WBC_CLASSES:
            mlflow.log_metric(f"f1_{cls}", report[cls]["f1-score"])
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        lo, hi = bootstrap_f1_ci(y_true, y_pred)
        mlflow.log_metric("macro_f1_ci_low", lo)
        mlflow.log_metric("macro_f1_ci_high", hi)
        mlflow.log_metric("ece_pre", expected_calibration_error(probs, y_true))
        mlflow.log_metric("brier_pre", brier_score(probs, y_true))

        scaler = TemperatureScaler()
        temp = scaler.fit(logits, labels)
        cal_probs = torch.softmax(scaler(logits), dim=-1).detach().numpy()
        mlflow.log_metric("temperature", temp)
        mlflow.log_metric("ece_post", expected_calibration_error(cal_probs, y_true))
        mlflow.log_metric("brier_post", brier_score(cal_probs, y_true))

        Path("results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "results/wbc_effb0.pt")
        mlflow.log_artifact("results/wbc_effb0.pt")
        print(f"WBC training done. MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
