"""Train the functional TinyCNN models on the synthetic toy set (CPU, seconds).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Produces REAL weights (results/*.pt) + ONNX exports so the end-to-end
pipeline, API, and edge inference all run for real. Reports the toy val
accuracy honestly — it is synthetic-data accuracy, not clinical accuracy.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from common.seed_everything import seed_everything
from ml.net import build_tiny
from scripts.make_toy_dataset import build as build_toy


def _train(data_dir: Path, num_classes: int, out: Path, epochs: int = 14) -> float:
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train = DataLoader(ImageFolder(data_dir / "train", tf), batch_size=16, shuffle=True)
    val = DataLoader(ImageFolder(data_dir / "val", tf), batch_size=16)
    model = build_tiny(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val:
            correct += int((model(xb).argmax(1) == yb).sum()); total += len(yb)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    # ONNX export for edge path
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model, dummy, str(out.with_suffix(".onnx")),
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    return correct / max(total, 1)


def main() -> None:
    seed_everything(1729)
    build_toy()
    acc_wbc = _train(Path("data/toy/raabin"), 5, Path("results/wbc_tiny.pt"))
    acc_mal = _train(Path("data/toy/malaria"), 2, Path("results/malaria_tiny.pt"))
    print(f"[SYNTHETIC] WBC toy val acc={acc_wbc:.3f}  Malaria toy val acc={acc_mal:.3f}")
    print("Weights -> results/wbc_tiny.pt, results/malaria_tiny.pt (+ .onnx)")


if __name__ == "__main__":
    main()
