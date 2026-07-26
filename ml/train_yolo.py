"""Train YOLOv8 cell detection on BCCD and export to ONNX.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Run: `make train-yolo`. yolov8n for edge, yolov8m for server. Fixed seed,
declared augmentations. Reports mAP@50 and mAP@50-95 on the BCCD test split
into results/detection_metrics.json with the [BOOTSTRAP] tag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.seed_everything import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/bccd/bccd.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    seed_everything(args.seed)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, seed=args.seed,
        deterministic=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, mosaic=1.0, project="results", name="yolo_bccd",
    )
    metrics = model.val(data=args.data, split="test")
    out = {
        "provenance": "BOOTSTRAP",
        "dataset": "BCCD",
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "seed": args.seed,
        "model": args.model,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/detection_metrics.json").write_text(json.dumps(out, indent=2))

    model.export(format="onnx", dynamic=True, opset=17)
    print("YOLO detection metrics:", out)


if __name__ == "__main__":
    main()
