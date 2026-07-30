"""HemoSight demo: run one image through the full real-model pipeline.

Usage:
    python demo_real.py                    # uses test_smear.png
    python demo_real.py path/to/image.jpg  # custom image

Outputs a JSON result and prints a human-readable summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import cv2

from ml.pipeline import load_models, run_pipeline

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_smear.png"
    if not Path(img_path).exists():
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: could not read image {img_path}")
        sys.exit(1)

    print("=" * 60)
    print("HemoSight Real-Model Pipeline Demo")
    print("=" * 60)
    print(f"Input: {img_path} ({img.shape[1]}x{img.shape[0]})")
    print()

    print("Loading models...")
    wbc, mal = load_models()
    print()

    print("Running pipeline...")
    result = run_pipeline(img, wbc, mal)

    if isinstance(result, dict) and "quality_error" in result:
        print("Quality gate rejected the image:")
        print(json.dumps(result["quality_error"], indent=2))
        return

    m = result.metrics
    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Analysis ID:      {result.analysis_id}")
    print(f"Model version:    {result.model_version}")
    print(f"Total cells:      {m.total_counts}")
    print(f"WBC differential: {dict(m.wbc_differential)}")
    print(f"Parasite flag:    {m.parasite_flag}")
    if m.parasitemia_estimate_pct is not None:
        print(f"Parasitemia:      {m.parasitemia_estimate_pct}%")
    ag = result.attention_gate; print(f"Attention gate:   status={ag.status}")
    print()
    print(f"Detections: {len(result.detections)}")
    for d in result.detections[:5]:
        print(f"  #{d.id}: {d.class_name}  conf={d.confidence}  uncertainty={d.uncertainty_std}")
    if len(result.detections) > 5:
        print(f"  ... and {len(result.detections)-5} more")
    print()

    out_json = Path("demo_result.json")
    out_json.write_text(result.model_dump_json(indent=2))
    print(f"Full JSON saved to: {out_json}")
    print()
    print(result.disclaimer)

if __name__ == "__main__":
    main()
