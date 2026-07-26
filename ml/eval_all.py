"""Aggregate every logged MLflow metric into results/metrics_summary.json.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Run: `make eval-all`. Reads the latest run of each experiment and writes a
single provenance-tagged summary the README references. If an experiment has
no runs yet, its section is written as "PENDING" rather than fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path

EXPERIMENTS = ["hemosight-wbc", "hemosight-malaria"]


def main() -> None:
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    summary: dict[str, object] = {"provenance": "BOOTSTRAP"}
    client = mlflow.tracking.MlflowClient()
    for exp_name in EXPERIMENTS:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            summary[exp_name] = "PENDING — run the corresponding make target"
            continue
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs:
            summary[exp_name] = "PENDING — no completed run"
            continue
        summary[exp_name] = {"run_id": runs[0].info.run_id, "metrics": runs[0].data.metrics}

    # Detection metrics from YOLO
    det = Path("results/detection_metrics.json")
    summary["detection"] = json.loads(det.read_text()) if det.exists() else "PENDING — make train-yolo"

    Path("results").mkdir(exist_ok=True)
    Path("results/metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
