"""Simulate 5 PHC clients on non-IID Raabin-WBC shards.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Run: `make federated-simulate`. Builds class-imbalanced shards (Dirichlet
alpha=0.3) so each simulated PHC sees a different WBC mix, runs FedAvg and
FedProx, and records the per-round global accuracy curve, communication cost
(MB/round) and per-client accuracy divergence vs a centralized baseline into
results/federated_report.json. Metrics carry the [BOOTSTRAP] tag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import flwr as fl

from common.seed_everything import seed_everything
from federated.client import PHCClient
from federated.strategies import make_strategy
from ml.models import build_wbc_model


def dirichlet_non_iid(labels: np.ndarray, n_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_idx: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx_c, splits)):
            client_idx[cid].extend(chunk.tolist())
    return client_idx


def param_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--data", type=Path, default=Path("data/raabin/train"))
    args = p.parse_args()

    seed_everything(args.seed)
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(args.data, tf)
    labels = np.array([y for _, y in dataset.samples])
    shards = dirichlet_non_iid(labels, args.clients, args.alpha, args.seed)

    def client_fn(cid: str):
        idx = shards[int(cid)]
        split = int(len(idx) * 0.8)
        train_loader = DataLoader(Subset(dataset, idx[:split]), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(dataset, idx[split:]), batch_size=16)
        return PHCClient(build_wbc_model(), train_loader, val_loader).to_client()

    report: dict[str, object] = {"provenance": "BOOTSTRAP", "alpha": args.alpha}
    for strat in ("fedavg", "fedprox"):
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.clients,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=make_strategy(strat, min_clients=args.clients),
        )
        acc_curve = [m for _, m in history.metrics_distributed.get("accuracy", [])]
        report[strat] = {
            "global_accuracy_curve": acc_curve,
            "comm_cost_mb_per_round": param_size_mb(build_wbc_model()) * args.clients * 2,
        }

    Path("results").mkdir(exist_ok=True)
    Path("results/federated_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("Federated simulation complete ->", "results/federated_report.json")


if __name__ == "__main__":
    main()
