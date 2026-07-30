"""End-to-end federated simulation with 3 simulated PHC clients.

Uses toy data (deterministic seeds) so the simulation is <60s but real code
path: PHCClient, non-IID sharding, FedAvg aggregation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from federated.client import PHCClient
from federated.strategies import make_strategy


def make_shard(seed, n=64):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.randn(n, 3, 32, 32)
    # Non-IID: shard 0 has more class 0, shard 1 has more class 1
    class_bias = seed % 2
    y = torch.tensor([class_bias if np.random.rand() < 0.7 else 1 - class_bias for _ in range(n)])
    return DataLoader(TensorDataset(x, y), batch_size=8)


def toy_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3*32*32, 64), torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 2),
    )


def run():
    N_CLIENTS = 3
    clients = []
    for i in range(N_CLIENTS):
        m = toy_model()
        loader = make_shard(seed=i)
        val = make_shard(seed=i + 100, n=32)
        clients.append(PHCClient(m, loader, val, local_epochs=1))

    # Get initial params
    init_params = clients[0].get_parameters({})

    # 3 rounds of FedAvg (manual, since we're not using real flower server)
    results = []
    for round_i in range(3):
        client_updates = []
        for c in clients:
            new_params, n_samples, _ = c.fit(init_params, {})
            client_updates.append((new_params, n_samples))
        # Weighted average
        total = sum(n for _, n in client_updates)
        avg = [sum(p[i] * n / total for p, n in client_updates) for i in range(len(init_params))]
        init_params = avg
        # Evaluate on all clients
        accs = []
        for c in clients:
            loss, n, m = c.evaluate(init_params, {})
            accs.append(m["accuracy"])
        results.append({"round": round_i + 1, "mean_accuracy": round(sum(accs) / len(accs), 4)})
        print(f"Round {round_i+1}: mean_acc={results[-1]['mean_accuracy']}")

    strat_name = "FedAvg"
    out = {
        "strategy": strat_name,
        "n_clients": N_CLIENTS,
        "n_rounds": 3,
        "per_round": results,
        "sharding": "non-IID class-imbalanced (seed-based)",
    }
    Path("results/federated_e2e.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
