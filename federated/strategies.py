"""Federated strategies: FedAvg and FedProx.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

FedAvg is Flower's built-in. FedProx adds a proximal term (mu) that keeps
each client's local update close to the global model — this is what makes
convergence robust to the non-IID per-PHC caseload we simulate. We expose a
factory so `simulate.py` can switch strategies from the CLI.
"""
from __future__ import annotations

import flwr as fl


def make_strategy(name: str, min_clients: int = 5, proximal_mu: float = 0.1):
    name = name.lower()
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    if name == "fedavg":
        return fl.server.strategy.FedAvg(**common)
    if name == "fedprox":
        return fl.server.strategy.FedProx(proximal_mu=proximal_mu, **common)
    raise ValueError(f"Unknown strategy {name!r}; use 'fedavg' or 'fedprox'")
