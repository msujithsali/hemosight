"""Flower aggregation server entrypoint (mTLS in production).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Starts the FL server with a chosen strategy. In production the gRPC transport
uses mutual TLS (certs mounted from keys/); for local simulation we run
in-process via federated/simulate.py.
"""
from __future__ import annotations

import argparse

import flwr as fl

from federated.strategies import make_strategy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="fedavg")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--address", default="0.0.0.0:8080")
    args = p.parse_args()
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=make_strategy(args.strategy),
    )


if __name__ == "__main__":
    main()
