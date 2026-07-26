"""Flower NumPyClient wrapping the WBC classifier for one simulated PHC.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Only weight deltas leave the client — never images (hard boundary #4). Each
client trains on a non-IID class-imbalanced shard to mimic real per-PHC
caseload variation. Optional Opacus DP can be toggled by the simulator.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl


class PHCClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, local_epochs: int = 1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                self.opt.zero_grad()
                self.loss_fn(self.model(xb), yb).backward()
                self.opt.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in self.val_loader:
                out = self.model(xb)
                loss_sum += float(self.loss_fn(out, yb).item())
                correct += int((out.argmax(1) == yb).sum().item())
                total += len(yb)
        acc = correct / max(total, 1)
        return loss_sum / max(len(self.val_loader), 1), total, {"accuracy": acc}
