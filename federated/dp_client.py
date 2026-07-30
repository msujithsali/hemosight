"""Differentially-private Flower client using Opacus DP-SGD.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Wraps the standard PHCClient training loop with Opacus PrivacyEngine to
guarantee (epsilon, delta)-differential privacy per FL round. Only noised,
clipped weight deltas leave the client — formal privacy budget prevents
reconstruction of individual patient images from gradients.

Key parameters:
    target_epsilon: privacy budget (lower = more private, noisier)
    target_delta:   failure probability (typically 1/n)
    max_grad_norm:  per-sample gradient clipping bound
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    HAS_OPACUS = True
except ImportError:
    HAS_OPACUS = False


class DPPHCClient(fl.client.NumPyClient):
    """Flower client with per-round DP-SGD via Opacus."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        local_epochs: int = 1,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        if not HAS_OPACUS:
            raise ImportError("Opacus required for DP training: pip install opacus")

        # Opacus requires BatchNorm -> GroupNorm conversion
        self.model = ModuleValidator.fix(model)
        ModuleValidator.validate(self.model, strict=True)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # These get set up fresh each fit() round
        self._optimizer = None
        self._privacy_engine = None

    def _setup_dp(self):
        """Attach a fresh PrivacyEngine for this FL round."""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        privacy_engine = PrivacyEngine()

        self.model, optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.local_epochs,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
        )
        self._optimizer = optimizer
        self._privacy_engine = privacy_engine

    def get_parameters(self, config):
        # Opacus wraps model in GradSampleModule; access _module for state_dict
        m = getattr(self.model, "_module", self.model)
        return [v.cpu().numpy() for v in m.state_dict().values()]

    def set_parameters(self, parameters):
        m = getattr(self.model, "_module", self.model)
        keys = list(m.state_dict().keys())
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        m.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self._setup_dp()
        self.model.train()

        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                self._optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self._optimizer.step()

        spent = self._privacy_engine.get_epsilon(self.target_delta)
        metrics = {"epsilon_spent": float(spent)}
        print(f"[DP] epsilon spent this round: {spent:.2f}")

        return self.get_parameters(config), len(self.train_loader.dataset), metrics

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
