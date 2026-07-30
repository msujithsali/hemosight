"""Tests for differential privacy federated client."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_opacus_import():
    from opacus import PrivacyEngine
    assert PrivacyEngine is not None


def test_dp_client_instantiation():
    from federated.dp_client import DPPHCClient
    # Simple model (no BatchNorm — already GN-compatible)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 2),
    )
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    client = DPPHCClient(model, loader, loader, target_epsilon=50.0)
    assert client is not None


def test_dp_client_get_parameters():
    from federated.dp_client import DPPHCClient
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 2),
    )
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    client = DPPHCClient(model, loader, loader, target_epsilon=50.0)
    params = client.get_parameters({})
    assert len(params) > 0
    assert all(isinstance(p, type(params[0])) for p in params)


def test_dp_privacy_engine_tracks_epsilon():
    """Verify epsilon is tracked after a fit round."""
    from federated.dp_client import DPPHCClient
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 2),
    )
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 2, (16,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    client = DPPHCClient(model, loader, loader, target_epsilon=50.0)
    initial_params = client.get_parameters({})
    new_params, n_samples, metrics = client.fit(initial_params, {})
    assert "epsilon_spent" in metrics
    assert metrics["epsilon_spent"] > 0
    assert n_samples == 16
