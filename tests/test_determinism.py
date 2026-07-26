"""Same seed -> identical trained weights (Module 0 acceptance test)."""
import torch
import torch.nn as nn

from common.seed_everything import seed_everything


def _train_tiny():
    seed_everything(1729)
    model = nn.Linear(10, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    for _ in range(50):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_determinism_same_seed_same_weights():
    w1 = _train_tiny()
    w2 = _train_tiny()
    assert torch.equal(w1, w2), "Deterministic training must reproduce weights"
