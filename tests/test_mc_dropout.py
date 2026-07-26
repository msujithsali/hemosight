"""MC-Dropout produces non-zero epistemic uncertainty and stable shapes."""
import torch
import torch.nn as nn

from common.seed_everything import seed_everything
from ml.mc_dropout import mc_dropout_predict


def _toy_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 3))


def test_mc_shapes_and_uncertainty():
    seed_everything(1729)
    model = _toy_model()
    x = torch.randn(4, 8)
    res = mc_dropout_predict(model, x, n_passes=10)
    assert res.mean_probs.shape == (4, 3)
    assert res.uncertainty.shape == (4,)
    assert float(res.std_probs.sum()) > 0.0  # dropout was actually active
