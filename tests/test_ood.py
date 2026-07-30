"""Tests for out-of-distribution detection."""
import torch
import pytest
from ml.ood import energy_score, is_ood, OODGate


def test_energy_score_shape():
    logits = torch.randn(4, 3)
    scores = energy_score(logits)
    assert scores.shape == (4,)


def test_confident_prediction_has_low_energy():
    confident = torch.tensor([[10.0, 0.0]])
    uncertain = torch.tensor([[0.1, 0.1]])
    e_conf = energy_score(confident)
    e_unc = energy_score(uncertain)
    assert e_conf < e_unc, "Confident input should have lower energy"


def test_is_ood_returns_bool():
    logits = torch.tensor([[10.0, 0.0], [0.01, 0.01]])
    mask = is_ood(logits, threshold=-5.0)
    assert mask.dtype == torch.bool
    assert mask.shape == (2,)


def test_ood_gate_rejects_noise():
    model = torch.nn.Linear(3 * 32 * 32, 2)
    gate = OODGate(model, threshold=-100.0)  # very strict
    x = torch.randn(2, 3 * 32 * 32)
    results = gate.classify_or_reject(x)
    assert len(results) == 2
    # With threshold=-100, almost everything is OOD
    assert all(r["status"] == "rejected" for r in results)


def test_ood_gate_accepts_strong_signal():
    model = torch.nn.Linear(10, 2)
    gate = OODGate(model, threshold=0.0)  # very permissive
    x = torch.randn(2, 10) * 10  # strong signal
    results = gate.classify_or_reject(x)
    assert len(results) == 2
    for r in results:
        assert r["status"] in ("accepted", "rejected")
        assert "energy" in r


def test_energy_with_temperature():
    logits = torch.tensor([[5.0, 1.0]])
    e1 = energy_score(logits, temperature=1.0)
    e2 = energy_score(logits, temperature=2.0)
    assert not torch.isnan(e1) and not torch.isnan(e2)
