"""Preprocessing keeps shape/dtype and is deterministic."""
import numpy as np

from ml.preprocess import preprocess


def test_preprocess_shape_dtype():
    rng = np.random.default_rng(1)
    img = rng.integers(30, 220, (256, 256, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_preprocess_deterministic():
    rng = np.random.default_rng(2)
    img = rng.integers(30, 220, (256, 256, 3), dtype=np.uint8)
    assert np.array_equal(preprocess(img), preprocess(img))
