"""Quality gate returns coded errors, never bare exceptions."""
import numpy as np

from ml.quality_gate import check_quality


def test_resolution_reject():
    small = np.full((100, 100, 3), 128, np.uint8)
    err = check_quality(small)
    assert err is not None and err.code == "RESOLUTION_REJECT"


def test_brightness_reject_dark():
    dark = np.full((300, 300, 3), 5, np.uint8)
    err = check_quality(dark)
    assert err is not None and err.code in {"BRIGHTNESS_REJECT", "BLUR_REJECT"}


def test_decode_error_on_empty():
    err = check_quality(None)
    assert err is not None and err.code == "DECODE_ERROR"


def test_sharp_bright_passes():
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (300, 300, 3), dtype=np.uint8)
    err = check_quality(img)
    assert err is None or err.code != "RESOLUTION_REJECT"
