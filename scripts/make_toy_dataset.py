"""Generate a class-SEPARABLE synthetic dataset so the functional pipeline
actually learns and runs end-to-end (no external download).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Each WBC class gets a distinct hue + nucleus pattern; parasitized malaria
cells get dark ring inclusions. This is SYNTHETIC data for wiring/demo only —
NOT clinical data. Real accuracy requires Raabin-WBC / NIH Malaria.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WBC = ["Neutrophil", "Eosinophil", "Basophil", "Lymphocyte", "Monocyte"]
WBC_HUES = [10, 25, 130, 160, 100]  # distinct HSV hues per class


def _cell(hue: int, rng, parasite: bool = False, size: int = 96) -> np.ndarray:
    img = np.full((size, size, 3), 235, np.uint8)
    cx, cy = size // 2 + rng.integers(-6, 6), size // 2 + rng.integers(-6, 6)
    r = rng.integers(size // 4, size // 3)
    hsv = np.uint8([[[hue, 180, 200]]])
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
    cv2.circle(img, (cx, cy), int(r), color, -1)
    cv2.circle(img, (cx, cy), int(r * 0.5), (120, 40, 120), -1)  # nucleus
    if parasite:
        for _ in range(rng.integers(5, 9)):
            px, py = cx + rng.integers(-r, r), cy + rng.integers(-r, r)
            cv2.circle(img, (int(px), int(py)), rng.integers(4, 7), (20, 20, 90), -1)
            cv2.circle(img, (int(px), int(py)), 2, (10, 10, 10), -1)
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def build(root: Path = Path("data/toy")) -> None:
    for split, n in (("train", 40), ("val", 10)):
        for ci, cls in enumerate(WBC):
            rng = np.random.default_rng(ci * 100 + (0 if split == "train" else 1))
            for i in range(n):
                out = root / "raabin" / split / cls
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"{i}.png"), _cell(WBC_HUES[ci], rng))
        for lbl, (cls, para) in enumerate([("Uninfected", False), ("Parasitized", True)]):
            rng = np.random.default_rng(500 + lbl + (0 if split == "train" else 1))
            for i in range(n):
                out = root / "malaria" / split / cls
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"{i}.png"), _cell(160, rng, parasite=para))
    print(f"Synthetic (class-separable) toy dataset written under {root}")


def make_field(n_cells: int = 12, size: int = 320, seed: int = 0,
               parasite: bool = False) -> "np.ndarray":
    """Compose a realistic multi-cell smear FIELD (passes the blur/brightness
    quality gate and gives the detector several cells)."""
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size, 3), 198, np.uint8)
    canvas += rng.integers(-6, 6, canvas.shape, dtype=np.int16).astype(np.uint8)
    for _ in range(n_cells):
        hue = int(rng.choice(WBC_HUES))
        cell = _cell(hue, rng, parasite=parasite and rng.random() < 0.5, size=64)
        x = int(rng.integers(0, size - 64)); y = int(rng.integers(0, size - 64))
        canvas[y:y + 64, x:x + 64] = cell
    return canvas


if __name__ == "__main__":
    build()
