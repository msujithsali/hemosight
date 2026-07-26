"""Global determinism utility for HemoSight.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

Fixes every source of nondeterminism we can reach: Python `random`,
NumPy, PyTorch CPU + CUDA, cuDNN, and the process hash seed. Import and
call ``seed_everything(seed)`` at the very top of every train/eval/infer
entrypoint BEFORE any model or dataloader is constructed.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass


DEFAULT_SEED = 1729  # Ramanujan-Hardy taxicab; fixed project-wide default.


@dataclass(frozen=True)
class SeedState:
    seed: int
    python_hash_seed: str
    torch_available: bool
    cuda_available: bool


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> SeedState:
    """Seed every RNG we can and (optionally) force deterministic kernels.

    Returns a :class:`SeedState` describing what was seeded so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep in practice; guard for tooling.
        pass

    torch_available = False
    cuda_available = False
    try:
        import torch

        torch_available = True
        torch.manual_seed(seed)
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt into deterministic algorithms where torch supports it.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass

    return SeedState(
        seed=seed,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        torch_available=torch_available,
        cuda_available=cuda_available,
    )


if __name__ == "__main__":  # pragma: no cover
    print(seed_everything())
