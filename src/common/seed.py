"""Deterministic seeding for data sampling, LoRA init, dropout and generation."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # numpy-only tests
        pass


def derive_seed(base: int, tag: str) -> int:
    """A stable sub-seed for one purpose (``"data"``, ``"lora"``, ``"gen"``)."""
    import hashlib

    h = hashlib.sha256(f"{base}:{tag}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)
