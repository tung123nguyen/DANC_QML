"""Reproducibility helpers.

Set seeds across all RNGs we use. Call set_all_seeds() at the start of each
experiment, BEFORE any data loading or model building.
"""
from __future__ import annotations
import os
import random
import numpy as np


def set_all_seeds(seed: int) -> None:
    """Set seeds for python random, numpy, torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic mode - slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
