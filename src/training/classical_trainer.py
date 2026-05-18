"""Classical model training.

Trivial wrapper - sklearn does the work. Returns trained model + log dict
in the same format as QNN trainer for consistency.
"""
from __future__ import annotations
import time
import numpy as np


def train_classical(model, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Fit a classical model. Returns log with timing."""
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    return {
        "model": model,
        "train_time_seconds": duration,
        "n_params": _count_params(model),
        # No epoch-level log for classical (one fit() call)
        "history": [],
    }


def _count_params(model) -> int:
    """Best-effort param count for sklearn models."""
    if hasattr(model, "coef_"):
        # LR, linear SVM
        return int(np.prod(model.coef_.shape)) + int(np.prod(model.intercept_.shape))
    if hasattr(model, "coefs_"):
        # MLP
        return sum(int(np.prod(w.shape)) for w in model.coefs_) + \
               sum(int(np.prod(b.shape)) for b in model.intercepts_)
    if hasattr(model, "n_estimators"):
        # Random Forest - count nodes
        try:
            return sum(t.tree_.node_count for t in model.estimators_)
        except AttributeError:
            return -1
    return -1  # unknown
