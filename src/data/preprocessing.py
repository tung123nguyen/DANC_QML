"""Preprocessing pipeline: scale + feature select (split is upstream).

The train/test split now happens in sampling.py (sample_train_test), so this
module only handles the post-split steps:
    1. Drop blacklisted columns
    2. Fit scaler on train, transform both
    3. Fit MI feature selector on train, transform both

Leakage prevention rule: scaler.fit() and selector.fit() see ONLY train data.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.data.feature_blacklist import get_blacklist

log = logging.getLogger(__name__)


def scale_and_select(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_features: int,
    label_col: str = "binary_label",
    seed: int = 0,
) -> dict:
    """Apply scaling and feature selection to pre-split train/test DataFrames.

    The train and test sets must already be sampled to their final sizes.
    For S1 (same source), train and test come from sample_train_test() on the
    same DataFrame. For S3 (cross-domain), train and test come from different
    source files.

    Args:
        train_df: Training data (already sampled, includes label_col)
        test_df: Test data (already sampled, includes label_col)
        n_features: Number of features to keep via MI selection
        label_col: Column name for the binary label
        seed: Random seed (mutual_info_classif uses RNG)

    Returns:
        dict with X_train, X_test, y_train, y_test, feature_names, scaler, selector
    """
    blacklist = set(get_blacklist())

    # Use the intersection of columns - important when train and test come
    # from different files (S3) and may have minor column differences
    train_cols = [c for c in train_df.columns if c not in blacklist]
    test_cols = [c for c in test_df.columns if c not in blacklist]
    feature_cols = [c for c in train_cols if c in test_cols]
    missing = set(train_cols) - set(test_cols)
    if missing:
        log.warning("Dropping %d columns missing from test: %s", len(missing), sorted(missing))

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df[label_col].values.astype(np.int64)

    log.info(
        "Pre-scaling: X_train=%s, X_test=%s, %d candidate features",
        X_train.shape, X_test.shape, X_train.shape[1],
    )

    # Scale: fit on train ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature select: fit on train ONLY
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=seed),
        k=n_features,
    )
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    selected_idx = selector.get_support(indices=True)
    selected_names = [feature_cols[i] for i in selected_idx]
    log.info("Selected %d features: %s", n_features, selected_names)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": selected_names,
        "scaler": scaler,
        "selector": selector,
    }
