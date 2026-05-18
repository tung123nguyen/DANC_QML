"""Correctness tests.

These prevent the most common ways thesis results can be wrong:
- data leakage
- incorrect class balance
- metric inconsistency
- train/test overlap

Run before any sweep:
    pytest tests/
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import scale_and_select
from src.data.sampling import sample_balanced, sample_train_test
from src.data.feature_blacklist import get_blacklist
from src.evaluation.metrics import compute_metrics


def _make_fake_df(n_per_class: int = 500, n_features: int = 20, seed: int = 0):
    """Synthetic dataset with CIC-IDS2017-like structure."""
    rng = np.random.default_rng(seed)
    X_benign = rng.normal(0, 1, (n_per_class, n_features))
    X_attack = rng.normal(1.5, 1, (n_per_class, n_features))
    X = np.vstack([X_benign, X_attack])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["binary_label"] = y
    df["Label"] = ["BENIGN"] * n_per_class + ["DDoS"] * n_per_class
    return df


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_balanced_exact_count():
    df = _make_fake_df(n_per_class=1000)
    out = sample_balanced(df, n_per_class=100, seed=0)
    assert len(out) == 200
    assert (out["binary_label"] == 0).sum() == 100
    assert (out["binary_label"] == 1).sum() == 100


def test_sample_balanced_reproducible():
    df = _make_fake_df(n_per_class=1000)
    a = sample_balanced(df, n_per_class=100, seed=42)
    b = sample_balanced(df, n_per_class=100, seed=42)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_sample_balanced_raises_on_insufficient():
    df = _make_fake_df(n_per_class=50)
    with pytest.raises(ValueError):
        sample_balanced(df, n_per_class=100)


def test_sample_train_test_correct_sizes():
    """Train and test should have exactly the requested sizes per class."""
    df = _make_fake_df(n_per_class=1000)
    train, test = sample_train_test(df, train_per_class=100, test_per_class=500, seed=0)
    assert (train["binary_label"] == 0).sum() == 100
    assert (train["binary_label"] == 1).sum() == 100
    assert (test["binary_label"] == 0).sum() == 500
    assert (test["binary_label"] == 1).sum() == 500


def test_sample_train_test_disjoint():
    """CRITICAL: train and test must not share any rows."""
    df = _make_fake_df(n_per_class=1000)
    df_with_id = df.reset_index().rename(columns={"index": "row_id"})
    train, test = sample_train_test(df_with_id, train_per_class=100, test_per_class=200, seed=0)
    train_ids = set(train["row_id"])
    test_ids = set(test["row_id"])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Train and test overlap on {len(overlap)} rows!"


def test_sample_train_test_raises_on_insufficient():
    """Should raise if total needed exceeds class size."""
    df = _make_fake_df(n_per_class=100)
    with pytest.raises(ValueError):
        sample_train_test(df, train_per_class=80, test_per_class=80, seed=0)


def test_sample_train_test_reproducible():
    """Same seed -> same split."""
    df = _make_fake_df(n_per_class=1000)
    df = df.reset_index().rename(columns={"index": "row_id"})
    a_train, a_test = sample_train_test(df, train_per_class=50, test_per_class=50, seed=7)
    b_train, b_test = sample_train_test(df, train_per_class=50, test_per_class=50, seed=7)
    assert set(a_train["row_id"]) == set(b_train["row_id"])
    assert set(a_test["row_id"]) == set(b_test["row_id"])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_scale_and_select_train_centered():
    """Train data should be ~0 mean after scaling (it was fit on train)."""
    df = _make_fake_df(n_per_class=500)
    train, test = sample_train_test(df, train_per_class=200, test_per_class=200, seed=0)
    out = scale_and_select(train, test, n_features=4, seed=0)
    assert np.abs(out["X_train"].mean()) < 0.1
    assert out["X_train"].shape == (400, 4)
    assert out["X_test"].shape == (400, 4)


def test_scale_and_select_no_leakage_in_test_mean():
    """Test mean should NOT be ~0 in general (otherwise scaler saw test = leak).

    With our synthetic data both classes have the same distribution as train,
    so test mean will be close to 0. The real test is: scaler.mean_ should be
    computed from train only.
    """
    df = _make_fake_df(n_per_class=500)
    train, test = sample_train_test(df, train_per_class=100, test_per_class=100, seed=0)
    out = scale_and_select(train, test, n_features=4, seed=0)
    # Verify the scaler was fit on train (not on train+test combined)
    train_raw = train[[c for c in train.columns if c not in get_blacklist()]].values
    expected_mean = train_raw.mean(axis=0)
    np.testing.assert_allclose(out["scaler"].mean_, expected_mean, rtol=1e-9)


# ---------------------------------------------------------------------------
# Feature blacklist
# ---------------------------------------------------------------------------

def test_blacklist_includes_ports_and_protocol():
    """These columns must be blacklisted to prevent shortcut learning."""
    bl = set(get_blacklist())
    assert "Destination Port" in bl
    assert "Source Port" in bl
    assert "Flow ID" in bl
    assert "Timestamp" in bl
    assert "Protocol" in bl
    assert "Label" in bl
    assert "binary_label" in bl


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1, 1])
    m = compute_metrics(y, y)
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["precision_attack"] == 1.0
    assert m["recall_attack"] == 1.0
    assert m["fnr"] == 0.0
    assert m["fpr"] == 0.0


def test_metrics_all_benign_predictions():
    """Model predicts all BENIGN: no false alarms (fpr=0) but missed all attacks."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["recall_attack"] == 0.0
    assert m["fnr"] == 1.0
    assert m["fpr"] == 0.0
    assert m["precision_attack"] == 0.0  # no positives predicted at all


def test_metrics_all_attack_predictions():
    """Model predicts all ATTACK: catches all but flags every benign."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m["recall_attack"] == 1.0
    assert m["fnr"] == 0.0
    assert m["fpr"] == 1.0
    assert m["precision_attack"] == 3 / 5


def test_metrics_fnr_recall_consistency():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert abs(m["fnr"] - (1.0 - m["recall_attack"])) < 1e-9
    assert abs(m["fpr"] - (1.0 - m["recall_benign"])) < 1e-9


def test_metrics_precision_attack():
    """precision = TP / (TP + FP). Sanity check the formula."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 0])  # TP=2, FP=2, TN=1, FN=1
    m = compute_metrics(y_true, y_pred)
    assert abs(m["precision_attack"] - 2 / 4) < 1e-9
    assert abs(m["recall_attack"] - 2 / 3) < 1e-9
