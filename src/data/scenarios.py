"""Experimental scenario builders.

Each scenario takes (train_per_class, test_per_class, n_features, seed) and
returns a preprocessed dict ready for the trainer.

Key change from v1: train and test samples are picked DISJOINTLY at the row
level. "train_per_class = 100" now means exactly 100 train rows per class,
not 100 rows total pre-split.

Scenarios:
    S1: Same-source Friday DDoS (BENIGN vs DDoS)
    S2: Same-source Friday PortScan (BENIGN vs PortScan)
    S3: Cross-attack (train Wednesday DoS, test Friday DDoS)
"""
from __future__ import annotations
import logging
from src.data.loader import load_raw
from src.data.sampling import sample_train_test, sample_balanced
from src.data.preprocessing import scale_and_select

log = logging.getLogger(__name__)


def build_scenario(
    scenario: str,
    train_per_class: int,
    test_per_class: int,
    n_features: int,
    seed: int = 0,
) -> dict:
    """Dispatch to scenario builder."""
    builders = {
        "S1": _build_s1_friday_ddos,
        "S2": _build_s2_friday_portscan,
        "S3": _build_s3_cross_attack,
    }
    if scenario not in builders:
        raise ValueError(f"Unknown scenario: {scenario}. Known: {list(builders)}")
    return builders[scenario](train_per_class, test_per_class, n_features, seed)


def _build_s1_friday_ddos(train_per_class, test_per_class, n_features, seed):
    """S1: BENIGN vs DDoS from Friday afternoon (same source)."""
    log.info(
        "Building S1 (Friday DDoS): train=%d/class, test=%d/class, %d features",
        train_per_class, test_per_class, n_features,
    )
    df = load_raw("friday_ddos")
    train_df, test_df = sample_train_test(
        df, train_per_class=train_per_class, test_per_class=test_per_class, seed=seed,
    )
    return scale_and_select(train_df, test_df, n_features=n_features, seed=seed)


def _build_s2_friday_portscan(train_per_class, test_per_class, n_features, seed):
    """S2: BENIGN vs PortScan from Friday afternoon (same source)."""
    log.info(
        "Building S2 (Friday PortScan): train=%d/class, test=%d/class, %d features",
        train_per_class, test_per_class, n_features,
    )
    df = load_raw("friday_portscan")
    train_df, test_df = sample_train_test(
        df, train_per_class=train_per_class, test_per_class=test_per_class, seed=seed,
    )
    return scale_and_select(train_df, test_df, n_features=n_features, seed=seed)


def _build_s3_cross_attack(train_per_class, test_per_class, n_features, seed):
    """S3: Train on Wednesday DoS, test on Friday DDoS.

    Different source files for train and test. Tests whether the model learns
    'attackness' or memorizes a specific attack signature.
    """
    log.info(
        "Building S3 (Wednesday DoS -> Friday DDoS): train=%d/class, test=%d/class, %d features",
        train_per_class, test_per_class, n_features,
    )
    train_src = load_raw("wednesday_dos")
    test_src = load_raw("friday_ddos")

    train_df = sample_balanced(train_src, n_per_class=train_per_class, seed=seed)
    # Use a different seed offset for test so it doesn't overlap with any
    # row indices reused in different scenarios
    test_df = sample_balanced(test_src, n_per_class=test_per_class, seed=seed + 1000)

    return scale_and_select(train_df, test_df, n_features=n_features, seed=seed)
