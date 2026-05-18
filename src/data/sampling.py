"""Stratified balanced subsampling.

Two patterns:
1. sample_balanced: pick N per class from a single DataFrame (used as a primitive)
2. sample_train_test: pick disjoint train and test sets from a DataFrame,
   with separate N per class for each. This is what scenario builders use.

The train/test separation matters: for the thesis we want to say
"trained on N samples/class", which means N samples/class in the train set,
NOT N samples/class before a 70/30 split.
"""
from __future__ import annotations
import pandas as pd


def sample_balanced(
    df: pd.DataFrame,
    n_per_class: int,
    label_col: str = "binary_label",
    seed: int = 0,
) -> pd.DataFrame:
    """Sample exactly n_per_class rows from each class."""
    parts = []
    for cls, group in df.groupby(label_col):
        if len(group) < n_per_class:
            raise ValueError(
                f"Class {cls} has only {len(group)} rows, "
                f"need {n_per_class}. Reduce n_per_class or check data."
            )
        sampled = group.sample(n=n_per_class, random_state=seed)
        parts.append(sampled)
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def sample_train_test(
    df: pd.DataFrame,
    train_per_class: int,
    test_per_class: int,
    label_col: str = "binary_label",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample DISJOINT train and test sets, balanced per class.

    For each class, pick (train_per_class + test_per_class) distinct rows,
    then split them so train and test have no overlap.

    This expresses "trained on N samples/class" correctly: N is exactly the
    train set size per class, not the pre-split total.

    Returns:
        (train_df, test_df), both shuffled.
    """
    train_parts = []
    test_parts = []
    for cls, group in df.groupby(label_col):
        total_needed = train_per_class + test_per_class
        if len(group) < total_needed:
            raise ValueError(
                f"Class {cls} has only {len(group)} rows, "
                f"need {total_needed} (train={train_per_class} + test={test_per_class})."
            )
        pool = group.sample(n=total_needed, random_state=seed)
        train_parts.append(pool.iloc[:train_per_class])
        test_parts.append(pool.iloc[train_per_class:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    return train_df, test_df
