"""CIC-IDS2017 data loading and cleaning.

Loads raw CSV files, cleans them (strip whitespace, handle inf/NaN,
binary labels), and caches the result as parquet for fast re-loading.

The raw CIC-IDS2017 columns have leading spaces (e.g. ' Label' not 'Label').
This is a well-known bug in the dataset. We strip whitespace once at load time.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import get_data_root

log = logging.getLogger(__name__)

# Maps scenario file ID to the actual CSV filename in data/raw/
SCENARIO_FILES = {
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "wednesday_dos": "Wednesday-workingHours.pcap_ISCX.csv",
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (CIC-IDS2017 has leading spaces)."""
    df.columns = df.columns.str.strip()
    return df


def _replace_inf_drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/- inf with NaN, then drop rows with any NaN.

    inf typically comes from feature ratios like 'Flow Bytes/s' when duration=0.
    These are unfixable - drop them.
    """
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    n_after = len(df)
    if n_before > n_after:
        log.info("Dropped %d rows with NaN/inf (kept %d)", n_before - n_after, n_after)
    return df


def _add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary_label column: 0 = BENIGN, 1 = anything else.

    Handles variants 'BENIGN', ' BENIGN', 'Benign'.
    """
    if "Label" not in df.columns:
        raise KeyError("'Label' column missing. Check column stripping.")

    def is_benign(label: str) -> bool:
        return str(label).strip().upper() == "BENIGN"

    df["binary_label"] = df["Label"].apply(lambda x: 0 if is_benign(x) else 1)
    return df


def load_raw(scenario_id: str, force_reload: bool = False) -> pd.DataFrame:
    """Load and clean a scenario file. Caches to parquet on first load.

    Args:
        scenario_id: One of SCENARIO_FILES keys.
        force_reload: If True, re-read CSV even if cache exists.

    Returns:
        Cleaned DataFrame with binary_label column.
    """
    if scenario_id not in SCENARIO_FILES:
        raise ValueError(f"Unknown scenario: {scenario_id}. Known: {list(SCENARIO_FILES)}")

    data_root = get_data_root()
    cache_path = data_root / "processed" / f"{scenario_id}.parquet"

    if cache_path.exists() and not force_reload:
        log.info("Loading cached: %s", cache_path)
        return pd.read_parquet(cache_path)

    csv_path = data_root / "raw" / SCENARIO_FILES[scenario_id]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {csv_path}\n"
            f"Download CIC-IDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
            f"and place '{SCENARIO_FILES[scenario_id]}' in {data_root}/raw/"
        )

    log.info("Loading raw CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    df = _clean_columns(df)
    df = _replace_inf_drop_na(df)
    df = _add_binary_label(df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    log.info("Cached cleaned data: %s (%d rows)", cache_path, len(df))

    return df


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Quick sanity check: how many of each class."""
    return df["Label"].value_counts().to_dict()
