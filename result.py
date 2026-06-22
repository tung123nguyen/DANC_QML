"""Summarize results: run, train_accuracy, train_f1, test_accuracy, test_f1."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.io import get_results_root

COLUMNS = ["train_accuracy", "train_f1", "test_accuracy", "test_f1"]
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")


def collect() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(get_results_root().iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("SMOKE_"):
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():  # skip incomplete / failed runs
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        row = {"run": TIMESTAMP_RE.sub("", run_dir.name)}
        row.update({c: metrics.get(c) for c in COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=["run", *COLUMNS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result.csv")
    args = parser.parse_args()

    df = collect()
    if df.empty:
        print("No completed runs found in", get_results_root())
        return

    # Some configs were run more than once; keep the most recent per run name.
    # The timestamp prefix sorts the newest folder last before it is stripped,
    # but it is already stripped here, so dedup on run name keeping last seen.
    df = df.drop_duplicates(subset="run", keep="last").reset_index(drop=True)

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
