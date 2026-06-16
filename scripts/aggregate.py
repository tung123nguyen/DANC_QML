"""Aggregate all results into a single CSV."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.io import get_results_root


def collect_runs() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(get_results_root().iterdir()):
        if not run_dir.is_dir():
            continue
        # Fix #2: ignore smoke-test result folders. Real runs never carry
        # this prefix; smoke runs use train=20 and would pollute the means.
        if run_dir.name.startswith("SMOKE_"):
            continue
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        if not (cfg_path.exists() and metrics_path.exists()):
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        row = {
            "run": run_dir.name,
            "scenario": cfg["data"]["scenario"],
            "train_samples_per_class": cfg["data"]["train_samples_per_class"],
            "test_samples_per_class": cfg["data"]["test_samples_per_class"],
            "n_features": cfg["data"]["n_features"],
            "seed": cfg["experiment"]["seed"],
            "model_type": cfg["model"]["type"],
            "model_name": cfg["model"].get("name", ""),
            "encoding": cfg["model"].get("encoding", ""),
            "ansatz": cfg["model"].get("ansatz", ""),
            "depth": cfg["model"].get("depth", ""),
            "n_qubits": cfg["model"].get("n_qubits", ""),
        }
        for k, v in metrics.items():
            if k in ("train_confusion", "test_confusion"):
                continue
            row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results_summary.csv")
    args = parser.parse_args()

    df = collect_runs()
    if df.empty:
        print("No completed runs found in", get_results_root())
        return

    # Fix #2: keep only the intended training-set sizes. A stray smoke-test
    # row (train=20) without the SMOKE_ prefix once leaked into the CSV.
    n_before = len(df)
    df = df[df["train_samples_per_class"].isin([100, 250, 500])]
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} row(s) with non-standard train size")

    # Fix #3: deduplicate. Some runs (e.g. Q4 S1 N500) were executed twice;
    # keep the most recent folder per (scenario, N, seed, model identity).
    # 'run' starts with a timestamp, so sorting by it puts newest last.
    dup_keys = [
        "scenario", "train_samples_per_class", "seed",
        "model_type", "model_name", "encoding", "ansatz",
    ]
    n_before = len(df)
    df = (
        df.sort_values("run")
        .drop_duplicates(subset=dup_keys, keep="last")
        .reset_index(drop=True)
    )
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} duplicate run(s)")

    # Strip the leading "YYYY-MM-DD_HH-MM-SS_" timestamp from the run name so
    # the CSV shows clean config identifiers. Done after dedup, which needs the
    # timestamp to sort the newest duplicate last.
    df["run"] = df["run"].str.replace(
        r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", regex=True
    )

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")

    if "test_f1" in df.columns:
        print("\nTest F1 by model (mean over seeds):")
        group_cols = ["model_type", "model_name", "encoding", "ansatz",
                      "scenario", "train_samples_per_class"]
        summary = (
            df.groupby(group_cols)["test_f1"]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print(summary)


if __name__ == "__main__":
    main()