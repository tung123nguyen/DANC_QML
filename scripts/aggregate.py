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
