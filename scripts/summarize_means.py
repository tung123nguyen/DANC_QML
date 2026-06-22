"""Mean-over-seeds summary of results_summary.csv.

Collapses the per-seed runs of one config into a single row by stripping the
trailing ``_seed{N}`` from the ``run`` name, then averages the three requested
metrics: train_accuracy, train_f1, test_f1. Output keeps just one identifier
column ``run`` (the config name, e.g. q2_angle_basic_entangler_s1_n100_f4_d4),
like the ``run`` column in results_summary.csv but without the seed.

Usage:
    python scripts/summarize_means.py
    python scripts/summarize_means.py --input results_z_summary.csv --output z_mean.csv
"""
from __future__ import annotations
import argparse
import pandas as pd

METRICS = ["train_accuracy", "train_f1", "test_f1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results_summary.csv")
    p.add_argument("--output", default="results_mean_summary.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)

    # Config name = run name with the trailing _seed<N> removed. Rows sharing it
    # are the seed repeats of the same config (the name already encodes model,
    # scenario, N, depth, readout, trainable-encoding, ...).
    df["run"] = df["run"].str.replace(r"_seed\d+$", "", regex=True)

    metrics = [c for c in METRICS if c in df.columns]
    g = df.groupby("run", dropna=False)
    out = g[metrics].mean().round(4)
    out["n_seeds"] = g.size()  # sanity: should be 5; drop this line if unwanted
    out = out.reset_index().sort_values("run")

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} configs (from {len(df)} runs) to {args.output}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
