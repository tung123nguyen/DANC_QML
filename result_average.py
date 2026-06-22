"""Average result.py output over the 5 seeds of each config.

Reads result.csv (produced by result.py), strips the _seed<N> suffix to get
the config name, and reports the mean of each metric across its seeds.
"""
from __future__ import annotations
import argparse
import pandas as pd

METRICS = ["train_accuracy", "train_f1", "test_accuracy", "test_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="result.csv")
    parser.add_argument("--output", default="result_average.csv")
    parser.add_argument(
        "--filter", default=None,
        help="only keep configs whose name contains this substring (e.g. n500)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # "q2_..._seed3" -> "q2_..." so all seeds of one config share a key.
    df["config"] = df["run"].str.replace(r"_seed\d+$", "", regex=True)

    if args.filter:
        df = df[df["config"].str.contains(args.filter)]

    avg = (
        df.groupby("config")[METRICS]
        .mean()
        .round(6)
        .reset_index()
    )
    avg.insert(1, "n_seeds", df.groupby("config").size().values)

    avg.to_csv(args.output, index=False)
    print(f"Wrote {len(avg)} configs to {args.output}")
    print(avg.to_string(index=False))


if __name__ == "__main__":
    main()
