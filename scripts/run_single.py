"""Run a single experiment from a config file.

Usage:
    python scripts/run_single.py configs/classical/rf_s1_n250_f4_seed0.yaml
    python scripts/run_single.py configs/qnn/q2_angle_basic_s1_n250_f4_d2_seed0.yaml --smoke-test
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_experiment(Path(args.config), smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
