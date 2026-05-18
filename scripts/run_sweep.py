"""Run a batch of experiments.

Idempotent: skips experiments whose metrics.json already exists.
Safe to interrupt and resume.

Usage:
    python scripts/run_sweep.py                          # all configs
    python scripts/run_sweep.py --filter classical       # only classical
    python scripts/run_sweep.py --filter qnn             # only QNN
    python scripts/run_sweep.py --filter "s1_n250"       # name substring
"""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

# Allow imports from src/ when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment import run_experiment, already_done, load_config

CONFIG_ROOT = Path(__file__).parent.parent / "configs"


def collect_configs(filter_str: str | None) -> list[Path]:
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    if filter_str:
        paths = [p for p in paths if filter_str in str(p)]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run configs whose path contains this substring")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run without running")
    args = parser.parse_args()

    configs = collect_configs(args.filter)
    if not configs:
        print(f"No configs found (filter={args.filter!r})")
        return

    print(f"Found {len(configs)} configs to consider")
    if args.dry_run:
        for p in configs:
            done = already_done(load_config(p))
            print(f"  {'SKIP' if done else 'RUN '} {p}")
        return

    n_run = n_skip = n_fail = 0
    for i, path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {path.name}")
        try:
            config = load_config(path)
            if already_done(config):
                print("  SKIP (already done)")
                n_skip += 1
                continue
            run_experiment(path)
            n_run += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            n_fail += 1
            continue

    print(f"\nSweep complete: ran={n_run}, skipped={n_skip}, failed={n_fail}")


if __name__ == "__main__":
    main()
