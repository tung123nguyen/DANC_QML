"""End-to-end smoke test with synthetic CIC-IDS2017-shaped data.

Verifies the classical pipeline runs all the way from CSV → metrics.json.
Does NOT test PennyLane (run a real QNN config separately for that).

Run: python scripts/smoke_test.py
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_fake_csv(path: Path, n_per_class: int = 2000, n_features: int = 25):
    """Write a fake CIC-IDS2017-shaped CSV."""
    rng = np.random.default_rng(0)
    X_benign = rng.normal(0, 1, (n_per_class, n_features))
    X_attack = rng.normal(1.5, 1, (n_per_class, n_features))
    X = np.vstack([X_benign, X_attack])

    cols = [f" feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df[" Label"] = ["BENIGN"] * n_per_class + ["DDoS"] * n_per_class
    df["Flow ID"] = "x"
    df[" Source Port"] = rng.integers(1024, 65535, len(df))
    df[" Destination Port"] = rng.integers(80, 443, len(df))
    df[" Protocol"] = rng.integers(0, 17, len(df))
    df.to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        (tmp_path / "results").mkdir()

        os.environ["DATA_ROOT"] = str(tmp_path / "data")
        os.environ["RESULTS_ROOT"] = str(tmp_path / "results")

        from src.data.loader import SCENARIO_FILES
        fake_csv = tmp_path / "data" / "raw" / SCENARIO_FILES["friday_ddos"]
        make_fake_csv(fake_csv)
        print(f"Wrote fake CSV: {fake_csv.stat().st_size:,} bytes")

        from src.experiment import run_experiment
        import yaml

        cfg = {
            "experiment": {"name": "smoke", "seed": 0},
            "data": {
                "scenario": "S1",
                "train_samples_per_class": 100,
                "test_samples_per_class": 300,
                "n_features": 4,
            },
            "model": {"type": "classical", "name": "rf"},
            "training": {},
            "evaluation": {"log_gradients": False},
        }
        cfg_path = tmp_path / "smoke.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        result = run_experiment(cfg_path)
        print("\n=== METRICS ===")
        for k, v in result["metrics"].items():
            if not isinstance(v, list):
                print(f"  {k}: {v}")

        result_dirs = list((tmp_path / "results").iterdir())
        assert len(result_dirs) == 1
        rd = result_dirs[0]
        for must_have in ("config.yaml", "metrics.json", "meta.json", "history.json", "run.log"):
            assert (rd / must_have).exists(), f"Missing: {must_have}"

        # Verify the sample semantics are correct
        import json
        # 100 train/class * 2 classes = 200 train samples
        # 300 test/class * 2 classes = 600 test samples
        with open(rd / "metrics.json") as f:
            m = json.load(f)
        cm_test = m["test_confusion"]
        total_test = sum(sum(row) for row in cm_test)
        assert total_test == 600, f"Expected 600 test samples, got {total_test}"
        cm_train = m["train_confusion"]
        total_train = sum(sum(row) for row in cm_train)
        assert total_train == 200, f"Expected 200 train samples, got {total_train}"

        print(f"\nTrain: {total_train} samples (100/class * 2)")
        print(f"Test:  {total_test} samples (300/class * 2)")
        print(f"Files OK in {rd.name}/")
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
