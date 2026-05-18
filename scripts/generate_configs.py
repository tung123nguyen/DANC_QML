"""Generate all YAML config files for the experiment matrix.

Edit the SWEEP block below to change what gets run.
Run: python scripts/generate_configs.py
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import yaml

CONFIG_ROOT = Path(__file__).parent.parent / "configs"

# =========================================================================
# SWEEP DEFINITIONS
# =========================================================================

SCENARIOS = ["S1", "S3"]
TRAIN_SAMPLE_SIZES = [100, 250, 500]   # samples per class IN TRAIN SET
TEST_SAMPLE_SIZE = 500                  # fixed test size per class
N_FEATURES = 4
SEEDS = [0, 1, 2]

CLASSICAL_MODELS = ["lr", "svm", "rf", "mlp"]

# QNN models: (id, encoding, ansatz)
QNN_MODELS = [
    ("q2", "angle", "basic_entangler"),
    ("q3", "angle", "strongly_entangling"),
    ("q4", "reuploading", "basic_entangler"),
]
QNN_DEPTH = 2
QNN_EPOCHS = 80
QNN_LR = 0.05
QNN_BATCH = 32
QNN_DEVICE = "default.qubit"  # change to "lightning.gpu" on Colab GPU


# =========================================================================
# CONFIG BUILDERS
# =========================================================================

def build_classical_config(model_name, scenario, train_n, seed):
    name = f"{model_name}_{scenario.lower()}_n{train_n}_f{N_FEATURES}"
    return {
        "experiment": {"name": name, "seed": seed},
        "data": {
            "scenario": scenario,
            "train_samples_per_class": train_n,
            "test_samples_per_class": TEST_SAMPLE_SIZE,
            "n_features": N_FEATURES,
        },
        "model": {"type": "classical", "name": model_name},
        "training": {},
        "evaluation": {"log_gradients": False},
    }


def build_qnn_config(qnn_id, encoding, ansatz, scenario, train_n, seed):
    name = (
        f"{qnn_id}_{encoding}_{ansatz}_{scenario.lower()}"
        f"_n{train_n}_f{N_FEATURES}_d{QNN_DEPTH}"
    )
    return {
        "experiment": {"name": name, "seed": seed},
        "data": {
            "scenario": scenario,
            "train_samples_per_class": train_n,
            "test_samples_per_class": TEST_SAMPLE_SIZE,
            "n_features": N_FEATURES,
        },
        "model": {
            "type": "qnn",
            "encoding": encoding,
            "ansatz": ansatz,
            "n_qubits": N_FEATURES,
            "depth": QNN_DEPTH,
            "device": QNN_DEVICE,
        },
        "training": {
            "epochs": QNN_EPOCHS,
            "learning_rate": QNN_LR,
            "batch_size": QNN_BATCH,
        },
        "evaluation": {"log_gradients": True},
    }


def write_config(config, subdir):
    out_dir = CONFIG_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    path = out_dir / f"{name}_seed{seed}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    return path


def main():
    n_classical = n_qnn = 0

    for model_name, scenario, train_n, seed in product(
        CLASSICAL_MODELS, SCENARIOS, TRAIN_SAMPLE_SIZES, SEEDS
    ):
        cfg = build_classical_config(model_name, scenario, train_n, seed)
        write_config(cfg, "classical")
        n_classical += 1

    for (qnn_id, enc, ansatz), scenario, train_n, seed in product(
        QNN_MODELS, SCENARIOS, TRAIN_SAMPLE_SIZES, SEEDS
    ):
        cfg = build_qnn_config(qnn_id, enc, ansatz, scenario, train_n, seed)
        write_config(cfg, "qnn")
        n_qnn += 1

    print(f"Generated {n_classical} classical configs")
    print(f"Generated {n_qnn} QNN configs")
    print(f"Total: {n_classical + n_qnn} experiments")
    print(f"\nTest set is FIXED at {TEST_SAMPLE_SIZE} samples/class for all configs.")
    print(f"Train sizes vary: {TRAIN_SAMPLE_SIZES}")


if __name__ == "__main__":
    main()
