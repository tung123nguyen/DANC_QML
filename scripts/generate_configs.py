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
SEEDS = [0, 1, 2, 3, 4]

CLASSICAL_MODELS = ["rf", "mlp", "mlp_tiny"]

# QNN models: (id, encoding, ansatz)
QNN_MODELS = [
    ("q2", "angle", "basic_entangler"),
    ("q3", "angle", "strongly_entangling"),
    ("q4", "reuploading", "basic_entangler"),
]
QNN_DEPTH = 4
QNN_EPOCHS = 80
QNN_LR = 0.05
QNN_BATCH = 32
QNN_DEVICE = "lightning.qubit"  # change to "lightning.gpu" on Colab GPU
QNN_ANGLE_CLIP = 3.0  # clip StandardScaler features to +/-3 sigma, scale to [-pi, pi]
QNN_LOG_GRADIENTS = False  # gradient stats OFF for the sweep (we compare F1/acc, not grads)

# QNN experiment variants (readout / encoding tweaks via extra model fields).
# Each is (name_suffix, extra model fields). "" = baseline (z readout, fixed encoding).
QNN_VARIANTS = [
    ("", {}),                                              # baseline: <Z_i>, fixed encoding
    ("_zz", {"readout": "z+zz"}),                          # readout + <Z_iZ_j>        (10-dim)
    ("_probs", {"readout": "probs", "readout_wires": 2}),  # readout P over qubits 0,1  (4-dim)
    ("_tenc", {"trainable_encoding": True}),               # affine encoding RY(a*x+b), readout z
]

# A small diagnostic subset re-enables gradient logging for barren-plateau analysis.
# Written to configs/qnn_diag/ with a '_diag' name suffix so it never collides with
# the sweep's already_done() bookkeeping.
DIAG_SCENARIO = "S1"
DIAG_TRAIN_N = 250


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


def build_qnn_config(qnn_id, encoding, ansatz, scenario, train_n, seed,
                     log_gradients=QNN_LOG_GRADIENTS, name_suffix="", readout_model=None):
    name = (
        f"{qnn_id}_{encoding}_{ansatz}_{scenario.lower()}"
        f"_n{train_n}_f{N_FEATURES}_d{QNN_DEPTH}{name_suffix}"
    )
    model = {
        "type": "qnn",
        "encoding": encoding,
        "ansatz": ansatz,
        "n_qubits": N_FEATURES,
        "depth": QNN_DEPTH,
        "device": QNN_DEVICE,
        "angle_clip": QNN_ANGLE_CLIP,
    }
    if readout_model:
        model.update(readout_model)  # e.g. {"readout": "z+zz"} or probs fields
    return {
        "experiment": {"name": name, "seed": seed},
        "data": {
            "scenario": scenario,
            "train_samples_per_class": train_n,
            "test_samples_per_class": TEST_SAMPLE_SIZE,
            "n_features": N_FEATURES,
        },
        "model": model,
        "training": {
            "epochs": QNN_EPOCHS,
            "learning_rate": QNN_LR,
            "batch_size": QNN_BATCH,
        },
        "evaluation": {"log_gradients": log_gradients},
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
        for suffix, rmodel in QNN_VARIANTS:
            cfg = build_qnn_config(qnn_id, enc, ansatz, scenario, train_n, seed,
                                   name_suffix=suffix, readout_model=rmodel)
            write_config(cfg, "qnn")
            n_qnn += 1

    # Diagnostic subset: a few QNN runs WITH gradient logging on (barren-plateau).
    n_qnn_diag = 0
    for (qnn_id, enc, ansatz), seed in product(QNN_MODELS, SEEDS):
        cfg = build_qnn_config(
            qnn_id, enc, ansatz, DIAG_SCENARIO, DIAG_TRAIN_N, seed,
            log_gradients=True,
        )
        cfg["experiment"]["name"] += "_diag"
        write_config(cfg, "qnn_diag")
        n_qnn_diag += 1

    # Entanglement ablation (targeted, NOT a full cross-product): angle encoding
    # with the rotation_only ansatz at S1 / N=500 only. rotation_only has the same
    # 3n params/layer as strongly_entangling (q3) but drops the CNOTs, so comparing
    # it against q3 at this cell isolates entanglement's contribution (thesis 3.6.3).
    n_ablation = 0
    for seed in SEEDS:
        cfg = build_qnn_config("q1", "angle", "rotation_only", "S1", 500, seed)
        write_config(cfg, "qnn")
        n_ablation += 1

    print(f"Generated {n_classical} classical configs")
    print(f"Generated {n_qnn} QNN configs "
          f"({len(QNN_VARIANTS)} variants: {[s or 'base' for s, _ in QNN_VARIANTS]})")
    print(f"Generated {n_qnn_diag} QNN diagnostic configs (log_gradients=True)")
    print(f"Generated {n_ablation} QNN rotation_only ablation configs (S1/N500)")
    print(f"Total: {n_classical + n_qnn + n_qnn_diag + n_ablation} experiments")
    print(f"\nTest set is FIXED at {TEST_SAMPLE_SIZE} samples/class for all configs.")
    print(f"Train sizes vary: {TRAIN_SAMPLE_SIZES}")


if __name__ == "__main__":
    main()
