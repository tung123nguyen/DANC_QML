"""Main experiment entry point.

Usage:
    python -m src.experiment configs/qnn/q2_s1_n250_seed0.yaml
    python -m src.experiment configs/qnn/q2_s1_n250_seed0.yaml --smoke-test
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import yaml

from src.utils.seeding import set_all_seeds
from src.utils.io import get_results_root, get_git_commit, save_json
from src.utils.logging import setup_logger
from src.data.scenarios import build_scenario
from src.evaluation.metrics import compute_train_test_metrics


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


<<<<<<< HEAD
def make_run_dir(config: dict, smoke_test: bool = False) -> Path:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    prefix = "SMOKE_" if smoke_test else ""
    run_dir = get_results_root() / f"{prefix}{timestamp}_{name}_seed{seed}"
=======
def make_run_dir(config: dict) -> Path:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    run_dir = get_results_root() / f"{timestamp}_{name}_seed{seed}"
>>>>>>> c5148461b61d1f2474d7530f198bde2c24524a35
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def already_done(config: dict) -> bool:
<<<<<<< HEAD
    """Check if this exact experiment has been run before (REAL runs only).

    Smoke test results are excluded from this check - they have a SMOKE_
    prefix on their folder name and don't count as "done".
    """
=======
    """Check if this exact experiment has been run before."""
>>>>>>> c5148461b61d1f2474d7530f198bde2c24524a35
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    pattern = f"*_{name}_seed{seed}"
    for d in get_results_root().glob(pattern):
<<<<<<< HEAD
        if d.name.startswith("SMOKE_"):
            continue  # ignore smoke results
=======
>>>>>>> c5148461b61d1f2474d7530f198bde2c24524a35
        if (d / "metrics.json").exists():
            return True
    return False


def run_classical(config: dict, data: dict) -> dict:
    from src.models.classical import build_classical
    from src.training.classical_trainer import train_classical

    model = build_classical(config["model"]["name"], seed=config["experiment"]["seed"])
    train_log = train_classical(model, data["X_train"], data["y_train"])
    trained_model = train_log["model"]

    y_train_pred = trained_model.predict(data["X_train"])
    y_test_pred = trained_model.predict(data["X_test"])
    try:
        y_train_prob = trained_model.predict_proba(data["X_train"])[:, 1]
        y_test_prob = trained_model.predict_proba(data["X_test"])[:, 1]
    except (AttributeError, NotImplementedError):
        y_train_prob = y_test_prob = None

    metrics = compute_train_test_metrics(
        data["y_train"], y_train_pred, y_train_prob,
        data["y_test"], y_test_pred, y_test_prob,
    )
    metrics["train_time_seconds"] = train_log["train_time_seconds"]
    metrics["n_params"] = train_log["n_params"]
    return {"metrics": metrics, "history": train_log["history"]}


def run_qnn(config: dict, data: dict) -> dict:
    from src.models.qnn import QNNConfig, build_qnn
    from src.training.qnn_trainer import train_qnn, qnn_predict

    qnn_cfg = QNNConfig(
        encoding=config["model"]["encoding"],
        ansatz=config["model"]["ansatz"],
        n_qubits=config["model"]["n_qubits"],
        depth=config["model"]["depth"],
        device_name=config["model"].get("device", "default.qubit"),
    )
    qnn = build_qnn(qnn_cfg)

    train_log = train_qnn(
        qnn,
        data["X_train"], data["y_train"],
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        log_gradients=config["evaluation"].get("log_gradients", True),
        seed=config["experiment"]["seed"],
    )
    params = train_log["params"]

    y_train_pred, y_train_prob = qnn_predict(qnn, params, data["X_train"])
    y_test_pred, y_test_prob = qnn_predict(qnn, params, data["X_test"])

    metrics = compute_train_test_metrics(
        data["y_train"], y_train_pred, y_train_prob,
        data["y_test"], y_test_pred, y_test_prob,
    )
    metrics["train_time_seconds"] = train_log["train_time_seconds"]
    metrics["n_params"] = train_log["n_params"]
    return {"metrics": metrics, "history": train_log["history"]}


def run_experiment(config_path: Path, smoke_test: bool = False) -> dict:
    """Run a single experiment from a config file."""
    config = load_config(config_path)

    if smoke_test:
        config["data"]["train_samples_per_class"] = 20
        config["data"]["test_samples_per_class"] = 20
        if config["model"]["type"] == "qnn":
            config["training"]["epochs"] = 2

    if not smoke_test and already_done(config):
        print(f"SKIP {config['experiment']['name']} (already done)")
        return {}

<<<<<<< HEAD
    run_dir = make_run_dir(config, smoke_test=smoke_test)
=======
    run_dir = make_run_dir(config)
>>>>>>> c5148461b61d1f2474d7530f198bde2c24524a35
    logger = setup_logger("experiment", log_file=run_dir / "run.log")
    logger.info("Starting run: %s", run_dir.name)
    logger.info("Config: %s", config_path)

    set_all_seeds(config["experiment"]["seed"])

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    meta = {
        "git_commit": get_git_commit(),
        "config_path": str(config_path),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "smoke_test": smoke_test,
    }
    save_json(meta, run_dir / "meta.json")

    # Build data with new schema: explicit train_per_class and test_per_class
    data = build_scenario(
        scenario=config["data"]["scenario"],
        train_per_class=config["data"]["train_samples_per_class"],
        test_per_class=config["data"]["test_samples_per_class"],
        n_features=config["data"]["n_features"],
        seed=config["experiment"]["seed"],
    )
    logger.info(
        "Data ready: train=%s, test=%s",
        data["X_train"].shape, data["X_test"].shape,
    )

    if config["model"]["type"] == "classical":
        result = run_classical(config, data)
    elif config["model"]["type"] == "qnn":
        result = run_qnn(config, data)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    save_json(result["metrics"], run_dir / "metrics.json")
    save_json(result["history"], run_dir / "history.json")

    meta["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["test_f1"] = result["metrics"].get("test_f1")
    save_json(meta, run_dir / "meta.json")

    logger.info(
        "Done. test_f1=%.4f, time=%.1fs",
        result["metrics"]["test_f1"],
        result["metrics"]["train_time_seconds"],
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_experiment(Path(args.config), smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
