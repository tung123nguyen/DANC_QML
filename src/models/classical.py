"""Classical ML model factory.

Returns sklearn-compatible models with .fit(), .predict(), .predict_proba().
All hyperparameters are 'reasonable defaults' - we don't grid search.
The point is comparison with QNN under same conditions, not winning a kaggle.
"""
from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def build_classical(model_type: str, seed: int = 0):
    """Build a classical model by name.

    Args:
        model_type: One of 'lr', 'svm', 'rf', 'mlp'
        seed: Random seed for stochastic models

    Returns:
        Unfit sklearn classifier
    """
    builders = {
        "lr": _build_lr,
        "svm": _build_svm,
        "rf": _build_rf,
        "mlp": _build_mlp,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown classical model: {model_type}")
    return builders[model_type](seed)


def _build_lr(seed: int):
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        C=1.0,
    )


def _build_svm(seed: int):
    return SVC(
        kernel="rbf",
        probability=True,  # needed for ROC-AUC
        random_state=seed,
        C=1.0,
        gamma="scale",
    )


def _build_rf(seed: int):
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
    )


def _build_mlp(seed: int):
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
    )
