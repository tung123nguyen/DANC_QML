"""Classical ML model factory.

Returns sklearn-compatible models with .fit(), .predict(), .predict_proba().
All hyperparameters are 'reasonable defaults' - we don't grid search.
The point is comparison with QNN under same conditions, not winning a kaggle.
"""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def build_classical(model_type: str, seed: int = 0):
    """Build a classical model by name.

    Args:
        model_type: One of 'rf', 'mlp', 'mlp_tiny'
        seed: Random seed for stochastic models

    Returns:
        Unfit sklearn classifier
    """
    builders = {
        "rf": _build_rf,
        "mlp": _build_mlp,
        "mlp_tiny": _build_mlp_tiny,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown classical model: {model_type}")
    return builders[model_type](seed)


def _build_rf(seed: int):
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
    )


def _build_mlp(seed: int):
    # early_stopping=False: with early_stopping=True sklearn carves a
    # validation split out of the already-tiny training fold and stops
    # before the net learns anything, which collapses MLP to all-benign
    # (F1=0.0) at small N. Disabling it keeps MLP comparable to the other
    # baselines under the small-sample budget.
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=seed,
        early_stopping=False,
    )


def _build_mlp_tiny(seed: int):
    # Parameter-MATCHED classical baseline. With 4 input features, a single
    # hidden layer of width h has 6h + 1 trainable params (4h + h weights +
    # h + 1 biases). h=4 -> 25 params, putting this MLP in the SAME range as
    # the QNNs (21-53 params) instead of the 705 of the default MLP. This is
    # the fair "same-capacity" counterpart to the QNN: it answers whether a
    # classical net with a comparable parameter budget matches the quantum
    # model, isolating any quantum benefit from the raw capacity advantage.
    # early_stopping=False for the same small-sample reason as _build_mlp.
    return MLPClassifier(
        hidden_layer_sizes=(4,),
        max_iter=500,
        random_state=seed,
        early_stopping=False,
    )