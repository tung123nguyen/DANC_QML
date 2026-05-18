"""QNN training loop.

Trains a variational circuit with Adam optimizer. Logs per-epoch:
- training loss (BCE)
- training accuracy
- gradient norm (mean over batches in the epoch)
- gradient variance (mean over batches in the epoch)
- gradient mean-abs

Gradient stats are essential for the trainability analysis (detecting
barren plateaus, comparing ansatz/encoding expressivity).

Important: gradient stats are MEAN OVER ALL BATCHES of the epoch, not just
the final batch. This gives a smoother signal and better reflects how the
optimizer "sees" the loss landscape.

Uses PennyLane's autograd interface. For speed on Colab GPU, switch the
qnode in qnn.py to interface='jax' and use 'lightning.gpu' device.
"""
from __future__ import annotations
import logging
import time
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

log = logging.getLogger(__name__)


def _expval_to_prob(z):
    """Map Pauli-Z expectation in [-1, 1] to probability in [0, 1]."""
    return (1.0 + z) / 2.0


def _bce_loss(params, circuit, X, y):
    """Binary cross-entropy on a batch."""
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        z = circuit(params, X[i])
        p = _expval_to_prob(z)
        p = pnp.clip(p, 1e-7, 1 - 1e-7)
        total = total + (-y[i] * pnp.log(p) - (1 - y[i]) * pnp.log(1 - p))
    return total / n


def train_qnn(
    qnn: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.05,
    batch_size: int = 32,
    log_gradients: bool = True,
    seed: int = 0,
) -> dict:
    """Train a QNN with mini-batch Adam."""
    if not PENNYLANE_AVAILABLE:
        raise ImportError("Install pennylane: pip install pennylane")

    circuit = qnn["circuit"]
    n_params = qnn["n_params"]

    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(-0.1, 0.1, n_params), requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    grad_fn = qml.grad(_bce_loss, argnum=0)

    n = X_train.shape[0]
    history = []
    t0 = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        epoch_loss = 0.0
        batch_grad_norms = []
        batch_grad_vars = []
        batch_grad_meanabs = []
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            # Gradient (separate from step so we can log per-batch stats)
            if log_gradients:
                g = np.asarray(grad_fn(params, circuit, xb, yb))
                batch_grad_norms.append(float(np.linalg.norm(g)))
                batch_grad_vars.append(float(np.var(g)))
                batch_grad_meanabs.append(float(np.mean(np.abs(g))))

            params, batch_loss = optimizer.step_and_cost(
                lambda p: _bce_loss(p, circuit, xb, yb), params
            )
            epoch_loss += float(batch_loss)
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Train accuracy at end of epoch
        preds = np.array([
            1 if _expval_to_prob(float(circuit(params, X_train[i]))) > 0.5 else 0
            for i in range(n)
        ])
        train_acc = float((preds == y_train).mean())

        row = {
            "epoch": epoch,
            "loss": float(epoch_loss),
            "train_acc": train_acc,
        }
        if log_gradients and batch_grad_norms:
            # MEAN over batches (not just last batch)
            row["grad_norm_mean"] = float(np.mean(batch_grad_norms))
            row["grad_norm_std"] = float(np.std(batch_grad_norms))
            row["grad_var_mean"] = float(np.mean(batch_grad_vars))
            row["grad_meanabs_mean"] = float(np.mean(batch_grad_meanabs))
            # Keep last batch grad norm too for legacy reference
            row["grad_norm_last_batch"] = batch_grad_norms[-1]

        history.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            log.info(
                "epoch=%d loss=%.4f acc=%.3f%s",
                epoch, epoch_loss, train_acc,
                f" grad_norm_mean={row.get('grad_norm_mean', 0):.4f}" if log_gradients else "",
            )

    duration = time.time() - t0

    return {
        "params": np.asarray(params),
        "history": history,
        "train_time_seconds": duration,
        "n_params": n_params,
    }


def qnn_predict(qnn: dict, params, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities."""
    circuit = qnn["circuit"]
    probs = np.array([
        _expval_to_prob(float(circuit(params, X[i]))) for i in range(X.shape[0])
    ])
    preds = (probs > 0.5).astype(int)
    return preds, probs
