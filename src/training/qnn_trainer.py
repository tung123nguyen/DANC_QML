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


def _circuit_probs(params, circuit, X, n_quantum, n_qubits):
    """Run circuit + linear head on a BATCH -> probabilities, shape (len(X),).

    params is the flat vector [quantum_params | w (n_qubits) | b (1)]. X has
    shape (B, n_qubits). PennyLane parameter broadcasting evaluates all B
    samples in ONE circuit call (no Python loop): the qnode returns one (B,)
    expectation per qubit, the linear head maps them to B logits and a sigmoid
    to B probabilities. Written with pennylane numpy (pnp) so gradients flow to
    BOTH quantum params and head (w, b).
    """
    q_params = params[:n_quantum]
    w = params[n_quantum:n_quantum + n_qubits]
    b = params[n_quantum + n_qubits]
    z = pnp.stack(circuit(q_params, X))      # (n_qubits, B) <Z_i> in [-1, 1]
    logit = pnp.dot(w, z) + b                # (B,)
    return 1.0 / (1.0 + pnp.exp(-logit))     # sigmoid -> (B,)


def _bce_loss(params, circuit, X, y, n_quantum, n_qubits):
    """Mean binary cross-entropy over a batch (vectorised over samples)."""
    p = _circuit_probs(params, circuit, X, n_quantum, n_qubits)
    p = pnp.clip(p, 1e-7, 1 - 1e-7)
    return pnp.mean(-y * pnp.log(p) - (1 - y) * pnp.log(1 - p))


def train_qnn(
    qnn: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.05,
    batch_size: int = 32,
    log_gradients: bool = True,
    seed: int = 0,
    early_stopping: bool = True,
    val_fraction: float = 0.2,
    patience: int = 15,
    min_delta: float = 0.0,
) -> dict:
    """Train a QNN with mini-batch Adam.

    Early stopping (on by default): carve a stratified ``val_fraction`` slice out
    of (X_train, y_train), train on the rest, and monitor validation BCE each
    epoch. If it does not improve by ``min_delta`` for ``patience`` epochs, stop
    and RESTORE the best-val params (so we never return a collapsed model). The
    held-out test set is untouched. Set early_stopping=False to train the full
    ``epochs`` on all of X_train (old behaviour).
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("Install pennylane: pip install pennylane")

    circuit = qnn["circuit"]
    n_params = qnn["n_params"]
    n_quantum = qnn["n_quantum_params"]
    n_qubits = qnn["n_qubits"]

    n_scale = qnn.get("n_scale_params", 0)

    rng = np.random.default_rng(seed)
    init = rng.uniform(-0.1, 0.1, n_params)
    if n_scale:
        init[:n_scale] = 1.0  # trainable input scale starts at 1 (== fixed encoding)
    init[-1] = 0.0  # linear-head bias b -> logit ~ 0 -> p ~ 0.5 at init
    params = pnp.array(init, requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    # Early stopping: hold out a stratified validation slice; train on the rest.
    use_es = bool(early_stopping) and val_fraction and 0.0 < val_fraction < 1.0
    if use_es:
        from sklearn.model_selection import train_test_split
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction,
            stratify=y_train, random_state=seed,
        )
    else:
        X_fit, y_fit, X_val, y_val = X_train, y_train, None, None

    best_val = np.inf
    best_params = None
    epochs_no_improve = 0
    stopped_epoch = None

    n = X_fit.shape[0]
    history = []
    t0 = time.time()

    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuf, y_shuf = X_fit[perm], y_fit[perm]

        epoch_loss = 0.0
        batch_grad_norms = []
        batch_grad_vars = []
        batch_grad_meanabs = []
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            obj = lambda p: _bce_loss(p, circuit, xb, yb, n_quantum, n_qubits)

            if log_gradients:
                # Compute the gradient ONCE and reuse it for BOTH the per-batch
                # stats and the Adam update. step_and_cost internally does
                # compute_grad + apply_grad; splitting it open lets us log the
                # gradient without triggering a second (redundant) evaluation.
                grad, batch_loss = optimizer.compute_grad(obj, (params,), {})
                g = np.asarray(grad[0])
                batch_grad_norms.append(float(np.linalg.norm(g)))
                batch_grad_vars.append(float(np.var(g)))
                batch_grad_meanabs.append(float(np.mean(np.abs(g))))
                params = optimizer.apply_grad(grad, (params,))[0]
                if batch_loss is None:
                    batch_loss = obj(params)
            else:
                params, batch_loss = optimizer.step_and_cost(obj, params)

            epoch_loss += float(batch_loss)
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        row = {"epoch": epoch, "loss": float(epoch_loss)}

        # Train accuracy is a full-dataset forward pass, so only compute it on
        # the epochs we actually report (every 10 + last). The per-epoch loss
        # already tracks convergence cheaply (it comes free from the batches).
        report_epoch = (epoch % 10 == 0) or (epoch == epochs - 1)
        if report_epoch:
            probs = np.asarray(
                _circuit_probs(params, circuit, X_fit, n_quantum, n_qubits), dtype=float
            )
            train_acc = float(((probs > 0.5).astype(int) == y_fit).mean())
            row["train_acc"] = train_acc

        # Early stopping: track validation BCE every epoch (cheap, vectorised),
        # keep the best-val params, and count epochs without improvement.
        if use_es:
            val_loss = float(_bce_loss(params, circuit, X_val, y_val, n_quantum, n_qubits))
            row["val_loss"] = val_loss
            if report_epoch:
                vprobs = np.asarray(
                    _circuit_probs(params, circuit, X_val, n_quantum, n_qubits), dtype=float
                )
                row["val_acc"] = float(((vprobs > 0.5).astype(int) == y_val).mean())
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_params = np.asarray(params).copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if log_gradients and batch_grad_norms:
            # MEAN over batches (not just last batch)
            row["grad_norm_mean"] = float(np.mean(batch_grad_norms))
            row["grad_norm_std"] = float(np.std(batch_grad_norms))
            row["grad_var_mean"] = float(np.mean(batch_grad_vars))
            row["grad_meanabs_mean"] = float(np.mean(batch_grad_meanabs))
            # Keep last batch grad norm too for legacy reference
            row["grad_norm_last_batch"] = batch_grad_norms[-1]

        history.append(row)

        if report_epoch:
            log.info(
                "epoch=%d loss=%.4f acc=%.3f%s",
                epoch, epoch_loss, train_acc,
                f" grad_norm_mean={row.get('grad_norm_mean', 0):.4f}" if log_gradients else "",
            )

        if use_es and epochs_no_improve >= patience:
            stopped_epoch = epoch
            log.info(
                "early stop at epoch %d (best val_loss=%.4f); restoring best params",
                epoch, best_val,
            )
            break

    # Restore the best-val params (never return a worse / collapsed model).
    if use_es and best_params is not None:
        params = pnp.array(best_params, requires_grad=True)

    duration = time.time() - t0

    return {
        "params": np.asarray(params),
        "history": history,
        "train_time_seconds": duration,
        "n_params": n_params,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": (best_val if use_es else None),
    }


def qnn_predict(qnn: dict, params, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities."""
    circuit = qnn["circuit"]
    n_quantum = qnn["n_quantum_params"]
    n_qubits = qnn["n_qubits"]
    probs = np.asarray(
        _circuit_probs(params, circuit, X, n_quantum, n_qubits), dtype=float
    )
    preds = (probs > 0.5).astype(int)
    return preds, probs
