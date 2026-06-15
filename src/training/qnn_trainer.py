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
) -> dict:
    """Train a QNN with mini-batch Adam."""
    if not PENNYLANE_AVAILABLE:
        raise ImportError("Install pennylane: pip install pennylane")

    circuit = qnn["circuit"]
    n_params = qnn["n_params"]
    n_quantum = qnn["n_quantum_params"]
    n_qubits = qnn["n_qubits"]

    rng = np.random.default_rng(seed)
    init = rng.uniform(-0.1, 0.1, n_params)
    init[-1] = 0.0  # linear-head bias b -> logit ~ 0 -> p ~ 0.5 at init
    params = pnp.array(init, requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

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
                _circuit_probs(params, circuit, X_train, n_quantum, n_qubits), dtype=float
            )
            train_acc = float(((probs > 0.5).astype(int) == y_train).mean())
            row["train_acc"] = train_acc

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
    n_quantum = qnn["n_quantum_params"]
    n_qubits = qnn["n_qubits"]
    probs = np.asarray(
        _circuit_probs(params, circuit, X, n_quantum, n_qubits), dtype=float
    )
    preds = (probs > 0.5).astype(int)
    return preds, probs
