"""QNN model factory: encoding x ansatz.

Builds a PennyLane variational circuit + a thin sklearn-compatible wrapper
so the experiment.py loop can treat classical and QNN models the same way.

Variants:
    encoding: 'angle' | 'reuploading'
    ansatz:   'basic_entangler' | 'strongly_entangling' | 'rotation_only'

PARAM SHAPE NOTES:
    - BasicEntangler: (depth, n_qubits)            - 1 rotation/qubit
    - StronglyEntangling: (depth, n_qubits, 3)     - 3 rotations/qubit
    - Rotation-only: (depth, n_qubits, 3)          - 3 rotations, NO CNOT

Re-uploading repeats (encoding + ansatz) `depth` times. With angle encoding
only the ansatz is repeated.

ENCODING NORMALISATION (angle_clip):
    Features arrive StandardScaler'd (mean 0, std 1), so their range is ~[-3, 3]
    with heavier outliers. Feeding them straight into RY angles is unsafe: RY is
    2*pi periodic, so a value like 30 wraps around to ~0 and collapses distinct
    inputs. We therefore clip to [-angle_clip, +angle_clip] sigma and linearly
    scale so +/-angle_clip*sigma maps to +/-pi, keeping every angle inside
    [-pi, pi]. Default angle_clip=3.0 keeps ~99.7% of Gaussian data in the
    linear region; set it to None/0 to disable (reproduces the old raw encoding).

Output: single Pauli-Z expectation on qubit 0, mapped to [0, 1] by (1+z)/2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


def _scale_to_angle(x, angle_clip):
    """Map StandardScaler output (in sigma units) to angles in [-pi, pi].

    Clip to [-angle_clip, angle_clip] sigma, then scale linearly so that
    +/-angle_clip maps to +/-pi. A falsy angle_clip (None or 0) disables the
    transform and returns x unchanged (reproduces the old raw encoding).
    """
    if not angle_clip:
        return x
    # x is data (never differentiated), so plain numpy is safe here and keeps
    # this helper testable without a pennylane import.
    return np.clip(x, -angle_clip, angle_clip) * (np.pi / angle_clip)


def _angle_embed(x, n_qubits, angle_clip=3.0):
    x = _scale_to_angle(x, angle_clip)
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")


def _apply_ansatz(params, ansatz: str, n_qubits: int):
    """One layer of ansatz."""
    if ansatz == "basic_entangler":
        # params shape: (n_qubits,) for one layer
        qml.BasicEntanglerLayers(
            params.reshape(1, n_qubits), wires=range(n_qubits)
        )
    elif ansatz == "strongly_entangling":
        # params shape: (n_qubits, 3) for one layer
        qml.StronglyEntanglingLayers(
            params.reshape(1, n_qubits, 3), wires=range(n_qubits)
        )
    elif ansatz == "rotation_only":
        # 3 rotations per qubit, no entanglement
        for q in range(n_qubits):
            qml.RX(params[q * 3 + 0], wires=q)
            qml.RY(params[q * 3 + 1], wires=q)
            qml.RZ(params[q * 3 + 2], wires=q)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


def _params_per_layer(ansatz: str, n_qubits: int) -> int:
    """Number of params for ONE layer of the given ansatz."""
    if ansatz == "basic_entangler":
        return n_qubits
    if ansatz == "strongly_entangling":
        return n_qubits * 3
    if ansatz == "rotation_only":
        return n_qubits * 3
    raise ValueError(ansatz)


def build_qnode(encoding: str, ansatz: str, n_qubits: int, depth: int, device_name: str = "default.qubit", angle_clip: float = 3.0):
    """Build the QNode and return (qnode, param_shape).

    The returned qnode takes (params_flat, x) and returns a Pauli-Z expectation.
    angle_clip controls the [-pi, pi] encoding normalisation (see module docstring).
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("Install pennylane: pip install pennylane")

    dev = qml.device(device_name, wires=n_qubits)
    ppl = _params_per_layer(ansatz, n_qubits)

    if encoding == "angle":
        # Encode once, then `depth` ansatz layers
        total_params = depth * ppl

        @qml.qnode(dev, interface="autograd")
        def circuit(params, x):
            _angle_embed(x, n_qubits, angle_clip)
            for d in range(depth):
                layer_params = params[d * ppl: (d + 1) * ppl]
                _apply_ansatz(layer_params, ansatz, n_qubits)
            return qml.expval(qml.PauliZ(0))

    elif encoding == "reuploading":
        # Repeat (encode + ansatz) `depth` times
        total_params = depth * ppl

        @qml.qnode(dev, interface="autograd")
        def circuit(params, x):
            for d in range(depth):
                _angle_embed(x, n_qubits, angle_clip)
                layer_params = params[d * ppl: (d + 1) * ppl]
                _apply_ansatz(layer_params, ansatz, n_qubits)
            return qml.expval(qml.PauliZ(0))

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return circuit, (total_params,)


@dataclass
class QNNConfig:
    encoding: str
    ansatz: str
    n_qubits: int
    depth: int
    device_name: str = "default.qubit"
    angle_clip: float = 3.0


def build_qnn(config: QNNConfig):
    """Build a QNN (circuit + initial params). The trainer wraps this."""
    circuit, param_shape = build_qnode(
        config.encoding, config.ansatz, config.n_qubits, config.depth,
        config.device_name, config.angle_clip,
    )
    return {
        "circuit": circuit,
        "param_shape": param_shape,
        "n_params": int(np.prod(param_shape)),
        "config": config,
    }
