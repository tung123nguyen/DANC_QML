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

Output: single Pauli-Z expectation on qubit 0, mapped to [0, 1] by (1+z)/2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


def _angle_embed(x, n_qubits):
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


def build_qnode(encoding: str, ansatz: str, n_qubits: int, depth: int, device_name: str = "default.qubit"):
    """Build the QNode and return (qnode, param_shape).

    The returned qnode takes (params_flat, x) and returns a Pauli-Z expectation.
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
            _angle_embed(x, n_qubits)
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
                _angle_embed(x, n_qubits)
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


def build_qnn(config: QNNConfig):
    """Build a QNN (circuit + initial params). The trainer wraps this."""
    circuit, param_shape = build_qnode(
        config.encoding, config.ansatz, config.n_qubits, config.depth, config.device_name
    )
    return {
        "circuit": circuit,
        "param_shape": param_shape,
        "n_params": int(np.prod(param_shape)),
        "config": config,
    }
