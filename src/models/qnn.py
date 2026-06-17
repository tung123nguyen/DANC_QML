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

READOUT (linear head):
    The circuit measures <Z_i> on EVERY qubit, giving a vector z in [-1, 1]^n.
    A classical linear head turns it into one logit (logit = w . z + b) and a
    sigmoid maps that to a probability in [0, 1]. The head weights w (n_qubits)
    and bias b (1) are trained jointly with the quantum params: build_qnn lays
    them out as a single flat vector [quantum_params | w | b] and the trainer
    (qnn_trainer.py) applies the head. See n_quantum_params / n_head_params.
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


def _angle_embed(x, n_qubits, angle_clip=3.0, scale=None):
    x = _scale_to_angle(x, angle_clip)
    if scale is not None:
        # Trainable input scaling (Form A): a learnable per-qubit weight on the
        # encoding angle. RY(s_i * angle_i) lets the model pick each feature's
        # frequency instead of the fixed unit scale. s starts at 1.0, so a fresh
        # trainable-scale model behaves identically to the fixed encoding.
        x = x * scale
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


def _observables(n_qubits: int, readout: str):
    """Readout observables fed to the linear head (circuit depth unchanged).

    'z'    : single-qubit <Z_i>                         -> n_qubits values
    'z+zz' : <Z_i> plus all 2-qubit correlations <Z_iZ_j> -> n_qubits + C(n,2)
    """
    obs = [qml.PauliZ(i) for i in range(n_qubits)]
    if readout == "z+zz":
        obs += [qml.PauliZ(i) @ qml.PauliZ(j)
                for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif readout != "z":
        raise ValueError(f"Unknown readout: {readout}")
    return obs


def _n_observables(n_qubits: int, readout: str, readout_wires: int = 2) -> int:
    """Linear-head input width = number of readout values."""
    if readout == "probs":
        return 2 ** readout_wires
    if readout == "z+zz":
        return n_qubits + n_qubits * (n_qubits - 1) // 2
    return n_qubits


def _measure(n_qubits: int, readout: str, readout_wires: int = 2):
    """The qnode return: a list of <obs> expvals, or a basis-prob distribution.

    readout='probs' returns P over the computational basis of the first
    readout_wires qubits (2**readout_wires probabilities, summing to 1) -- a
    single qml.probs measurement, not a list of expvals.
    """
    if readout == "probs":
        return qml.probs(wires=range(readout_wires))
    return [qml.expval(o) for o in _observables(n_qubits, readout)]


def build_qnode(encoding: str, ansatz: str, n_qubits: int, depth: int, device_name: str = "lightning.qubit", angle_clip: float = 3.0, trainable_scale: bool = False, readout: str = "z", readout_wires: int = 2):
    """Build the QNode and return (qnode, param_shape).

    The returned qnode takes (circuit_params, x) and returns a list of readout
    expectations (the linear head lives in the trainer). circuit_params is laid
    out as [scale (n_qubits, if trainable_scale) | ansatz].
    angle_clip controls the [-pi, pi] encoding normalisation (see module docstring).
    trainable_scale prepends a learnable per-qubit input scale (Form A feature map).
    readout selects the observable set ('z' or 'z+zz', see _observables).
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("Install pennylane: pip install pennylane")

    dev = qml.device(device_name, wires=n_qubits)
    ppl = _params_per_layer(ansatz, n_qubits)
    n_scale = n_qubits if trainable_scale else 0
    total_params = n_scale + depth * ppl

    def _split(params):
        """Return (scale_or_None, ansatz_params) from the flat circuit params."""
        if trainable_scale:
            return params[:n_scale], params[n_scale:]
        return None, params

    if encoding == "angle":
        # Encode once, then `depth` ansatz layers
        @qml.qnode(dev, interface="autograd")
        def circuit(params, x):
            scale, aparams = _split(params)
            _angle_embed(x, n_qubits, angle_clip, scale)
            for d in range(depth):
                layer_params = aparams[d * ppl: (d + 1) * ppl]
                _apply_ansatz(layer_params, ansatz, n_qubits)
            return _measure(n_qubits, readout, readout_wires)

    elif encoding == "reuploading":
        # Repeat (encode + ansatz) `depth` times
        @qml.qnode(dev, interface="autograd")
        def circuit(params, x):
            scale, aparams = _split(params)
            for d in range(depth):
                _angle_embed(x, n_qubits, angle_clip, scale)
                layer_params = aparams[d * ppl: (d + 1) * ppl]
                _apply_ansatz(layer_params, ansatz, n_qubits)
            return _measure(n_qubits, readout, readout_wires)

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return circuit, (total_params,)


@dataclass
class QNNConfig:
    encoding: str
    ansatz: str
    n_qubits: int
    depth: int
    device_name: str = "lightning.qubit"
    angle_clip: float = 3.0
    trainable_scale: bool = False
    readout: str = "z"
    readout_wires: int = 2


def build_qnn(config: QNNConfig):
    """Build a QNN (circuit + initial params). The trainer wraps this.

    The trainable vector is laid out as
        [scale (n_qubits, if trainable_scale) | ansatz | w (n_qubits) | b (1)]
    where [scale | ansatz] = circuit params (n_quantum_params) feed the circuit
    and the linear-head params (w, b) turn the per-qubit <Z_i> vector into one
    logit. n_params covers ALL parts. n_scale_params marks the leading scale
    block so the trainer can initialise it to 1.0.
    """
    circuit, param_shape = build_qnode(
        config.encoding, config.ansatz, config.n_qubits, config.depth,
        config.device_name, config.angle_clip, config.trainable_scale,
        config.readout, config.readout_wires,
    )
    n_quantum_params = int(np.prod(param_shape))
    n_scale_params = config.n_qubits if config.trainable_scale else 0
    n_obs = _n_observables(config.n_qubits, config.readout, config.readout_wires)  # head input width
    n_head_params = n_obs + 1  # w (n_obs) + b (1)
    return {
        "circuit": circuit,
        "param_shape": param_shape,
        "n_quantum_params": n_quantum_params,
        "n_scale_params": n_scale_params,
        "n_obs": n_obs,
        "n_head_params": n_head_params,
        "n_qubits": config.n_qubits,
        "n_params": n_quantum_params + n_head_params,
        "config": config,
    }
