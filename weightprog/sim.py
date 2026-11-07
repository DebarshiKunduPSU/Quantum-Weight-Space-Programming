"""Batched statevector simulator for CircuitSpec.

The state is an array of shape (B, 2, ..., 2): axis 0 is the batch of input images and
axis q + 1 is qubit q, so qubit 0 is the most significant bit of a flat basis index.
Gate definitions follow Qiskit; tests/test_sim.py cross-checks against Qiskit.
"""

from __future__ import annotations

import numpy as np

from .spec import CircuitSpec

_S2 = 1 / np.sqrt(2)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_FIXED_1Q = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "x": _X,
    "y": _Y,
    "z": _Z,
    "s": np.diag([1, 1j]).astype(complex),
    "sdg": np.diag([1, -1j]).astype(complex),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
}
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_PAULI_PAIRS = {"rxx": np.kron(_X, _X), "ryy": np.kron(_Y, _Y), "rzz": np.kron(_Z, _Z)}


def _mat2(a, b, c, d) -> np.ndarray:
    """(b, 2, 2) matrices [[a, b], [c, d]] from entries of shape (b,) or (1,)."""
    a, b, c, d = np.broadcast_arrays(*(np.asarray(v, dtype=complex) for v in (a, b, c, d)))
    return np.stack([np.stack([a, b], -1), np.stack([c, d], -1)], -2)


def _rotation(name: str, angles: list[np.ndarray]) -> np.ndarray:
    if name == "u3":
        theta, phi, lam = angles
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return _mat2(c, -np.exp(1j * lam) * s, np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c)
    (t,) = angles
    c, s = np.cos(t / 2), np.sin(t / 2)
    if name == "rx":
        return _mat2(c, -1j * s, -1j * s, c)
    if name == "ry":
        return _mat2(c, -s, s, c)
    if name == "rz":
        return _mat2(np.exp(-0.5j * t), 0, 0, np.exp(0.5j * t))
    if name == "p":
        return _mat2(1, 0, 0, np.exp(1j * t))
    raise KeyError(name)


def _controlled(u: np.ndarray) -> np.ndarray:
    """Add one control qubit (placed first) to matrices of shape (b, d, d)."""
    b, d, _ = u.shape
    out = np.zeros((b, 2 * d, 2 * d), dtype=complex)
    out[:, :d, :d] = np.eye(d)
    out[:, d:, d:] = u
    return out


def gate_matrix(name: str, angles: list[np.ndarray]) -> np.ndarray:
    """Unitaries of shape (b, 2^k, 2^k), with b = 1 for data-independent gates."""
    if name in _FIXED_1Q:
        return _FIXED_1Q[name][None]
    if name in ("rx", "ry", "rz", "p", "u3"):
        return _rotation(name, angles)
    if name == "swap":
        return _SWAP[None]
    if name in ("cx", "cy", "cz"):
        return _controlled(_FIXED_1Q[name[1]][None])
    if name == "ccx":
        return _controlled(_controlled(_X[None]))
    if name in ("crx", "cry", "crz", "cp"):
        return _controlled(_rotation(name[1:], angles))
    if name in _PAULI_PAIRS:
        (t,) = angles
        c, s = np.cos(t / 2)[:, None, None], np.sin(t / 2)[:, None, None]
        return c * np.eye(4) - 1j * s * _PAULI_PAIRS[name]
    raise KeyError(name)


def _apply(state: np.ndarray, u: np.ndarray, qubits: list[int]) -> np.ndarray:
    n = state.ndim - 1
    k = len(qubits)
    src = [q + 1 for q in qubits]
    dst = list(range(n - k + 1, n + 1))
    st = np.moveaxis(state, src, dst)
    shape = st.shape
    st = st.reshape(shape[0], -1, 2**k)
    st = st @ np.swapaxes(u, -1, -2)  # new[b, r, i] = sum_j u[b, i, j] * st[b, r, j]
    return np.moveaxis(st.reshape(shape), dst, src)


def initial_state(spec: CircuitSpec, X: np.ndarray) -> np.ndarray:
    B, n = X.shape[0], spec.n_qubits
    if spec.state_prep == "zero":
        state = np.zeros((B, 2**n), dtype=complex)
        state[:, 0] = 1.0
        return state.reshape((B,) + (2,) * n)
    amp = X.astype(complex)
    norms = np.linalg.norm(X, axis=1)
    blank = norms == 0
    amp[blank, 0] = 1.0  # an all-black image has no direction; fall back to |0...0>
    amp /= np.where(blank, 1.0, norms)[:, None]
    order = spec.amplitude_qubits + [q for q in range(n) if q not in spec.amplitude_qubits]
    state = np.zeros((B, amp.shape[1], 2 ** (n - len(spec.amplitude_qubits))), dtype=complex)
    state[:, :, 0] = amp
    state = state.reshape((B,) + (2,) * n)  # axis k + 1 currently holds qubit order[k]
    return np.moveaxis(state, range(1, n + 1), [q + 1 for q in order])


def statevector(spec: CircuitSpec, X: np.ndarray) -> np.ndarray:
    """Final states, shape (B, 2, ..., 2)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    state = initial_state(spec, X)
    for gate in spec.gates:
        angles = [p.evaluate(X) for p in gate.params]
        state = _apply(state, gate_matrix(gate.name, angles), gate.qubits)
    return state


def class_scores(
    spec: CircuitSpec,
    X: np.ndarray,
    class_labels: list[int],
    shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Class scores of shape (B, n_classes). Exact probabilities, or estimates from ``shots`` samples."""
    state = statevector(spec, X)
    B, n = state.shape[0], spec.n_qubits
    probs = (state.real**2 + state.imag**2).reshape(B, -1)
    if shots:
        rng = rng or np.random.default_rng()
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs = np.stack([rng.multinomial(shots, p) for p in probs]) / shots
    probs = probs.reshape((B,) + (2,) * n)

    ro = spec.readout
    scores = np.zeros((B, len(class_labels)))
    if ro.kind == "bitstrings":
        measured = [q + 1 for q in ro.qubits]
        rest = [a for a in range(1, n + 1) if a not in measured]
        marginal = probs.transpose([0] + measured + rest).reshape(B, 2 ** len(measured), -1).sum(-1)
        for j, label in enumerate(class_labels):
            scores[:, j] = marginal[:, [int(s, 2) for s in ro.classes[label]]].sum(-1)
    else:
        for j, label in enumerate(class_labels):
            axis = ro.classes[label] + 1
            other = tuple(a for a in range(1, n + 1) if a != axis)
            scores[:, j] = probs.sum(axis=other)[:, 1]
    return scores


def predict(spec, X, class_labels, shots=None, rng=None) -> np.ndarray:
    scores = class_scores(spec, X, class_labels, shots, rng)
    return np.asarray(class_labels)[scores.argmax(axis=1)]
