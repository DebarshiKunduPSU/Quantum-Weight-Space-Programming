"""Cross-check the simulator against Qiskit and test the readout and encoding conventions."""

import numpy as np
import pytest

from weightprog.sim import class_scores, statevector
from weightprog.spec import GATES, Limits, parse_spec

LIMITS = Limits(max_qubits=8, max_gates=500, max_features_per_angle=None)


def _random_circuit(rng, n_qubits, n_gates):
    gates = []
    for _ in range(n_gates):
        name = rng.choice(list(GATES))
        n_q, n_p = GATES[name]
        qubits = [int(q) for q in rng.choice(n_qubits, size=n_q, replace=False)]
        params = [f"{rng.uniform(-3, 3):.6f} + {rng.uniform(-3, 3):.6f}*x[{rng.integers(64)}]" for _ in range(n_p)]
        gates.append({"gate": str(name), "qubits": qubits, "params": params})
    return gates


def _spec(n_qubits, gates, state_prep=None, readout=None):
    obj = {
        "n_qubits": n_qubits,
        "state_prep": state_prep or {"type": "zero"},
        "gates": gates,
        "readout": readout or {"type": "bitstrings", "qubits": [0], "classes": {"0": ["0"], "1": ["1"]}},
    }
    return parse_spec(obj, [0, 1], LIMITS)


@pytest.mark.parametrize("seed", range(5))
def test_matches_qiskit(seed):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    n = 4
    spec = _spec(n, _random_circuit(rng, n, 60))
    X = rng.uniform(0, 1, size=(3, 64))
    ours = statevector(spec, X).reshape(3, -1)

    for b in range(3):
        qc = qiskit.QuantumCircuit(n)
        for gate in spec.gates:
            angles = [float(p.evaluate(X[b : b + 1])[0]) for p in gate.params]
            method = "u" if gate.name == "u3" else gate.name
            getattr(qc, method)(*angles, *gate.qubits)
        # Qiskit is little-endian (qubit 0 = least significant bit); we are big-endian.
        theirs = Statevector(qc.reverse_bits()).data
        np.testing.assert_allclose(ours[b], theirs, atol=1e-10)


def test_bitstring_readout_order():
    # X on qubit 2 only: measuring qubits [2, 0] must give "10".
    readout = {"type": "bitstrings", "qubits": [2, 0], "classes": {"0": ["10"], "1": ["01"]}}
    spec = _spec(3, [{"gate": "x", "qubits": [2]}], readout=readout)
    np.testing.assert_allclose(class_scores(spec, np.zeros((1, 64)), [0, 1]), [[1.0, 0.0]])


def test_qubit_per_class_readout():
    readout = {"type": "qubit_per_class", "classes": {"0": 2, "1": 0}}
    spec = _spec(3, [{"gate": "ry", "qubits": [2], "params": ["pi * x[5]"]}], readout=readout)
    X = np.zeros((2, 64))
    X[1, 5] = 0.5  # ry(pi/2) -> P(1) = 0.5
    np.testing.assert_allclose(class_scores(spec, X, [0, 1]), [[0.0, 0.0], [0.5, 0.0]], atol=1e-12)


def test_amplitude_encoding_row_and_column_qubits():
    # A single lit pixel at row 5, column 2 must put the row qubits in |101> and the column qubits in |010>.
    X = np.zeros((1, 64))
    X[0, 8 * 5 + 2] = 0.7
    prep = {"type": "amplitude", "qubits": [6, 4, 2, 0, 1, 3]}
    readout = {"type": "bitstrings", "qubits": [6, 4, 2, 0, 1, 3], "classes": {"0": ["101010"], "1": ["000000"]}}
    spec = _spec(7, [], state_prep=prep, readout=readout)
    np.testing.assert_allclose(class_scores(spec, X, [0, 1]), [[1.0, 0.0]])


def test_amplitude_encoding_is_normalized():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(4, 64))
    spec = _spec(6, [], state_prep={"type": "amplitude", "qubits": [0, 1, 2, 3, 4, 5]})
    state = statevector(spec, X).reshape(4, -1)
    np.testing.assert_allclose(state, X / np.linalg.norm(X, axis=1, keepdims=True))


def test_shots_converge_to_exact():
    spec = _spec(1, [{"gate": "ry", "qubits": [0], "params": ["pi * x[0]"]}])
    X = np.full((1, 64), 1 / 3)
    exact = class_scores(spec, X, [0, 1])
    sampled = class_scores(spec, X, [0, 1], shots=200_000, rng=np.random.default_rng(0))
    np.testing.assert_allclose(sampled, exact, atol=0.01)
