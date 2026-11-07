"""Fixed circuit templates whose angles are the weights the LLM writes.

An Ansatz builds a CircuitSpec from a parameter vector ``theta``. Each gate angle is
``coeff * theta[k]``, so one parameter may drive several gates (QAOA layers). ``describe()``
renders the same gate list with symbolic ``theta[k]`` for the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spec import Limits, parse_spec

# A gate with a symbolic angle: (name, qubits, [(coeff, param_index), ...]) or no params.
SymGate = tuple[str, list[int], list[tuple[float, int]]]

_LIMITS = Limits(max_qubits=64, max_gates=10**6, max_features_per_angle=None)


@dataclass
class Ansatz:
    n_qubits: int
    n_params: int
    gates: list[SymGate]
    state_prep: dict = field(default_factory=lambda: {"type": "zero"})
    readout: dict | None = None
    class_labels: list[int] = field(default_factory=lambda: [0, 1])
    name: str = ""
    notes: str = ""  # extra conventions shown to the model (e.g. what theta[k] means)

    def build(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"{self.name}: expected {self.n_params} parameters, got {theta.shape}")
        gates = []
        for name, qubits, params in self.gates:
            g = {"gate": name, "qubits": list(qubits)}
            if params:
                g["params"] = [float(c * theta[k]) for c, k in params]
            gates.append(g)
        obj = {"n_qubits": self.n_qubits, "state_prep": self.state_prep, "gates": gates,
               "readout": self.readout or {"type": "bitstrings", "qubits": [0], "classes": {"0": ["0"], "1": ["1"]}}}
        return parse_spec(obj, self.class_labels, _LIMITS)

    def describe(self) -> str:
        """One gate per line with symbolic angles, in application order."""
        lines = []
        for name, qubits, params in self.gates:
            args = []
            for c, k in params:
                args.append(f"theta[{k}]" if c == 1 else f"{c:g}*theta[{k}]")
            qs = ", ".join(f"q{q}" for q in qubits)
            lines.append(f"{name}({', '.join(args)}) on {qs}" if args else f"{name} on {qs}")
        return "\n".join(lines)


def hardware_efficient(n: int, layers: int, name: str = "") -> Ansatz:
    """RY on every qubit, then a CNOT ladder, repeated; a final RY layer. n*(layers+1) parameters."""
    gates: list[SymGate] = []
    k = 0
    for layer in range(layers + 1):
        for q in range(n):
            gates.append(("ry", [q], [(1.0, k)]))
            k += 1
        if layer < layers:
            for q in range(n - 1):
                gates.append(("cx", [q, q + 1], []))
    return Ansatz(n, k, gates, name=name or f"HEA(n={n}, L={layers})",
                  notes="Parameters are listed layer by layer, qubit by qubit, in the order they appear above.")


def hva_tfim(n: int, p: int, periodic: bool = False) -> Ansatz:
    """Hamiltonian-variational ansatz for the TFIM: |+>^n, then p layers of
    rzz(theta[2k]) on every bond followed by rx(theta[2k+1]) on every qubit."""
    bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    gates: list[SymGate] = [("h", [q], []) for q in range(n)]
    for layer in range(p):
        gates += [("rzz", [i, j], [(1.0, 2 * layer)]) for i, j in bonds]
        gates += [("rx", [q], [(1.0, 2 * layer + 1)]) for q in range(n)]
    return Ansatz(n, 2 * p, gates, name=f"HVA-TFIM(n={n}, p={p})",
                  notes="theta[2k] is the ZZ angle of layer k and theta[2k+1] its X angle. "
                        "With all angles 0 the state is |+>^n, the ground state of the pure transverse field.")


def h2_single_parameter() -> Ansatz:
    """x on q0 prepares the Hartree-Fock state |10>; ry(theta[0]) on q1 then cx(q1,q0) gives
    cos(theta/2)|10> + sin(theta/2)|01>, which spans the exact ground state."""
    gates: list[SymGate] = [("x", [0], []), ("ry", [1], [(1.0, 0)]), ("cx", [1, 0], [])]
    return Ansatz(2, 1, gates, name="H2 one-parameter",
                  notes="With theta[0] = 0 the state is the Hartree-Fock state |10> (q0 = 1, q1 = 0); the circuit "
                        "prepares cos(theta[0]/2)|10> + sin(theta[0]/2)|01>.")


def qaoa_maxcut(n: int, edges: list[tuple[int, int, float]], p: int) -> Ansatz:
    """Standard QAOA: |+>^n, then p layers of U_C(gamma) = exp(-i gamma C) and U_B(beta) = exp(-i beta sum X).
    Parameters are [gamma_1, beta_1, ..., gamma_p, beta_p]."""
    gates: list[SymGate] = [("h", [q], []) for q in range(n)]
    for layer in range(p):
        # exp(-i gamma w (1 - Z Z)/2) = exp(+i gamma w Z Z / 2) up to phase = rzz(-gamma * w)
        gates += [("rzz", [i, j], [(-w, 2 * layer)]) for i, j, w in edges]
        gates += [("rx", [q], [(2.0, 2 * layer + 1)]) for q in range(n)]  # exp(-i beta X) = rx(2 beta)
    return Ansatz(n, 2 * p, gates, name=f"QAOA-MaxCut(n={n}, p={p})",
                  notes="theta[2k] = gamma of layer k+1 and theta[2k+1] = beta of layer k+1, in the convention "
                        "U_C(gamma) = exp(-i gamma C) with C = sum_ij w_ij (1 - Z_i Z_j)/2 and "
                        "U_B(beta) = exp(-i beta sum_i X_i).")


def qml_amplitude_hea(layers: int, class_labels: list[int]) -> Ansatz:
    """Amplitude encoding of the 64 pixels on 6 qubits, then a hardware-efficient ansatz.
    Binary tasks read qubit 0; multi-class tasks read the first ceil(log2 C) qubits."""
    n = 6
    hea = hardware_efficient(n, layers)
    n_ro = max(1, int(np.ceil(np.log2(len(class_labels)))))
    classes = {str(c): [format(j, f"0{n_ro}b")] for j, c in enumerate(class_labels)}
    return Ansatz(
        n, hea.n_params, hea.gates,
        state_prep={"type": "amplitude", "qubits": list(range(n))},
        readout={"type": "bitstrings", "qubits": list(range(n_ro)), "classes": classes},
        class_labels=class_labels,
        name=f"amplitude+HEA(L={layers})",
        notes=hea.notes,
    )
