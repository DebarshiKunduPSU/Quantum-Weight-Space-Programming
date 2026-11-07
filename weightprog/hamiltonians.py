"""Pauli-sum Hamiltonians, exact diagonalization, and expectation values on simulator states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import _I2, _X, _Y, _Z, _apply

_PAULI = {"X": _X, "Y": _Y, "Z": _Z}


@dataclass
class Hamiltonian:
    n_qubits: int
    terms: list[tuple[float, dict[int, str]]]  # (coefficient, {qubit: "X"|"Y"|"Z"}); {} is the identity
    name: str = ""

    def describe(self) -> str:
        """Human-readable sum, e.g. ``-1.0 Z0 Z1 - 0.5 X0``."""
        parts = []
        for coeff, ops in self.terms:
            label = " ".join(f"{p}{q}" for q, p in sorted(ops.items())) or "I"
            parts.append(f"{coeff:+.4f} {label}")
        return " ".join(parts)

    def matrix(self) -> np.ndarray:
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for coeff, ops in self.terms:
            term = np.array([[1.0]], dtype=complex)
            for q in range(self.n_qubits):
                term = np.kron(term, _PAULI[ops[q]] if q in ops else _I2)
            H += coeff * term
        return H

    def ground(self) -> tuple[float, np.ndarray]:
        """Exact ground energy and state (big-endian basis order, matching the simulator)."""
        energies, states = np.linalg.eigh(self.matrix())
        return float(energies[0]), states[:, 0]

    def expectation(self, state: np.ndarray) -> np.ndarray:
        """<psi|H|psi> for a batch of simulator states of shape (B, 2, ..., 2)."""
        B = state.shape[0]
        flat = state.reshape(B, -1)
        energy = np.zeros(B)
        for coeff, ops in self.terms:
            phi = state
            for q, p in ops.items():
                phi = _apply(phi, _PAULI[p][None], [q])
            energy += coeff * np.real(np.einsum("bi,bi->b", flat.conj(), phi.reshape(B, -1)))
        return energy


def tfim(n: int, h: float, J: float = 1.0, periodic: bool = False) -> Hamiltonian:
    """Transverse-field Ising chain  H = -J sum Z_i Z_{i+1} - h sum X_i."""
    bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    terms = [(-J, {i: "Z", j: "Z"}) for i, j in bonds] + [(-h, {i: "X"}) for i in range(n)]
    return Hamiltonian(n, terms, f"TFIM n={n} J={J} h={h}" + (" periodic" if periodic else " open"))


def heisenberg_xxz(n: int, delta: float = 1.0, J: float = 1.0, periodic: bool = False) -> Hamiltonian:
    """XXZ chain  H = J sum (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1})."""
    bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    terms = []
    for i, j in bonds:
        terms += [(J, {i: "X", j: "X"}), (J, {i: "Y", j: "Y"}), (J * delta, {i: "Z", j: "Z"})]
    return Hamiltonian(n, terms, f"XXZ n={n} J={J} delta={delta}" + (" periodic" if periodic else " open"))


def h2_two_qubit() -> Hamiltonian:
    """H2 in a minimal basis reduced to two qubits (coefficients in Hartree, as in
    O'Malley et al., PRX 6, 031007 (2016), near the equilibrium bond length)."""
    g0, g1, g2, g3, g4, g5 = -0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910
    terms = [
        (g0, {}),
        (g1, {0: "Z"}),
        (g2, {1: "Z"}),
        (g3, {0: "Z", 1: "Z"}),
        (g4, {0: "Y", 1: "Y"}),
        (g5, {0: "X", 1: "X"}),
    ]
    return Hamiltonian(2, terms, "H2 (2-qubit reduced, STO-3G, R≈0.74 Å)")


def maxcut(n: int, edges: list[tuple[int, int, float]]) -> Hamiltonian:
    """MaxCut cost  C = sum_ij w_ij (1 - Z_i Z_j) / 2  as a Hamiltonian to maximize."""
    terms = [(sum(w for _, _, w in edges) / 2, {})]
    terms += [(-w / 2, {i: "Z", j: "Z"}) for i, j, w in edges]
    return Hamiltonian(n, terms, f"MaxCut n={n} m={len(edges)}")


def maxcut_optimum(n: int, edges: list[tuple[int, int, float]]) -> float:
    """Brute-force maximum cut value."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, ::-1]) & 1  # big-endian rows
    cut = np.zeros(2**n)
    for i, j, w in edges:
        cut += w * (bits[:, i] != bits[:, j])
    return float(cut.max())
