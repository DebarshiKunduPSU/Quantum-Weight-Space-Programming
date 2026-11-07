"""VQE weight programming: the model writes the angles of a fixed ansatz to minimize an energy.

    python -m weightprog.vqe --model fable --trials 3
    python -m weightprog.vqe --model gpt-6-astra --cases tfim
"""

from __future__ import annotations

import sys

import numpy as np

from . import ansatz as az
from . import hamiltonians as hm
from .harness import GATE_CONVENTIONS, Case, Message, evals_to_reach, optimize, parse_weights, standard_cli, weights_format
from .sim import statevector

SYSTEM = (
    "You are an expert in variational quantum algorithms. You will set the parameters of a fixed "
    "quantum circuit by hand so that the state it prepares has the lowest possible energy for a given "
    "Hamiltonian. No optimizer runs afterwards: the circuit is simulated exactly with the numbers you "
    "write and its energy is evaluated. You receive no feedback, so reason carefully about the physics "
    "of the ground state and how each parameter shapes the state.\n\n" + GATE_CONVENTIONS
)


def _user(H: hm.Hamiltonian, physics: str, ans: az.Ansatz) -> Message:
    text = (
        f"Hamiltonian: {physics}\n"
        f"H = {H.describe()}\n"
        f"(Pauli operators act on qubits q0..q{H.n_qubits - 1}; the ground energy is not given.)\n\n"
        f"Circuit ({ans.name}), {ans.n_qubits} qubit(s), {ans.n_params} parameter(s):\n{ans.describe()}\n\n"
        f"{ans.notes}\n\nChoose theta[0..{ans.n_params - 1}] to minimize the energy <psi|H|psi>.\n\n"
        + weights_format(ans.n_params)
    )
    return Message("user", [text])


def _score_factory(H: hm.Hamiltonian, ans: az.Ansatz, n_random: int = 50, n_restarts: int = 5):
    E0, psi0 = H.ground()
    E_max = float(np.linalg.eigvalsh(H.matrix())[-1])
    X0 = np.zeros((1, 64))

    def energy(theta: np.ndarray) -> float:
        return float(H.expectation(statevector(ans.build(theta), X0))[0])

    def ratio(E: float) -> float:
        """1 at the ground state, 0 at the highest eigenvalue."""
        return (E_max - E) / (E_max - E0)

    def score(theta: np.ndarray, rng: np.random.Generator, trial: int) -> dict:
        E = energy(theta)
        state = statevector(ans.build(theta), X0).reshape(-1)
        fidelity = float(abs(np.vdot(psi0, state)) ** 2)
        random_E = [energy(rng.uniform(-np.pi, np.pi, ans.n_params)) for _ in range(n_random)]
        runs = [optimize(energy, rng.uniform(-np.pi, np.pi, ans.n_params)) for _ in range(n_restarts)]
        best = min(runs, key=lambda r: r["value"])
        warm = optimize(energy, theta)
        target = best["value"] + 1e-3 * (E_max - E0)
        return {
            "weights": theta.tolist(),
            "energy": E,
            "exact_ground_energy": E0,
            "highest_energy": E_max,
            "ground_state_fidelity": fidelity,
            "random_params": {"n": n_random, "mean_energy": float(np.mean(random_E)), "best_energy": float(np.min(random_E))},
            "optimizer_random_init": {"restarts": n_restarts, "best_energy": best["value"],
                                      "mean_energy": float(np.mean([r["value"] for r in runs])),
                                      "evals_to_target": [evals_to_reach(r["trace"], target) for r in runs],
                                      "traces": [r["trace"] for r in runs]},
            "optimizer_llm_init": {"energy": warm["value"], "evals_to_target": evals_to_reach(warm["trace"], target),
                                   "trace": warm["trace"]},
            "summary": {
                "ratio_llm": ratio(E),
                "ratio_random": ratio(float(np.mean(random_E))),
                "ratio_optimizer": ratio(best["value"]),
                "ratio_llm_then_opt": ratio(warm["value"]),
                "error_llm": E - E0,
                "fidelity_llm": fidelity,
            },
        }

    return score


def build_cases() -> list[Case]:
    cases = []
    problems: list[tuple[str, hm.Hamiltonian, str, az.Ansatz]] = []
    for h in (0.5, 1.0, 2.0):
        H = hm.tfim(6, h)
        physics = f"transverse-field Ising chain of 6 spins, open boundary, J = 1, transverse field h = {h}"
        problems.append((f"tfim_n6_h{h}_hva_p2", H, physics, az.hva_tfim(6, 2)))
        problems.append((f"tfim_n6_h{h}_hea_L2", H, physics, az.hardware_efficient(6, 2)))
    H = hm.heisenberg_xxz(6, 1.0)
    physics = "antiferromagnetic Heisenberg chain of 6 spins, open boundary, J = 1 (XXX point)"
    problems.append(("xxz_n6_hea_L2", H, physics, az.hardware_efficient(6, 2)))
    H = hm.h2_two_qubit()
    physics = ("hydrogen molecule H2 near its equilibrium bond length in a minimal (STO-3G) basis, reduced to "
               "2 qubits; electronic energies in Hartree, nuclear repulsion excluded")
    problems.append(("h2_ucc_1param", H, physics, az.h2_single_parameter()))
    problems.append(("h2_hea_L1", H, physics, az.hardware_efficient(2, 1)))
    for cid, H, physics, ans in problems:
        n = ans.n_params
        cases.append(Case(cid, SYSTEM, _user(H, physics, ans), lambda t, n=n: parse_weights(t, n),
                          _score_factory(H, ans), {"hamiltonian": H.name, "ansatz": ans.name, "n_params": n}))
    return cases


if __name__ == "__main__":
    sys.exit(standard_cli("vqe", build_cases, __doc__))
