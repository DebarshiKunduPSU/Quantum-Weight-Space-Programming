"""QAOA weight programming: the model writes (gamma, beta) for MaxCut on small graphs.

    python -m weightprog.qaoa --model fable --trials 3
"""

from __future__ import annotations

import sys

import numpy as np

from . import ansatz as az
from . import hamiltonians as hm
from .harness import GATE_CONVENTIONS, Case, Message, evals_to_reach, optimize, parse_weights, standard_cli, weights_format
from .sim import statevector

SYSTEM = (
    "You are an expert in the Quantum Approximate Optimization Algorithm (QAOA). You will set the "
    "QAOA angles by hand for a MaxCut instance. No optimizer runs afterwards: the circuit is simulated "
    "exactly with the numbers you write and the expected cut value is evaluated. You receive no feedback, "
    "so reason carefully about the structure of the graph and what good angles look like.\n\n" + GATE_CONVENTIONS
)


def _graph(kind: str, n: int, rng: np.random.Generator) -> list[tuple[int, int, float]]:
    if kind == "3regular":
        while True:  # pairing model; retry until simple
            stubs = np.repeat(np.arange(n), 3)
            rng.shuffle(stubs)
            pairs = {tuple(sorted(p)) for p in stubs.reshape(-1, 2)}
            if len(pairs) == 3 * n // 2 and all(i != j for i, j in pairs):
                return [(int(i), int(j), 1.0) for i, j in sorted(pairs)]
    if kind == "er":
        edges = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n) if rng.uniform() < 0.5]
        return edges
    if kind == "wring":
        return [(i, (i + 1) % n, float(rng.integers(1, 6))) for i in range(n)]
    raise ValueError(kind)


def _user(n: int, edges: list[tuple[int, int, float]], kind: str, p: int, ans: az.Ansatz) -> Message:
    weighted = any(w != 1.0 for _, _, w in edges)
    edge_text = ", ".join(f"({i},{j}, w={w:g})" if weighted else f"({i},{j})" for i, j, w in edges)
    text = (
        f"MaxCut instance: {kind}, {n} nodes (0..{n - 1}), {len(edges)} edges"
        + (" with integer weights" if weighted else ", all weights 1") + f":\n{edge_text}\n\n"
        "Cut value C(z) = sum over edges w_ij * [z_i != z_j]. Node i is qubit q_i.\n\n"
        f"QAOA circuit with p = {p} layer(s), {ans.n_params} parameters, applied in order:\n{ans.describe()}\n\n"
        f"{ans.notes}\n\nChoose the angles to maximize the expected cut value <C>.\n\n" + weights_format(ans.n_params)
    )
    return Message("user", [text])


def _score_factory(n: int, edges, ans: az.Ansatz, n_random: int = 50, n_restarts: int = 5):
    C = hm.maxcut(n, edges)
    C_max = hm.maxcut_optimum(n, edges)
    X0 = np.zeros((1, 64))

    def cut(theta: np.ndarray) -> float:
        return float(C.expectation(statevector(ans.build(theta), X0))[0])

    def score(theta: np.ndarray, rng: np.random.Generator, trial: int) -> dict:
        value = cut(theta)
        random_v = [cut(rng.uniform(0, np.pi, ans.n_params)) for _ in range(n_random)]
        runs = [optimize(lambda t: -cut(t), rng.uniform(0, np.pi, ans.n_params)) for _ in range(n_restarts)]
        best = max(runs, key=lambda r: -r["value"])
        warm = optimize(lambda t: -cut(t), theta)
        target = best["value"] + 1e-3 * C_max  # traces hold -<C>; target is within 0.1% of the best
        return {
            "weights": theta.tolist(),
            "expected_cut": value,
            "max_cut": C_max,
            "random_params": {"n": n_random, "mean_cut": float(np.mean(random_v)), "best_cut": float(np.max(random_v))},
            "optimizer_random_init": {"restarts": n_restarts, "best_cut": -best["value"],
                                      "mean_cut": float(np.mean([-r["value"] for r in runs])),
                                      "evals_to_target": [evals_to_reach(r["trace"], target) for r in runs],
                                      "traces": [r["trace"] for r in runs]},
            "optimizer_llm_init": {"cut": -warm["value"], "evals_to_target": evals_to_reach(warm["trace"], target),
                                   "trace": warm["trace"]},
            "summary": {
                "ratio_llm": value / C_max,
                "ratio_random": float(np.mean(random_v)) / C_max,
                "ratio_optimizer": -best["value"] / C_max,
                "ratio_llm_then_opt": -warm["value"] / C_max,
            },
        }

    return score


def build_cases() -> list[Case]:
    n = 8
    graphs = []
    for k in range(3):
        graphs.append((f"3reg{k}", "3-regular graph", _graph("3regular", n, np.random.default_rng(100 + k))))
    for k in range(2):
        graphs.append((f"er{k}", "Erdős–Rényi graph (edge probability 0.5)", _graph("er", n, np.random.default_rng(200 + k))))
    graphs.append(("wring0", "weighted ring", _graph("wring", n, np.random.default_rng(300))))
    cases = []
    for gid, kind, edges in graphs:
        for p in (1, 3):
            ans = az.qaoa_maxcut(n, edges, p)
            cases.append(Case(f"maxcut_{gid}_p{p}", SYSTEM, _user(n, edges, kind, p, ans),
                              lambda t, m=ans.n_params: parse_weights(t, m), _score_factory(n, edges, ans),
                              {"graph": kind, "n": n, "edges": edges, "p": p, "n_params": ans.n_params}))
    return cases


if __name__ == "__main__":
    sys.exit(standard_cli("qaoa", build_cases, __doc__))
