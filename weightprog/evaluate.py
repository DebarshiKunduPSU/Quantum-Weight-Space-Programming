"""Scoring and the baselines that put the LLM's hand-written weights in context."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .sim import class_scores
from .spec import CircuitSpec


def metrics(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[int]) -> dict:
    confusion = [[int(np.sum((y_true == t) & (y_pred == p))) for p in class_labels] for t in class_labels]
    recalls = [row[j] / max(1, sum(row)) for j, row in enumerate(confusion)]
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion": confusion,  # rows = true class, columns = predicted class, in class_labels order
    }


def evaluate(spec: CircuitSpec, X, y, class_labels, shots=None, rng=None) -> dict:
    scores = class_scores(spec, X, class_labels, shots, rng)
    return metrics(y, np.asarray(class_labels)[scores.argmax(axis=1)], class_labels)


def random_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-np.pi, np.pi, size=n)


def random_weight_baseline(spec, X, y, class_labels, n_draws: int, rng) -> dict:
    """The LLM's architecture with every weight redrawn uniformly from [-pi, pi].

    Separates what the weights contribute from what the architecture and readout contribute.
    """
    n_weights = len(spec.get_weights())
    if n_weights == 0:
        return {"n_draws": 0, "note": "circuit has no weights"}
    accs = [
        evaluate(spec.with_weights(random_weights(n_weights, rng)), X, y, class_labels)["accuracy"]
        for _ in range(n_draws)
    ]
    return {"n_draws": n_draws, "mean": float(np.mean(accs)), "std": float(np.std(accs)), "max": float(np.max(accs))}


class _OutOfTime(Exception):
    pass


def train_weights(
    spec: CircuitSpec,
    X: np.ndarray,
    y: np.ndarray,
    class_labels: list[int],
    init: np.ndarray,
    max_iter: int = 100,
    max_seconds: float = 60.0,
) -> tuple[CircuitSpec, dict]:
    """Optimize the weights of a fixed architecture (L-BFGS-B, finite-difference gradients).

    Minimizes cross-entropy of the normalized class scores. Stops at max_iter or when the
    time budget runs out, returning the best weights seen.
    """
    target = np.array([class_labels.index(int(label)) for label in y])
    rows = np.arange(len(y))
    best = {"loss": np.inf, "w": np.array(init, dtype=float)}
    start = time.time()

    def loss(w: np.ndarray) -> float:
        if time.time() - start > max_seconds:
            raise _OutOfTime
        scores = class_scores(spec.with_weights(w), X, class_labels)
        p = scores[rows, target] / np.clip(scores.sum(axis=1), 1e-12, None)
        value = float(-np.mean(np.log(np.clip(p, 1e-9, None))))
        if value < best["loss"]:
            best["loss"], best["w"] = value, w.copy()
        return value

    stopped = "converged"
    try:
        result = minimize(loss, best["w"], method="L-BFGS-B", options={"maxiter": max_iter})
        if result.nit >= max_iter:
            stopped = "max_iter"
    except _OutOfTime:
        stopped = "time_budget"
    info = {"train_loss": best["loss"], "stopped": stopped, "seconds": round(time.time() - start, 1)}
    return spec.with_weights(best["w"]), info


def trained_baseline(spec, X_fit, y_fit, X_test, y_test, class_labels, init, max_iter, max_seconds) -> dict:
    if len(init) == 0:
        return {"note": "circuit has no weights"}
    trained, info = train_weights(spec, X_fit, y_fit, class_labels, init, max_iter, max_seconds)
    return {
        **info,
        "n_fit": int(len(y_fit)),
        "fit_accuracy": evaluate(trained, X_fit, y_fit, class_labels)["accuracy"],
        "test_accuracy": evaluate(trained, X_test, y_test, class_labels)["accuracy"],
    }


def logistic_baseline(X_fit, y_fit, X_test, y_test) -> float:
    """Classical reference: logistic regression on the raw 64 pixels."""
    clf = LogisticRegression(max_iter=2000).fit(X_fit, y_fit)
    return float(clf.score(X_test, y_test))
