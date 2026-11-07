"""QML with a fixed ansatz: the model only writes the weights of a given circuit.

Amplitude encoding of all 64 pixels on 6 qubits, then a hardware-efficient ansatz; the model
sees the same few-shot images as in the free-design experiment.

    python -m weightprog.qml_fixed --model fable --trials 5
"""

from __future__ import annotations

import sys

import numpy as np

from . import ansatz as az
from . import data, evaluate
from .harness import GATE_CONVENTIONS, Case, Message, load_base_config, parse_weights, standard_cli, weights_format

SYSTEM = (
    "You are an expert in quantum machine learning. You will set the weights of a fixed parameterized "
    "quantum circuit classifier by hand. No optimizer or training loop runs afterwards: the circuit is "
    "simulated exactly with the numbers you write and scored on held-out images. You receive no feedback.\n\n"
    "Input features: each image is 8x8 grayscale pixels; x[i] is the pixel in row r, column c with "
    "i = 8*r + c (row 0 at the top, column 0 at the left), scaled to [0, 1] (0 = black, 1 = brightest ink).\n\n"
    + GATE_CONVENTIONS
)


def _user(ans: az.Ansatz, class_labels, X_sup, y_sup, shots, cell_px) -> Message:
    msg = Message("user", [])
    labels = ", ".join(map(str, class_labels))
    n_ro = len(ans.readout["qubits"])
    classes = "; ".join(f"class {c} ↔ {ans.readout['classes'][str(c)][0]}" for c in class_labels)
    msg.parts.append(
        f"Task: classify 8x8 images of handwritten digits into the classes {labels}. "
        f"Below are {shots} labelled training example(s) per class, shown side by side and separated by blue bars; "
        "pixel brightness is the feature value."
    )
    for label in class_labels:
        xs = X_sup[y_sup == label]
        msg.parts.append(f"Class {label}, {len(xs)} example(s), left to right:")
        msg.parts.append(data.render_row(xs, cell_px))
    msg.parts.append(
        f"Circuit ({ans.name}), {ans.n_qubits} qubits, {ans.n_params} weights.\n"
        "State preparation: amplitude encoding. The 6 qubits start in sum_i (x[i] / ||x||) |i>, where i is "
        "read in binary with q0 as the most significant bit, so q0 q1 q2 hold the pixel row and q3 q4 q5 the "
        "pixel column.\n"
        f"Gates, in order:\n{ans.describe()}\n\n{ans.notes}\n\n"
        f"Readout: qubit(s) {', '.join(f'q{q}' for q in ans.readout['qubits'])} are measured; the class score is "
        f"the probability of the bitstring assigned to it ({classes}; character k is the outcome of the k-th "
        f"listed qubit). The predicted class is the one with the highest score.\n\n"
        "Choose the weights so that the circuit separates the classes. It is scored by accuracy on held-out "
        "test images.\n\n" + weights_format(ans.n_params)
    )
    return msg


def _support(task: data.Task, shots: int, seed: int, trial: int):
    # Same seeding as the free-design runner: trial k shows the same examples in both experiments.
    return data.sample_support(task, shots, np.random.default_rng([seed, trial]))


def _score_factory(task: data.Task, ans: az.Ansatz, shots: int, seed: int, b_cfg: dict):
    labels = task.class_labels

    def score(theta: np.ndarray, rng: np.random.Generator, trial: int) -> dict:
        X_sup, y_sup = _support(task, shots, seed, trial)
        spec = ans.build(theta)
        test = evaluate.evaluate(spec, task.X_test, task.y_test, labels)
        support = evaluate.evaluate(spec, X_sup, y_sup, labels)
        budget = (b_cfg["train_max_iter"], b_cfg["train_max_seconds"])
        random = evaluate.random_weight_baseline(spec, task.X_test, task.y_test, labels, b_cfg["random_weight_draws"], rng)
        trained = evaluate.trained_baseline(spec, X_sup, y_sup, task.X_test, task.y_test, labels,
                                            evaluate.random_weights(ans.n_params, rng), *budget)
        finetuned = evaluate.trained_baseline(spec, X_sup, y_sup, task.X_test, task.y_test, labels, theta, *budget)
        n_fit = min(len(task.y_train), b_cfg["train_full_max_samples"])
        idx = rng.choice(len(task.y_train), size=n_fit, replace=False)
        full = evaluate.trained_baseline(spec, task.X_train[idx], task.y_train[idx], task.X_test, task.y_test, labels,
                                         evaluate.random_weights(ans.n_params, rng), *budget)
        return {
            "weights": theta.tolist(),
            "llm": {"test": test, "support": support},
            "baselines": {"random_weights": random, "trained_on_support": trained,
                          "llm_finetuned_on_support": finetuned, "trained_on_full": full,
                          "logreg_on_support": evaluate.logistic_baseline(X_sup, y_sup, task.X_test, task.y_test)},
            "summary": {
                "acc_llm": test["accuracy"],
                "acc_support": support["accuracy"],
                "acc_random": random["mean"],
                "acc_trained_support": trained["test_accuracy"],
                "acc_llm_finetuned": finetuned["test_accuracy"],
                "acc_trained_full": full["test_accuracy"],
                "chance": 1 / len(labels),
            },
        }

    return score


def build_cases() -> list[Case]:
    cfg = load_base_config()
    p_cfg, b_cfg = cfg["prompt"], cfg["baselines"]
    cases = []
    for classes in ([3, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        task = data.load_task(classes, cfg["task"]["test_fraction"], cfg["task"]["split_seed"])
        shots = p_cfg["shots_per_class"] if len(classes) == 2 else 3
        for layers in (1, 2):
            ans = az.qml_amplitude_hea(layers, task.class_labels)
            cid = f"digits_{''.join(map(str, classes)) if len(classes) < 4 else 'all10'}_amp_hea_L{layers}"
            def user(trial, task=task, ans=ans, shots=shots):
                X_sup, y_sup = _support(task, shots, cfg["seed"], trial)
                return _user(ans, task.class_labels, X_sup, y_sup, shots, p_cfg["cell_px"])

            cases.append(Case(cid, SYSTEM, user, lambda t, n=ans.n_params: parse_weights(t, n),
                              _score_factory(task, ans, shots, cfg["seed"], b_cfg),
                              {"classes": task.class_labels, "shots_per_class": shots, "ansatz": ans.name, "n_params": ans.n_params}))
    return cases


if __name__ == "__main__":
    sys.exit(standard_cli("qml_fixed", build_cases, __doc__))
