"""Angle expressions, spec validation, and weight round-tripping."""

import math

import numpy as np
import pytest

from weightprog.spec import Limits, SpecError, extract_json, parse_angle, parse_spec


def _obj(**overrides):
    obj = {
        "n_qubits": 2,
        "gates": [
            {"gate": "ry", "qubits": [0], "params": ["0.5*pi*x[27] - 0.3"]},
            {"gate": "cnot", "qubits": [0, 1]},
            {"gate": "rz", "qubits": [1], "params": [0.7]},
        ],
        "readout": {"type": "bitstrings", "qubits": [1], "classes": {"3": ["0"], "8": ["1"]}},
    }
    obj.update(overrides)
    return obj


def test_angle_expressions():
    a = parse_angle("2*(x[3] - 0.5)*pi/4 + -x[10]/2 + 1")
    assert a.const == pytest.approx(1 - math.pi / 4)
    assert a.coeffs == pytest.approx({3: math.pi / 2, 10: -0.5})
    X = np.zeros((2, 64))
    X[1, 3] = 1.0
    np.testing.assert_allclose(a.evaluate(X), [1 - math.pi / 4, 1 + math.pi / 4])
    assert parse_angle(1.5).const == 1.5 and not parse_angle("pi").coeffs


@pytest.mark.parametrize(
    "expr", ["x[1]*x[2]", "1/x[3]", "sin(x[1])", "x[64]", "x[i]", "__import__('os')", "1/0", "x[1] +", True, None]
)
def test_bad_angles_are_rejected(expr):
    with pytest.raises(SpecError):
        parse_angle(expr)


def test_valid_spec_and_weights_roundtrip():
    spec = parse_spec(_obj(), [3, 8], Limits())
    assert spec.gates[1].name == "cx"
    w = spec.get_weights()
    np.testing.assert_allclose(w, [-0.3, 0.5 * math.pi, 0.7])
    np.testing.assert_allclose(spec.with_weights(w + 1).get_weights(), w + 1)
    np.testing.assert_allclose(spec.get_weights(), w)  # original untouched


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"n_qubits": 9}, "exceeds the limit"),
        ({"gates": [{"gate": "foo", "qubits": [0]}]}, "unknown gate"),
        ({"gates": [{"gate": "cx", "qubits": [0, 0]}]}, "distinct"),
        ({"gates": [{"gate": "ry", "qubits": [0]}]}, "takes 1 parameter"),
        ({"gates": [{"gate": "h", "qubits": [2]}]}, "out of range"),
        ({"gates": [{"gate": "ry", "qubits": [0], "params": ["x[1] + x[2]"]}]}, "at most 1 feature"),
        ({"readout": {"type": "bitstrings", "qubits": [1], "classes": {"3": ["0"]}}}, "no outcome assigned"),
        ({"readout": {"type": "bitstrings", "qubits": [1], "classes": {"3": ["0"], "8": ["0"]}}}, "more than one class"),
        ({"readout": {"type": "bitstrings", "qubits": [1], "classes": {"3": ["0"], "7": ["1"]}}}, "not one of"),
        ({"readout": {"type": "qubit_per_class", "classes": {"3": 0, "8": 0}}}, "its own qubit"),
        ({"state_prep": {"type": "amplitude", "qubits": [0, 1]}}, "exactly 6 qubits"),
    ],
)
def test_invalid_specs_give_actionable_errors(overrides, fragment):
    with pytest.raises(SpecError, match=fragment):
        parse_spec(_obj(**overrides), [3, 8], Limits())


def test_feature_limit_can_be_lifted():
    gates = [{"gate": "ry", "qubits": [0], "params": ["x[1] + x[2] + x[3]"]}]
    parse_spec(_obj(gates=gates), [3, 8], Limits(max_features_per_angle=None))


def test_extract_json_prefers_last_fenced_block():
    text = 'First idea:\n```json\n{"n_qubits": 1}\n```\nBetter:\n```json\n{"n_qubits": 2, "a": {"b": 1}}\n```\nDone.'
    assert extract_json(text)["n_qubits"] == 2
    assert extract_json('prose {"n_qubits": 3} prose')["n_qubits"] == 3
    with pytest.raises(SpecError):
        extract_json("no circuit here")
