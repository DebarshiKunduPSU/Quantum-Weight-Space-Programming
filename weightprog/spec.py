"""Circuit DSL: the JSON format the LLM writes, plus parsing and validation.

Every gate angle is an affine function of the input features, ``c + sum_i w_i * x[i]``.
A plain number is a fixed weight; a term in ``x[i]`` is data encoding. The constants and
coefficients together are the "weights" the LLM programs by hand.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

N_FEATURES = 64

# name -> (number of qubits, number of angle parameters)
GATES: dict[str, tuple[int, int]] = {
    "h": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0),
    "s": (1, 0), "sdg": (1, 0), "t": (1, 0), "tdg": (1, 0), "sx": (1, 0),
    "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1), "u3": (1, 3),
    "cx": (2, 0), "cy": (2, 0), "cz": (2, 0), "swap": (2, 0),
    "crx": (2, 1), "cry": (2, 1), "crz": (2, 1), "cp": (2, 1),
    "rxx": (2, 1), "ryy": (2, 1), "rzz": (2, 1),
    "ccx": (3, 0),
}
ALIASES = {"cnot": "cx", "toffoli": "ccx", "phase": "p", "u": "u3", "cphase": "cp"}


class SpecError(ValueError):
    """The circuit spec is malformed. The message is shown to the LLM for repair."""


@dataclass
class Affine:
    """``const + sum(coeffs[i] * x[i])``."""

    const: float = 0.0
    coeffs: dict[int, float] = field(default_factory=dict)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Angles for a batch of feature vectors. Shape (B,), or (1,) if data-independent."""
        if not self.coeffs:
            return np.array([self.const])
        idx = list(self.coeffs)
        return self.const + X[:, idx] @ np.array([self.coeffs[i] for i in idx])


def _combine(a: Affine, b: Affine, sign: float) -> Affine:
    coeffs = dict(a.coeffs)
    for i, w in b.coeffs.items():
        coeffs[i] = coeffs.get(i, 0.0) + sign * w
    return Affine(a.const + sign * b.const, {i: w for i, w in coeffs.items() if w != 0.0})


def _scale(a: Affine, k: float) -> Affine:
    return Affine(a.const * k, {i: w * k for i, w in a.coeffs.items() if w * k != 0.0})


def _eval_node(node: ast.AST) -> Affine:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Affine(float(node.value))
    if isinstance(node, ast.Name) and node.id == "pi":
        return Affine(math.pi)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "x":
        idx = node.slice
        if not (isinstance(idx, ast.Constant) and isinstance(idx.value, int) and not isinstance(idx.value, bool)):
            raise SpecError("feature index must be an integer literal, e.g. x[27]")
        if not 0 <= idx.value < N_FEATURES:
            raise SpecError(f"feature index x[{idx.value}] out of range 0..{N_FEATURES - 1}")
        return Affine(0.0, {idx.value: 1.0})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _eval_node(node.operand)
        return _scale(operand, -1.0) if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return _combine(left, right, 1.0)
        if isinstance(node.op, ast.Sub):
            return _combine(left, right, -1.0)
        if isinstance(node.op, ast.Mult):
            if left.coeffs and right.coeffs:
                raise SpecError("angles must be affine in x: a product of two x-dependent terms is not allowed")
            return _scale(right, left.const) if not left.coeffs else _scale(left, right.const)
        if isinstance(node.op, ast.Div):
            if right.coeffs:
                raise SpecError("angles must be affine in x: division by an x-dependent term is not allowed")
            if right.const == 0.0:
                raise SpecError("division by zero")
            return _scale(left, 1.0 / right.const)
    raise SpecError("only numbers, pi, x[i], + - * / and parentheses are allowed")


def parse_angle(expr) -> Affine:
    """Parse an angle given as a number or a string such as ``"0.5*pi*x[27] - 0.3"``."""
    if isinstance(expr, bool):
        raise SpecError(f"invalid angle {expr!r}")
    if isinstance(expr, (int, float)):
        value = float(expr)
    elif isinstance(expr, str):
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as e:
            raise SpecError(f"cannot parse angle expression {expr!r}: {e.msg}") from None
        try:
            affine = _eval_node(tree.body)
        except SpecError as e:
            raise SpecError(f"invalid angle expression {expr!r}: {e}") from None
        if not all(math.isfinite(v) for v in (affine.const, *affine.coeffs.values())):
            raise SpecError(f"angle expression {expr!r} is not finite")
        return affine
    else:
        raise SpecError(f"angle must be a number or an expression string, got {type(expr).__name__}")
    if not math.isfinite(value):
        raise SpecError(f"angle {expr!r} is not finite")
    return Affine(value)


@dataclass
class Gate:
    name: str
    qubits: list[int]
    params: list[Affine]


@dataclass
class Readout:
    """How measurement outcomes become class scores.

    kind "bitstrings":      score(c) = sum of P(bitstring) over the bitstrings assigned to c,
                            measured on ``qubits`` (character k of a bitstring is qubits[k]).
    kind "qubit_per_class": score(c) = P(class qubit of c measures 1).
    """

    kind: str
    qubits: list[int]
    classes: dict[int, list[str]] | dict[int, int]


@dataclass
class CircuitSpec:
    n_qubits: int
    state_prep: str  # "zero" | "amplitude"
    amplitude_qubits: list[int]
    gates: list[Gate]
    readout: Readout
    reasoning: str = ""

    # -- weights: every angle constant and every feature coefficient, in a fixed order --

    def _slots(self):
        for gate in self.gates:
            for affine in gate.params:
                yield affine, None
                for i in sorted(affine.coeffs):
                    yield affine, i

    def get_weights(self) -> np.ndarray:
        return np.array([a.const if i is None else a.coeffs[i] for a, i in self._slots()], dtype=float)

    def with_weights(self, weights: np.ndarray) -> "CircuitSpec":
        """Same architecture with different weight values."""
        new = copy.deepcopy(self)
        slots = list(new._slots())
        if len(slots) != len(weights):
            raise ValueError(f"expected {len(slots)} weights, got {len(weights)}")
        for (affine, i), w in zip(slots, weights):
            if i is None:
                affine.const = float(w)
            else:
                affine.coeffs[i] = float(w)
        return new

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for g in self.gates:
            counts[g.name] = counts.get(g.name, 0) + 1
        return {
            "n_qubits": self.n_qubits,
            "state_prep": self.state_prep,
            "n_gates": len(self.gates),
            "n_two_qubit_gates": sum(1 for g in self.gates if len(g.qubits) >= 2),
            "n_weights": len(self.get_weights()),
            "n_data_dependent_angles": sum(1 for g in self.gates for a in g.params if a.coeffs),
            "features_used": sorted({i for g in self.gates for a in g.params for i in a.coeffs}),
            "gate_counts": counts,
            "readout": self.readout.kind,
        }


@dataclass
class Limits:
    max_qubits: int = 8
    max_gates: int = 200
    # Max number of distinct x[i] in one angle. 1 = plain angle encoding (one pixel per
    # rotation); None = any linear combination of pixels (lets a single rotation act as a
    # classical linear classifier, so treat results under None with that in mind).
    max_features_per_angle: int | None = 1
    allow_amplitude_encoding: bool = True
    # Structural requirements for the "structured" condition: they rule out the single-qubit
    # rotation chain (a linear classifier) that models otherwise converge to.
    min_qubits: int = 1
    require_entangling: bool = False
    forbid_repeated_rotation: bool = False  # no two consecutive same-axis rotations on one qubit


def _qubit_list(value, n_qubits: int, where: str) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(q, int) and not isinstance(q, bool) for q in value):
        raise SpecError(f"{where}: 'qubits' must be a list of integers")
    if len(set(value)) != len(value):
        raise SpecError(f"{where}: qubits must be distinct, got {value}")
    for q in value:
        if not 0 <= q < n_qubits:
            raise SpecError(f"{where}: qubit {q} out of range 0..{n_qubits - 1}")
    return value


def _class_label(key, class_labels: list[int]) -> int:
    try:
        label = int(key)
    except (TypeError, ValueError):
        raise SpecError(f"readout: class key {key!r} is not one of {class_labels}") from None
    if label not in class_labels:
        raise SpecError(f"readout: class key {key!r} is not one of {class_labels}")
    return label


def parse_spec(obj: dict, class_labels: list[int], limits: Limits) -> CircuitSpec:
    """Validate the LLM's JSON object and build a CircuitSpec. Raises SpecError."""
    if not isinstance(obj, dict):
        raise SpecError("top level must be a JSON object")

    n_qubits = obj.get("n_qubits")
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise SpecError("'n_qubits' must be a positive integer")
    if n_qubits > limits.max_qubits:
        raise SpecError(f"n_qubits={n_qubits} exceeds the limit of {limits.max_qubits}")
    if n_qubits < limits.min_qubits:
        raise SpecError(f"n_qubits={n_qubits} is below the required minimum of {limits.min_qubits}")

    prep = obj.get("state_prep") or {"type": "zero"}
    if isinstance(prep, str):
        prep = {"type": prep}
    if not isinstance(prep, dict):
        raise SpecError("'state_prep' must be an object")
    prep_type = prep.get("type", "zero")
    amplitude_qubits: list[int] = []
    if prep_type == "amplitude":
        if not limits.allow_amplitude_encoding:
            raise SpecError("amplitude encoding is not allowed in this experiment; use state_prep type 'zero'")
        amplitude_qubits = _qubit_list(prep.get("qubits"), n_qubits, "state_prep")
        if 2 ** len(amplitude_qubits) != N_FEATURES:
            raise SpecError(f"state_prep: amplitude encoding needs exactly 6 qubits, got {len(amplitude_qubits)}")
    elif prep_type != "zero":
        raise SpecError(f"state_prep type must be 'zero' or 'amplitude', got {prep_type!r}")

    raw_gates = obj.get("gates")
    if not isinstance(raw_gates, list):
        raise SpecError("'gates' must be a list")
    if len(raw_gates) > limits.max_gates:
        raise SpecError(f"{len(raw_gates)} gates exceeds the limit of {limits.max_gates}")
    gates: list[Gate] = []
    for k, g in enumerate(raw_gates):
        where = f"gates[{k}]"
        if not isinstance(g, dict):
            raise SpecError(f"{where}: each gate must be an object")
        name = str(g.get("gate", "")).lower()
        name = ALIASES.get(name, name)
        if name not in GATES:
            raise SpecError(f"{where}: unknown gate {g.get('gate')!r}; allowed: {', '.join(GATES)}")
        n_q, n_p = GATES[name]
        qubits = _qubit_list(g.get("qubits"), n_qubits, where)
        if len(qubits) != n_q:
            raise SpecError(f"{where}: gate '{name}' acts on {n_q} qubit(s), got {len(qubits)}")
        raw_params = g.get("params") or []
        if not isinstance(raw_params, list):
            raw_params = [raw_params]
        if len(raw_params) != n_p:
            raise SpecError(f"{where}: gate '{name}' takes {n_p} parameter(s), got {len(raw_params)}")
        params = []
        for p in raw_params:
            try:
                affine = parse_angle(p)
            except SpecError as e:
                raise SpecError(f"{where}: {e}") from None
            limit = limits.max_features_per_angle
            if limit is not None and len(affine.coeffs) > limit:
                raise SpecError(
                    f"{where}: angle {p!r} uses {len(affine.coeffs)} features; "
                    f"at most {limit} feature(s) per angle is allowed"
                )
            params.append(affine)
        gates.append(Gate(name, qubits, params))

    if limits.require_entangling and not any(len(g.qubits) >= 2 for g in gates):
        raise SpecError("the circuit must contain at least one two-qubit gate (cx, cz, cy, swap, crx, cry, crz, cp, rxx, ryy, rzz, ccx)")
    if limits.forbid_repeated_rotation:
        last: dict[int, str] = {}
        for k, g in enumerate(gates):
            for q in g.qubits:
                if g.name in ("rx", "ry", "rz") and last.get(q) == g.name:
                    raise SpecError(
                        f"gates[{k}]: two consecutive {g.name} rotations on qubit {q} with no other gate on that "
                        f"qubit in between are not allowed (they merge into one rotation); use a different axis "
                        f"or put another gate between them"
                    )
                last[q] = g.name

    ro = obj.get("readout")
    if not isinstance(ro, dict):
        raise SpecError("'readout' must be an object")
    kind = ro.get("type")
    raw_classes = ro.get("classes")
    if not isinstance(raw_classes, dict):
        raise SpecError("readout: 'classes' must be an object keyed by class label")
    if kind == "bitstrings":
        ro_qubits = _qubit_list(ro.get("qubits"), n_qubits, "readout")
        if not ro_qubits:
            raise SpecError("readout: 'qubits' must not be empty")
        classes: dict = {}
        seen: dict[str, int] = {}
        for key, strings in raw_classes.items():
            label = _class_label(key, class_labels)
            if isinstance(strings, str):
                strings = [strings]
            if not isinstance(strings, list) or not strings:
                raise SpecError(f"readout: class {label} needs a non-empty list of bitstrings")
            for s in strings:
                if not isinstance(s, str) or not re.fullmatch(f"[01]{{{len(ro_qubits)}}}", s):
                    raise SpecError(f"readout: bitstring {s!r} must be {len(ro_qubits)} characters of 0/1")
                if s in seen:
                    raise SpecError(f"readout: bitstring {s!r} is assigned to more than one class")
                seen[s] = label
            classes[label] = list(strings)
        readout = Readout("bitstrings", ro_qubits, classes)
    elif kind == "qubit_per_class":
        classes = {}
        for key, q in raw_classes.items():
            label = _class_label(key, class_labels)
            if not isinstance(q, int) or isinstance(q, bool) or not 0 <= q < n_qubits:
                raise SpecError(f"readout: class {label} must map to a qubit index 0..{n_qubits - 1}")
            classes[label] = q
        if len(set(classes.values())) != len(classes):
            raise SpecError("readout: each class needs its own qubit")
        readout = Readout("qubit_per_class", sorted(classes.values()), classes)
    else:
        raise SpecError(f"readout type must be 'bitstrings' or 'qubit_per_class', got {kind!r}")
    missing = [c for c in class_labels if c not in readout.classes]
    if missing:
        raise SpecError(f"readout: no outcome assigned to class(es) {missing}")

    return CircuitSpec(n_qubits, prep_type, amplitude_qubits, gates, readout, str(obj.get("reasoning", "")))


def extract_json(text: str) -> dict:
    """Pull the circuit JSON object out of a model reply (last fenced block, else outermost braces)."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)[::-1]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    error = "no JSON object found in the reply"
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            error = f"invalid JSON: {e}"
    raise SpecError(error)
