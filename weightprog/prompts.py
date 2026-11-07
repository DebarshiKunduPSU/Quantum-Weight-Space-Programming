"""Prompt construction: what the model is told and shown."""

from __future__ import annotations

import numpy as np

from . import data
from .providers import Message
from .spec import Limits

SYSTEM_TEMPLATE = """\
You are an expert in quantum machine learning. You will hand-write a complete parameterized \
quantum circuit (PQC) classifier: the data encoding, the gate layout, every numeric gate angle, \
and the readout. No optimizer or training loop runs afterwards. Your circuit is simulated exactly \
as written and scored on held-out images, so the numbers you write are the final weights.

## Input features
Each image is 8x8 grayscale pixels. Feature x[i] is the pixel in row r, column c, with \
i = 8*r + c (row 0 is the top row, column 0 the leftmost column), scaled to [0, 1]: \
0 is black background and 1 is the brightest ink.

## Circuit format
Describe the circuit as one JSON object:

```json
{{
  "reasoning": "a few sentences on the design idea",
  "n_qubits": 2,
  "state_prep": {{"type": "zero"}},
  "gates": [
    {{"gate": "ry", "qubits": [0], "params": ["pi * x[27]"]}},
    {{"gate": "cx", "qubits": [0, 1]}},
    {{"gate": "rz", "qubits": [1], "params": [0.7]}}
  ],
  "readout": {{"type": "bitstrings", "qubits": [1], "classes": {{"{c0}": ["0"], "{c1}": ["1"]}}}}
}}
```

The example only illustrates the syntax; it is not a suggested design. Gates are applied in \
list order. Limits: at most {max_qubits} qubits and {max_gates} gates.{structure_rules}

### Gates
- No parameters: h, x, y, z, s, sdg, t, tdg, sx (1 qubit); cx, cy, cz, swap (2 qubits); ccx (3 qubits).
- One angle: rx, ry, rz, p (1 qubit); crx, cry, crz, cp, rxx, ryy, rzz (2 qubits).
- Three angles: u3 (theta, phi, lambda).

Control qubits come first in "qubits". Definitions are the standard Qiskit ones: \
rx(t) = exp(-i t X/2), ry(t) = exp(-i t Y/2), rz(t) = exp(-i t Z/2), p(t) = diag(1, e^(i t)), \
rzz(t) = exp(-i t Z⊗Z/2) and likewise rxx, ryy.

### Angles
An angle is a number or a string expression built from numbers, pi, features x[i], + - * / \
and parentheses. It must be affine in the features, c + w1*x[i1] + w2*x[i2] + ... . \
{feature_rule} A constant angle is a fixed weight; an angle containing x[i] encodes data. \
A feature may appear in any number of gates.

### State preparation
- {{"type": "zero"}}: every qubit starts in |0>.
{amplitude_rule}
### Readout
Class scores come from computational-basis measurement probabilities ({shots_rule}). \
The predicted class is the one with the highest score. Choose one readout:
- {{"type": "bitstrings", "qubits": [a, b, ...], "classes": {{"<label>": ["01", ...], ...}}}}: \
the listed qubits are measured, and character k of a bitstring is the outcome of the k-th listed \
qubit. A class's score is the total probability of its bitstrings. A bitstring may belong to at \
most one class; unassigned bitstrings count for no class.
- {{"type": "qubit_per_class", "classes": {{"<label>": <qubit>, ...}}}}: a class's score is the \
probability that its qubit is measured as 1.

Every class needs an entry in "classes".
"""

_AMPLITUDE_RULE = """\
- {"type": "amplitude", "qubits": [q0, q1, q2, q3, q4, q5]}: the six listed qubits start in the \
state sum_i (x[i] / ||x||) |i>, where i is read in binary with q0 as the most significant bit, so \
q0 q1 q2 hold the pixel row and q3 q4 q5 the pixel column. All other qubits start in |0>.
"""


def system_prompt(limits: Limits, shots: int | None, class_labels: list[int]) -> str:
    if limits.max_features_per_angle is None:
        feature_rule = "An angle may combine any number of features."
    elif limits.max_features_per_angle == 1:
        feature_rule = "Each angle may contain at most one feature, i.e. c + w*x[i]."
    else:
        feature_rule = f"Each angle may contain at most {limits.max_features_per_angle} different features."
    rules = []
    if limits.min_qubits > 1:
        rules.append(f"use at least {limits.min_qubits} qubits")
    if limits.require_entangling:
        rules.append("include at least one two-qubit gate")
    if limits.forbid_repeated_rotation:
        rules.append("never apply two consecutive rotations about the same axis (rx, ry or rz) to the same qubit "
                     "with no other gate on that qubit in between, since they merge into one rotation")
    structure_rules = (" Required structure: " + "; ".join(rules) + ".") if rules else ""
    return SYSTEM_TEMPLATE.format(
        structure_rules=structure_rules,
        c0=class_labels[0],
        c1=class_labels[1],
        max_qubits=limits.max_qubits,
        max_gates=limits.max_gates,
        feature_rule=feature_rule,
        amplitude_rule=_AMPLITUDE_RULE if limits.allow_amplitude_encoding else "",
        shots_rule="computed exactly" if not shots else f"estimated from {shots} shots per image",
    )


def user_message(
    class_labels: list[int],
    X_support: np.ndarray,
    y_support: np.ndarray,
    modality: str = "image",
    image_layout: str = "grid",
    cell_px: int = 24,
) -> Message:
    """The task statement plus the labelled support examples as images, pixel grids, or both."""
    if modality not in ("image", "text", "both"):
        raise ValueError(f"modality must be image, text or both, got {modality!r}")
    if image_layout not in ("grid", "individual"):
        raise ValueError(f"image_layout must be grid or individual, got {image_layout!r}")
    show_image, show_text = modality in ("image", "both"), modality in ("text", "both")
    shots = int(np.sum(y_support == class_labels[0]))
    labels = ", ".join(str(c) for c in class_labels)

    intro = [
        f"Task: classify 8x8 images of handwritten digits into the classes {labels}.",
        f"Below are {shots} labelled training example(s) per class.",
    ]
    if show_image:
        intro.append(
            "In the images every pixel is drawn as a square whose brightness is its feature value "
            "(black = 0, white = 1)."
            + (" Examples of one class are shown side by side, separated by blue bars." if image_layout == "grid" else "")
        )
    if show_text:
        intro.append("Pixel grids are given as integers 0..16, one image row per line; x[i] is the integer divided by 16.")
    parts: list = [" ".join(intro)]

    for label in class_labels:
        xs = X_support[y_support == label]
        if show_image and image_layout == "grid":
            parts.append(f"Class {label}, {len(xs)} example(s), left to right:")
            parts.append(data.render_row(xs, cell_px))
        for k, x in enumerate(xs, start=1):
            if show_image and image_layout == "individual":
                parts.append(f"Class {label}, example {k}:")
                parts.append(data.render_digit(x, cell_px))
            if show_text:
                parts.append(f"Class {label}, example {k}, pixel grid:\n{data.pixel_grid_text(x)}")

    parts.append(
        "Work out what distinguishes these classes, then design the circuit and choose its weights. "
        "It will be scored by classification accuracy on held-out test images of the same classes. "
        "End your reply with the complete circuit as a single JSON object in a ```json code block."
    )
    return Message("user", parts)


def repair_message(error: str) -> Message:
    return Message(
        "user",
        [
            f"That circuit could not be used: {error}\n"
            "Reply with the complete corrected circuit as a single JSON object in a ```json code block."
        ],
    )
