"""Digits data: fixed train/test split, few-shot support sampling, image rendering."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

_BAR_COLOR = (40, 90, 220)  # separator between examples; never a grayscale pixel value


@dataclass
class Task:
    class_labels: list[int]
    X_train: np.ndarray  # (N, 64), pixel / 16 in [0, 1], row-major
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_task(classes: list[int], test_fraction: float = 0.3, split_seed: int = 0) -> Task:
    """The split depends only on (classes, test_fraction, split_seed), so every model and
    trial is scored on the same held-out images."""
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    keep = np.isin(y, classes)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X[keep], y[keep], test_size=test_fraction, random_state=split_seed, stratify=y[keep]
    )
    return Task(sorted(classes), X_tr, y_tr, X_te, y_te)


def sample_support(task: Task, shots_per_class: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Few-shot examples drawn from the train split, grouped by class in label order."""
    idx = []
    for label in task.class_labels:
        pool = np.flatnonzero(task.y_train == label)
        idx.extend(rng.choice(pool, size=shots_per_class, replace=False))
    idx = np.array(idx)
    return task.X_train[idx], task.y_train[idx]


def _digit_image(x: np.ndarray, cell_px: int) -> Image.Image:
    gray = np.round(x.reshape(8, 8) * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").resize((8 * cell_px, 8 * cell_px), Image.NEAREST).convert("RGB")


def render_digit(x: np.ndarray, cell_px: int = 24) -> bytes:
    """One digit as PNG bytes; each pixel is a cell_px square with brightness = x."""
    return _png(_digit_image(x, cell_px))


def render_row(xs: np.ndarray, cell_px: int = 24, bar_px: int = 12) -> bytes:
    """Several digits side by side, separated by coloured bars."""
    side = 8 * cell_px
    width = len(xs) * side + (len(xs) + 1) * bar_px
    sheet = Image.new("RGB", (width, side + 2 * bar_px), _BAR_COLOR)
    for k, x in enumerate(xs):
        sheet.paste(_digit_image(x, cell_px), (bar_px + k * (side + bar_px), bar_px))
    return _png(sheet)


def pixel_grid_text(x: np.ndarray) -> str:
    """The raw 0..16 integer grid, one image row per line."""
    grid = np.round(x.reshape(8, 8) * 16).astype(int)
    return "\n".join(" ".join(f"{v:2d}" for v in row) for row in grid)


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
