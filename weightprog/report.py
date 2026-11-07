"""Aggregate results into one table per condition and model.

    python -m weightprog.report                 # everything under results/
    python -m weightprog.report results/myrun   # one run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .run import ROOT

COLUMNS = [
    "run", "model", "classes", "shots", "modality", "feat/angle", "valid",
    "LLM acc", "chance", "random w", "trained", "LLM+finetune", "logreg(sup)", "logreg(full)",
    "qubits", "gates", "weights", "out tok",
]


def _mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


def _mean(values: list[float], fmt: str = ".1f") -> str:
    return format(float(np.mean(values)), fmt) if values else "-"


def collect(root: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("result.json")):
        r = json.loads(path.read_text())
        if r.get("status") == "error":
            continue
        key = (
            r.get("run_name", path.parent.parent.parent.name),
            r.get("model", path.parent.parent.name),
            ",".join(map(str, r["classes"])),
            r["shots_per_class"],
            r["modality"],
            "any" if r.get("max_features_per_angle") is None else r["max_features_per_angle"],
        )
        groups[key].append(r)

    rows = []
    for key, results in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        ok = [r for r in results if r["status"] == "ok"]

        def baseline(name: str, field: str | None = None) -> list[float]:
            values = [r["baselines"].get(name) for r in ok]
            if field:
                values = [v.get(field) if isinstance(v, dict) else None for v in values]
            return [v for v in values if v is not None]

        rows.append(dict(zip(COLUMNS, [
            *key,
            f"{len(ok)}/{len(results)}",
            _mean_std([r["llm"]["test"]["accuracy"] for r in ok]),
            f"{results[0]['chance']:.3f}",
            _mean_std(baseline("random_weights", "mean")),
            _mean_std(baseline("trained_on_support", "test_accuracy")),
            _mean_std(baseline("llm_finetuned_on_support", "test_accuracy")),
            _mean_std(baseline("logreg_on_support")),
            _mean_std(baseline("logreg_on_full")),
            _mean([r["circuit"]["n_qubits"] for r in ok]),
            _mean([r["circuit"]["n_gates"] for r in ok]),
            _mean([r["circuit"]["n_weights"] for r in ok]),
            _mean([r["output_tokens"] for r in results], ".0f"),
        ])))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", type=Path, default=ROOT / "results")
    ap.add_argument("--csv", type=Path, help="also write the table to this CSV file")
    args = ap.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print(f"no results under {args.root}")
        return 1
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLUMNS}
    print("| " + " | ".join(c.ljust(widths[c]) for c in COLUMNS) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in COLUMNS) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in COLUMNS) + " |")
    print(
        "\nAccuracies are on the held-out test split, mean ± std over trials with a valid circuit. "
        "'valid' = usable circuits / trials.\nrandom w = LLM's architecture with random weights; "
        "trained = optimizer on the support set from a random start;\nLLM+finetune = optimizer started "
        "from the LLM's weights; logreg = logistic regression on raw pixels."
    )
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
