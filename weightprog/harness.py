"""Shared machinery: the ask-parse-repair loop, the trial runner with resume, the weights-only
reply format, finite-difference optimization for baselines, and a generic results table."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.optimize import minimize

from .providers import FatalProviderError, LLMResponse, Message, make_provider, resolve_model
from .spec import SpecError, extract_json

ROOT = Path(__file__).resolve().parent.parent
FINISHED = ("ok", "invalid", "refused")


def repair_message(error: str) -> Message:
    return Message("user", [f"That reply could not be used: {error}\nReply again with the complete corrected JSON object in a ```json code block."])


def query(provider, system: str, user: Message, parse: Callable[[str], Any], max_repairs: int, out_dir: Path):
    """Ask, parse, and send validation errors back up to ``max_repairs`` times.

    Returns (parsed object or None, list of attempt records, status). Raw replies are saved
    as response_<k>.txt. The model only ever sees the validation error, never a score.
    """
    messages, attempts = [user], []
    for attempt in range(1 + max_repairs):
        resp: LLMResponse = provider.complete(system, messages)
        (out_dir / f"response_{attempt}.txt").write_text(resp.text)
        record = {"model": resp.model, "stop_reason": resp.stop_reason, "input_tokens": resp.input_tokens,
                  "output_tokens": resp.output_tokens, "latency_s": round(resp.latency_s, 1), **resp.extra}
        attempts.append(record)
        if resp.stop_reason == "refusal":
            return None, attempts, "refused"
        try:
            return parse(resp.text), attempts, "ok"
        except SpecError as e:
            record["error"] = str(e)
            messages += [Message("assistant", [resp.text or "(empty reply)"]), repair_message(str(e))]
    return None, attempts, "invalid"


def parse_weights(text: str, n: int) -> np.ndarray:
    """The weights-only reply: {"reasoning": "...", "weights": [n numbers]}."""
    obj = extract_json(text)
    if not isinstance(obj, dict) or "weights" not in obj:
        raise SpecError('the JSON object needs a "weights" key')
    w = obj["weights"]
    if not isinstance(w, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in w):
        raise SpecError('"weights" must be a flat list of numbers')
    if len(w) != n:
        raise SpecError(f'"weights" must have exactly {n} numbers, got {len(w)}')
    if not all(math.isfinite(v) for v in w):
        raise SpecError('"weights" must be finite numbers')
    return np.array(w, dtype=float)


def weights_format(n: int) -> str:
    return (
        "Reply with one JSON object in a ```json code block:\n"
        '{"reasoning": "<a few sentences on how you chose the values>", "weights": [theta[0], theta[1], ...]}\n'
        f"with exactly {n} number(s), in radians, in index order. You may reason before the JSON block."
    )


GATE_CONVENTIONS = (
    "Gate definitions are the standard Qiskit ones: rx(t) = exp(-i t X/2), ry(t) = exp(-i t Y/2), "
    "rz(t) = exp(-i t Z/2), rzz(t) = exp(-i t Z⊗Z/2), h = Hadamard, x = Pauli X, cx = CNOT with the "
    "control listed first. Gates are applied in the order listed. Every qubit starts in |0>."
)


def optimize(objective: Callable[[np.ndarray], float], init: np.ndarray, max_iter: int = 200, max_seconds: float = 60.0) -> dict:
    """L-BFGS-B with finite differences; returns the best point seen and the best-so-far trace.

    The trace lets a warm start (LLM weights) be compared with a random start by the number
    of objective evaluations needed to reach a given value.
    """
    best = {"value": np.inf, "x": np.array(init, dtype=float)}
    trace: list[float] = []
    start = time.time()

    class _OutOfTime(Exception):
        pass

    def f(x):
        if time.time() - start > max_seconds:
            raise _OutOfTime
        v = float(objective(x))
        if v < best["value"]:
            best["value"], best["x"] = v, x.copy()
        trace.append(best["value"])
        return v

    try:
        minimize(f, best["x"], method="L-BFGS-B", options={"maxiter": max_iter})
    except _OutOfTime:
        pass
    return {"value": best["value"], "x": best["x"], "nfev": len(trace), "trace": trace,
            "seconds": round(time.time() - start, 1)}


def evals_to_reach(trace: list[float], target: float) -> int | None:
    """Number of objective evaluations until the best-so-far value is <= target."""
    for k, v in enumerate(trace, start=1):
        if v <= target:
            return k
    return None


@dataclass
class Case:
    """One problem instance: a prompt, how to parse the reply, and how to score it."""

    id: str
    system: str
    user: Message | Callable[[int], Message]  # fixed prompt, or one built per trial index
    parse: Callable[[str], Any]
    score: Callable[[Any, np.random.Generator, int], dict]  # (obj, rng, trial) -> {"summary": {name: number}, ...}
    meta: dict = field(default_factory=dict)

    def prompt(self, trial: int) -> Message:
        return self.user(trial) if callable(self.user) else self.user


def load_base_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _save_prompt(out_dir: Path, system: str, user: Message) -> None:
    lines, n_img = [f"=== SYSTEM ===\n{system}\n=== USER ==="], 0
    for part in user.parts:
        if isinstance(part, str):
            lines.append(part)
        else:
            (out_dir / f"image_{n_img:02d}.png").write_bytes(part)
            lines.append(f"[image: image_{n_img:02d}.png]")
            n_img += 1
    (out_dir / "prompt.txt").write_text("\n\n".join(lines) + "\n")


def run_cases(experiment: str, cases: list[Case], model_name: str, cfg: dict, trials: int, dry_run: bool = False) -> int:
    """Run every case for ``trials`` repeats; results/<experiment>/<model>/<case>/trial_k/."""
    entry = resolve_model(model_name, cfg.get("models") or {})
    out_root = ROOT / "results" / experiment / _slug(model_name)
    out_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        for case in cases:
            d = out_root / _slug(case.id) / "trial_000"
            d.mkdir(parents=True, exist_ok=True)
            _save_prompt(d, case.system, case.prompt(0))
        print(f"{len(cases)} prompt(s) written under {out_root}")
        return 0
    try:
        provider = make_provider(entry, cfg["llm"], [0, 1])
    except Exception as e:
        print(f"cannot set up {entry['provider']}:{entry['model']}: {e}", file=sys.stderr)
        return 1
    (out_root / "config.yaml").write_text(yaml.safe_dump({"experiment": experiment, "model": model_name,
                                                         "resolved_model": entry, "llm": cfg["llm"],
                                                         "seed": cfg["seed"], "trials": trials}, sort_keys=False))
    print(f"{experiment}: {entry['provider']}:{entry['model']} | {len(cases)} case(s) x {trials} trial(s)")

    for case in cases:
        for trial in range(trials):
            out_dir = out_root / _slug(case.id) / f"trial_{trial:03d}"
            path = out_dir / "result.json"
            if path.exists() and json.loads(path.read_text()).get("status") in FINISHED:
                print(f"{case.id} trial {trial}: already done, skipping")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            user = case.prompt(trial)
            _save_prompt(out_dir, case.system, user)
            result = {"experiment": experiment, "model": model_name, "case": case.id, "trial": trial, "meta": case.meta}
            try:
                obj, attempts, status = query(provider, case.system, user, case.parse, cfg["llm"]["max_repair_attempts"], out_dir)
                result.update(status=status, attempts=attempts, n_attempts=len(attempts),
                              model_reported=attempts[-1]["model"],
                              input_tokens=sum(a["input_tokens"] for a in attempts),
                              output_tokens=sum(a["output_tokens"] for a in attempts))
                if status == "ok":
                    rng = np.random.default_rng([cfg["seed"], trial, zlib.crc32(case.id.encode())])
                    result.update(case.score(obj, rng, trial))
            except FatalProviderError as e:
                print(f"stopping: {e}", file=sys.stderr)
                return 1
            except Exception as e:
                result.update(status="error", error=f"{type(e).__name__}: {e}")
                (out_dir / "error.txt").write_text(traceback.format_exc())
            path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
            if result["status"] == "ok":
                summary = ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in result["summary"].items())
                print(f"{case.id} trial {trial}: {summary} ({result['n_attempts']} attempt(s))")
            else:
                print(f"{case.id} trial {trial}: {result['status']} {result.get('error', '')}")
    print(f"results in {out_root}\ntable: python -m weightprog.harness report results/{experiment}")
    return 0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not serializable: {type(o)}")


def report(root: Path, csv_path: Path | None = None) -> int:
    """Mean ± std of every summary field, per (model, case)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("result.json")):
        r = json.loads(path.read_text())
        if r.get("status") != "error":
            groups[(r["model"], r["case"])].append(r)
    if not groups:
        print(f"no results under {root}")
        return 1
    fields = sorted({k for rs in groups.values() for r in rs if r["status"] == "ok" for k in r["summary"]})
    columns = ["model", "case", "valid", *fields]
    rows = []
    for (model, case), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        row = {"model": model, "case": case, "valid": f"{len(ok)}/{len(rs)}"}
        for f in fields:
            vals = [r["summary"][f] for r in ok if r["summary"].get(f) is not None]
            row[f] = "-" if not vals else (f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}")
        rows.append(row)
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in columns) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in columns) + " |")
    if csv_path:
        import csv

        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {csv_path}")
    return 0


def standard_cli(experiment: str, build_cases: Callable[[], list[Case]], doc: str) -> int:
    """CLI shared by the weights-only experiments: --model, --trials, --cases, --dry-run."""
    from .run import load_env

    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="alias from configs/default.yaml 'models', or provider:model_id")
    ap.add_argument("--trials", type=int, default=3, help="repeats per case (default 3)")
    ap.add_argument("--cases", help="only run cases whose id contains this substring")
    ap.add_argument("--list", action="store_true", help="list case ids and exit")
    ap.add_argument("--dry-run", action="store_true", help="write prompts, call no model")
    args = ap.parse_args()
    load_env()
    cfg = load_base_config()
    cases = build_cases()
    if args.cases:
        cases = [c for c in cases if args.cases in c.id]
    if args.list:
        for c in cases:
            print(c.id)
        return 0
    return run_cases(experiment, cases, args.model, cfg, args.trials, args.dry_run)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        sys.exit(report(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None))
    print("usage: python -m weightprog.harness report <results dir> [table.csv]")
