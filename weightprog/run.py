"""Run the experiment: show a model a few labelled digits, get a PQC back, score it.

    python -m weightprog.run --model gpt-6-astra
    python -m weightprog.run --model fable --set task.classes=[3,8] --set prompt.shots_per_class=10
    python -m weightprog.run --model mock:test --dry-run     # write the prompt, call nothing
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

from . import data, evaluate, prompts
from .providers import FatalProviderError, Message, make_provider, resolve_model
from .spec import Limits, SpecError, extract_json, parse_spec

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
FINISHED = ("ok", "invalid", "refused")  # trial outcomes that are not retried on resume


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: Path | None, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path and path.resolve() != DEFAULT_CONFIG:
        cfg = _merge(cfg, yaml.safe_load(path.read_text()) or {})
    for item in overrides:
        dotted, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        node, keys = cfg, dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _save_prompt(trial_dir: Path, system: str, user: Message) -> None:
    lines, n_images = [f"=== SYSTEM ===\n{system}\n=== USER ==="], 0
    for part in user.parts:
        if isinstance(part, str):
            lines.append(part)
        else:
            name = f"support_{n_images:02d}.png"
            (trial_dir / name).write_bytes(part)
            lines.append(f"[image: {name}]")
            n_images += 1
    (trial_dir / "prompt.txt").write_text("\n\n".join(lines) + "\n")


def run_trial(cfg: dict, task: data.Task, provider, trial: int, trial_dir: Path, dry_run: bool = False) -> dict | None:
    labels = task.class_labels
    p_cfg, c_cfg, b_cfg = cfg["prompt"], cfg["circuit"], cfg["baselines"]
    limits = Limits(
        c_cfg["max_qubits"], c_cfg["max_gates"], c_cfg["max_features_per_angle"], c_cfg["allow_amplitude_encoding"],
        c_cfg.get("min_qubits", 1), c_cfg.get("require_entangling", False), c_cfg.get("forbid_repeated_rotation", False),
    )
    shots = c_cfg.get("shots")
    # Seeded by (seed, trial) only, so every model sees the same support set in a given trial.
    rng = np.random.default_rng([cfg["seed"], trial])
    X_sup, y_sup = data.sample_support(task, p_cfg["shots_per_class"], rng)

    system = prompts.system_prompt(limits, shots, labels)
    user = prompts.user_message(labels, X_sup, y_sup, p_cfg["modality"], p_cfg["image_layout"], p_cfg["cell_px"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    _save_prompt(trial_dir, system, user)
    if dry_run:
        return None

    result = {
        "trial": trial,
        "classes": labels,
        "shots_per_class": p_cfg["shots_per_class"],
        "modality": p_cfg["modality"],
        "max_features_per_angle": limits.max_features_per_angle,
        "structured": bool(limits.require_entangling or limits.forbid_repeated_rotation or limits.min_qubits > 1),
        "n_test": int(len(task.y_test)),
        "chance": 1.0 / len(labels),
        "status": "invalid",
        "attempts": [],
    }
    messages, spec, raw = [user], None, None
    for attempt in range(1 + cfg["llm"]["max_repair_attempts"]):
        resp = provider.complete(system, messages)
        (trial_dir / f"response_{attempt}.txt").write_text(resp.text)
        record = {
            "model": resp.model,
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_s": round(resp.latency_s, 1),
            **resp.extra,
        }
        result["attempts"].append(record)
        if resp.stop_reason == "refusal":
            result["status"] = "refused"
            break
        try:
            raw = extract_json(resp.text)
            spec = parse_spec(raw, labels, limits)
            break
        except SpecError as e:
            record["error"] = str(e)
            messages += [Message("assistant", [resp.text or "(empty reply)"]), prompts.repair_message(str(e))]

    result["model_reported"] = result["attempts"][-1]["model"]
    result["n_attempts"] = len(result["attempts"])
    result["input_tokens"] = sum(a["input_tokens"] for a in result["attempts"])
    result["output_tokens"] = sum(a["output_tokens"] for a in result["attempts"])
    if spec is None:
        return result

    (trial_dir / "circuit.json").write_text(json.dumps(raw, indent=2) + "\n")
    result["status"] = "ok"
    result["circuit"] = spec.summary()
    result["llm"] = {
        "test": evaluate.evaluate(spec, task.X_test, task.y_test, labels, shots, rng),
        "support": evaluate.evaluate(spec, X_sup, y_sup, labels, shots, rng),
    }

    baselines: dict = {}
    n_weights = len(spec.get_weights())
    if b_cfg["random_weight_draws"]:
        baselines["random_weights"] = evaluate.random_weight_baseline(
            spec, task.X_test, task.y_test, labels, b_cfg["random_weight_draws"], rng
        )
    train_args = (task.X_test, task.y_test, labels)
    budget = (b_cfg["train_max_iter"], b_cfg["train_max_seconds"])
    if b_cfg["train_on_support"]:
        # Same architecture, same examples the LLM saw, but an optimizer picks the weights.
        baselines["trained_on_support"] = evaluate.trained_baseline(
            spec, X_sup, y_sup, *train_args, evaluate.random_weights(n_weights, rng), *budget
        )
    if b_cfg["finetune_llm_on_support"]:
        # Optimizer started from the LLM's weights: how far were they from a local optimum?
        baselines["llm_finetuned_on_support"] = evaluate.trained_baseline(
            spec, X_sup, y_sup, *train_args, spec.get_weights(), *budget
        )
    if b_cfg["train_on_full"]:
        # Architecture ceiling: optimizer with (a capped subset of) the whole train split.
        n_fit = min(len(task.y_train), b_cfg["train_full_max_samples"])
        idx = rng.choice(len(task.y_train), size=n_fit, replace=False)
        baselines["trained_on_full"] = evaluate.trained_baseline(
            spec, task.X_train[idx], task.y_train[idx], *train_args, evaluate.random_weights(n_weights, rng), *budget
        )
    if b_cfg["logistic_regression"]:
        baselines["logreg_on_support"] = evaluate.logistic_baseline(X_sup, y_sup, task.X_test, task.y_test)
        baselines["logreg_on_full"] = evaluate.logistic_baseline(task.X_train, task.y_train, task.X_test, task.y_test)
    result["baselines"] = baselines
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML file merged over configs/default.yaml")
    ap.add_argument("--model", help="alias from the config's 'models' table, or provider:model_id")
    ap.add_argument("--run-name", help="results go to results/<run-name>/<model>/")
    ap.add_argument("--trials", type=int, help="number of trials (independent support sets)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. task.classes=[3,8]")
    ap.add_argument("--dry-run", action="store_true", help="write the trial-0 prompt and images, call no model")
    args = ap.parse_args(argv)

    load_env()
    cfg = load_config(args.config, args.set)
    for key, value in (("model", args.model), ("run_name", args.run_name), ("trials", args.trials)):
        if value is not None:
            cfg[key] = value

    task = data.load_task(cfg["task"]["classes"], cfg["task"]["test_fraction"], cfg["task"]["split_seed"])
    if len(task.class_labels) < 2:
        raise SystemExit("task.classes needs at least two classes")
    entry = resolve_model(cfg["model"], cfg.get("models") or {})
    out_dir = ROOT / "results" / _slug(cfg["run_name"]) / _slug(cfg["model"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        run_trial(cfg, task, None, 0, out_dir / "trial_000", dry_run=True)
        print(f"prompt and images written to {out_dir / 'trial_000'}")
        return 0

    try:
        provider = make_provider(entry, cfg["llm"], task.class_labels)
    except Exception as e:  # typically a missing API key
        print(f"cannot set up {entry['provider']}:{entry['model']}: {e}\n"
              f"Put the key in {ROOT / '.env'} (see .env.example) or export it.", file=sys.stderr)
        return 1
    (out_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "resolved_model": entry}, sort_keys=False))
    print(f"model {entry['provider']}:{entry['model']} | classes {task.class_labels} | "
          f"{cfg['prompt']['shots_per_class']} shot(s)/class | {len(task.y_test)} test images")

    accs = []
    for trial in range(cfg["trials"]):
        trial_dir = out_dir / f"trial_{trial:03d}"
        result_path = trial_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("status") in FINISHED:
                print(f"trial {trial}: already done ({result['status']}), skipping")
                if result["status"] == "ok":
                    accs.append(result["llm"]["test"]["accuracy"])
                continue
        try:
            result = run_trial(cfg, task, provider, trial, trial_dir)
        except FatalProviderError as e:
            print(f"stopping: {e}", file=sys.stderr)
            return 1
        except Exception as e:  # transient API failure etc.: record it, keep going, retry on the next run
            trial_dir.mkdir(parents=True, exist_ok=True)
            result = {"trial": trial, "status": "error", "error": f"{type(e).__name__}: {e}"}
            (trial_dir / "error.txt").write_text(traceback.format_exc())
        result["model"] = cfg["model"]
        result["run_name"] = cfg["run_name"]
        result_path.write_text(json.dumps(result, indent=2) + "\n")

        if result["status"] != "ok":
            print(f"trial {trial}: {result['status']} {result.get('error', '')}")
            continue
        acc = result["llm"]["test"]["accuracy"]
        accs.append(acc)
        b, c = result["baselines"], result["circuit"]
        parts = [f"trial {trial}: LLM test acc {acc:.3f}"]
        if "mean" in b.get("random_weights", {}):
            parts.append(f"random weights {b['random_weights']['mean']:.3f}")
        if "test_accuracy" in b.get("trained_on_support", {}):
            parts.append(f"trained {b['trained_on_support']['test_accuracy']:.3f}")
        if "logreg_on_support" in b:
            parts.append(f"logreg {b['logreg_on_support']:.3f}")
        parts.append(f"{c['n_qubits']}q/{c['n_gates']}g/{c['n_weights']}w, {result['n_attempts']} attempt(s)")
        print(" | ".join(parts))

    if accs:
        print(f"\n{len(accs)}/{cfg['trials']} valid circuits; LLM test accuracy "
              f"{np.mean(accs):.3f} ± {np.std(accs):.3f} (chance {1 / len(task.class_labels):.3f})")
    print(f"results in {out_dir}\nsummary table: python -m weightprog.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
