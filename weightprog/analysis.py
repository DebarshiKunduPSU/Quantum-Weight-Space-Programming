"""Aggregate every result.json into tidy tables, and render the paper's figures and LaTeX tables.

    python -m weightprog.analysis            # writes paper/figures/*.pdf, paper/tables/*.tex, results/summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .harness import ROOT  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
TAB_DIR = ROOT / "paper" / "tables"

MODEL_LABEL = {"gpt-6-astra": "GPT-6 Astra", "fable": "Claude Fable 5.1"}
MODEL_ORDER = ["gpt-6-astra"]  # models in the figures/tables; override with --models
# Fixed categorical assignment (never cycled): one hue per model, aqua for the optimizer, gray for random.
COLOR = {"gpt-6-astra": "#2a78d6", "fable": "#eb6834", "optimizer": "#1baf7a", "random": "#8a8984", "classical": "#6250d6"}
MARKER = {"gpt-6-astra": "o", "fable": "s"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"

QML_FREE_LABEL = {
    "default": "3 vs 8",
    "pair49": "4 vs 9",
    "all10": "10 classes (3-shot)",
    "shots1": "3 vs 8, 1-shot",
    "shots10": "3 vs 8, 10-shot",
    "text": "3 vs 8, pixels as text",
    "structured": "3 vs 8, structured",
    "structured_all10": "10 classes, structured",
}
QML_FIXED_LABEL = {
    "digits_38_amp_hea_L1": "3 vs 8, amp+HEA L=1 (12 w)",
    "digits_38_amp_hea_L2": "3 vs 8, amp+HEA L=2 (18 w)",
    "digits_all10_amp_hea_L1": "10 classes, amp+HEA L=1 (12 w)",
    "digits_all10_amp_hea_L2": "10 classes, amp+HEA L=2 (18 w)",
    "digits_38_amp_hea_L4": "3 vs 8, amp+HEA L=4 (30 w)",
    "digits_38_amp_hea_L8": "3 vs 8, amp+HEA L=8 (54 w)",
    "digits_all10_amp_hea_L4": "10 classes, amp+HEA L=4 (30 w)",
    "digits_all10_amp_hea_L8": "10 classes, amp+HEA L=8 (54 w)",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_rows(results_dir: Path = ROOT / "results") -> pd.DataFrame:
    """One row per trial with a flat set of metrics; ``experiment`` ∈ qml_free, qml_fixed, vqe, qaoa."""
    rows = []
    for path in sorted(results_dir.rglob("result.json")):
        r = _load(path)
        if r.get("status") == "error":
            continue
        base = {"status": r["status"], "trial": r["trial"], "n_attempts": r.get("n_attempts"),
                "output_tokens": r.get("output_tokens"), "input_tokens": r.get("input_tokens"),
                "latency_s": sum(a.get("latency_s", 0) for a in r.get("attempts", [])), "path": str(path.relative_to(ROOT))}
        if "experiment" in r:  # harness experiments
            row = {**base, "experiment": r["experiment"], "model": r["model"], "case": r["case"], **r.get("summary", {})}
            if r["experiment"] in ("vqe", "qaoa") and r["status"] == "ok":
                opt = r["optimizer_random_init"]
                row["evals_random_init"] = np.mean([e for e in opt["evals_to_target"] if e is not None]) if any(e is not None for e in opt["evals_to_target"]) else np.nan
                row["evals_random_init_hit"] = np.mean([e is not None for e in opt["evals_to_target"]])
                e = r["optimizer_llm_init"]["evals_to_target"]
                row["evals_llm_init"] = e if e is not None else np.nan
                row["evals_llm_init_hit"] = float(e is not None)
            rows.append(row)
        else:  # free-design QML
            row = {**base, "experiment": "qml_free", "model": r["model"], "case": r["run_name"], "chance": r.get("chance")}
            if r["status"] == "ok":
                b, c = r["baselines"], r["circuit"]
                row.update(acc_llm=r["llm"]["test"]["accuracy"], acc_support=r["llm"]["support"]["accuracy"],
                           acc_random=b.get("random_weights", {}).get("mean"),
                           acc_trained_support=b.get("trained_on_support", {}).get("test_accuracy"),
                           acc_llm_finetuned=b.get("llm_finetuned_on_support", {}).get("test_accuracy"),
                           logreg_support=b.get("logreg_on_support"), logreg_full=b.get("logreg_on_full"),
                           n_qubits=c["n_qubits"], n_gates=c["n_gates"], n_weights=c["n_weights"],
                           n_two_qubit=c["n_two_qubit_gates"], state_prep=c["state_prep"])
            rows.append(row)
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """mean/std per (experiment, model, case) over valid trials, plus validity counts."""
    ok = df[df.status == "ok"]
    g = ok.groupby(["experiment", "model", "case"])
    out = g[metrics].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_trials"] = df.groupby(["experiment", "model", "case"]).size()
    out["n_valid"] = g.size()
    return out.reset_index()


# ---------------------------------------------------------------- figures

def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def _style(ax, xlabel: str):
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)


def dot_plot(ax, cases: list[str], labels: list[str], summary: pd.DataFrame, llm_metric: str,
             extras: list[tuple[str, str, str, str]], xlabel: str, xlim=(0, 1)):
    """Rows = cases; x = metric. Filled markers per model (mean ± std over trials); hollow/ticks for baselines.

    extras: (metric, color key, marker, legend label) drawn per model-row in muted form.
    """
    y_positions = {}
    y = 0
    for case in cases:
        for k, model in enumerate(MODEL_ORDER):
            y_positions[(case, model)] = y + (0.18 if k == 0 else -0.18)
        y += 1
    handles = {}
    for (case, model), yy in y_positions.items():
        sub = summary[(summary.case == case) & (summary.model == model)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for metric, ckey, marker, label in extras:
            m = row.get(f"{metric}_mean")
            if m is not None and not np.isnan(m):
                hollow = ckey in ("random", "optimizer") or marker == "v"
                h = ax.scatter(m, yy, marker=marker, s=46, facecolors="white" if hollow else COLOR[ckey],
                               edgecolors=COLOR[ckey], linewidths=1.4, zorder=6 if ckey == "optimizer" else 3)
                handles.setdefault(label, h)
        m, s = row[f"{llm_metric}_mean"], row[f"{llm_metric}_std"]
        if not np.isnan(s):
            ax.plot([m - s, m + s], [yy, yy], color=COLOR[model], linewidth=2, solid_capstyle="round", zorder=4)
        h = ax.scatter(m, yy, marker=MARKER[model], s=64, color=COLOR[model], edgecolors="white", linewidths=1.5, zorder=5)
        handles.setdefault(f"{MODEL_LABEL[model]} (LLM-written)", h)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax.set_ylim(-0.6, len(cases) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    _style(ax, xlabel)
    return handles


def fig_qml(df: pd.DataFrame) -> None:
    metrics = ["acc_llm", "acc_random", "acc_trained_support", "acc_llm_finetuned", "logreg_support", "chance"]
    free = agg(df[df.experiment == "qml_free"], metrics)
    fixed = agg(df[df.experiment == "qml_fixed"], ["acc_llm", "acc_random", "acc_trained_support", "acc_llm_finetuned", "acc_trained_full", "chance"])
    free_cases = [c for c in QML_FREE_LABEL if c in set(free.case)]
    fixed_cases = [c for c in QML_FIXED_LABEL if c in set(fixed.case)]
    if not free_cases and not fixed_cases:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 0.3 * (len(free_cases) + len(fixed_cases)) + 2.0),
                             gridspec_kw={"height_ratios": [max(1, len(free_cases)), max(1, len(fixed_cases))]})
    extras = [("acc_random", "random", "o", "same circuit, random weights"),
              ("acc_trained_support", "optimizer", "^", "same circuit, optimizer on the shown examples"),
              ("logreg_support", "classical", "d", "logistic regression on the shown examples")]
    h1 = dot_plot(axes[0], free_cases, [QML_FREE_LABEL[c] for c in free_cases], free, "acc_llm", extras, "", (0, 1))
    axes[0].set_title("A. Free design: the model writes the circuit and its weights", loc="left", fontsize=10, color=TEXT)
    extras_fixed = [("acc_random", "random", "o", "same circuit, random weights"),
                    ("acc_trained_support", "optimizer", "^", "same circuit, optimizer on the shown examples"),
                    ("acc_trained_full", "optimizer", "v", "same circuit, optimizer on 200 training images")]
    h2 = dot_plot(axes[1], fixed_cases, [QML_FIXED_LABEL[c] for c in fixed_cases], fixed, "acc_llm", extras_fixed, "test accuracy", (0, 1))
    axes[1].set_title("B. Fixed ansatz: the model writes only the weights", loc="left", fontsize=10, color=TEXT)
    for ax, cases, table in ((axes[0], free_cases, free), (axes[1], fixed_cases, fixed)):
        for i, c in enumerate(cases):  # chance level per row
            ch = table[table.case == c]["chance_mean"].iloc[0]
            ax.plot([ch, ch], [i - 0.4, i + 0.4], color=GRID, linewidth=1.5, zorder=1)
    handles = {**h1, **h2}
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "qml")


VQE_LABEL = {
    "tfim_n6_h0.5_hva_p2": "TFIM h=0.5, HVA p=2 (4 w)", "tfim_n6_h0.5_hea_L2": "TFIM h=0.5, HEA L=2 (18 w)",
    "tfim_n6_h1.0_hva_p2": "TFIM h=1.0, HVA p=2 (4 w)", "tfim_n6_h1.0_hea_L2": "TFIM h=1.0, HEA L=2 (18 w)",
    "tfim_n6_h2.0_hva_p2": "TFIM h=2.0, HVA p=2 (4 w)", "tfim_n6_h2.0_hea_L2": "TFIM h=2.0, HEA L=2 (18 w)",
    "tfim_n6_h1.0_hva_p4": "TFIM h=1.0, HVA p=4 (8 w)", "tfim_n6_h1.0_hea_L4": "TFIM h=1.0, HEA L=4 (30 w)",
    "xxz_n6_hea_L2": "Heisenberg, HEA L=2 (18 w)", "h2_ucc_1param": "H$_2$, 1-parameter ansatz",
    "h2_hea_L1": "H$_2$, HEA L=1 (4 w)",
}


def fig_energy(df: pd.DataFrame, experiment: str, labels: dict[str, str], xlabel: str, fname: str, title: str) -> None:
    sub = df[df.experiment == experiment]
    if sub.empty:
        return
    summary = agg(sub, ["ratio_llm", "ratio_random", "ratio_optimizer", "ratio_llm_then_opt"])
    cases = [c for c in labels if c in set(summary.case)] + sorted(set(summary.case) - set(labels))
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(cases) + 1.4))
    extras = [("ratio_random", "random", "o", "random angles (mean of 50)"),
              ("ratio_optimizer", "optimizer", "^", "optimizer, best of 5 random starts")]
    handles = dot_plot(ax, cases, [labels.get(c, c) for c in cases], summary, "ratio_llm", extras, xlabel, (0, 1.02))
    ax.set_title(title, loc="left", fontsize=10, color=TEXT)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, fname)


def fig_warmstart(df: pd.DataFrame) -> None:
    """Objective evaluations the optimizer needs to reach within 0.1% of its best value: random vs LLM start."""
    sub = df[(df.experiment.isin(["vqe", "qaoa"])) & (df.status == "ok")].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for ax, exp in zip(axes, ["vqe", "qaoa"]):
        s = sub[sub.experiment == exp]
        xs = np.arange(len(MODEL_ORDER))
        for k, model in enumerate(MODEL_ORDER):
            m = s[s.model == model]
            if m.empty:
                continue
            vals = [m.evals_random_init.mean(), m.evals_llm_init.mean()]
            errs = [m.evals_random_init.std() / np.sqrt(len(m)), m.evals_llm_init.std() / np.sqrt(len(m))]
            for j, (v, e, c, lab) in enumerate(zip(vals, errs, [COLOR["random"], COLOR[model]], ["random start", "LLM start"])):
                ax.bar(xs[k] + (j - 0.5) * 0.36, v, width=0.32, color=c, yerr=e, error_kw={"ecolor": MUTED, "lw": 1}, label=lab if k == 0 else None)
                ax.text(xs[k] + (j - 0.5) * 0.36, v + (e if not np.isnan(e) else 0) + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=9, color=TEXT)
        ax.set_title({"vqe": "VQE", "qaoa": "QAOA"}[exp], loc="left", fontsize=10, color=TEXT)
        _style(ax, "")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.spines["bottom"].set_visible(True)
    axes[0].set_ylabel("objective evaluations to reach\nthe optimizer's best value (±0.1%)", fontsize=9, color=TEXT)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "warmstart")


def fig_scaling(df: pd.DataFrame) -> None:
    """Fixed-ansatz QML: accuracy versus the number of hand-written weights, with the optimizer ceilings."""
    sub = df[(df.experiment == "qml_fixed")]
    if sub.empty:
        return
    summary = agg(sub, ["acc_llm", "acc_trained_support", "acc_trained_full", "chance"])
    weights = {"L1": 12, "L2": 18, "L4": 30, "L8": 54}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    series = [("acc_llm", None, "LLM-written weights"),
              ("acc_trained_support", "optimizer", "optimizer on the shown examples"),
              ("acc_trained_full", "classical", "optimizer on 200 training images")]
    for ax, task, title in zip(axes, ["38", "all10"], ["3 vs 8", "10 classes"]):
        for model in MODEL_ORDER:
            for metric, ckey, label in series:
                xs, ys, es = [], [], []
                for L, w in weights.items():
                    r = summary[(summary.case == f"digits_{task}_amp_hea_{L}") & (summary.model == model)]
                    if r.empty or np.isnan(r.iloc[0][f"{metric}_mean"]):
                        continue
                    xs.append(w); ys.append(r.iloc[0][f"{metric}_mean"]); es.append(r.iloc[0][f"{metric}_std"])
                if not xs:
                    continue
                color = COLOR[model] if ckey is None else COLOR[ckey]
                lab = f"{MODEL_LABEL[model]}, {label}" if ckey is None and len(MODEL_ORDER) > 1 else label
                ax.errorbar(xs, ys, yerr=np.nan_to_num(es), color=color, marker=MARKER[model] if ckey is None else "^",
                            markersize=6, linewidth=2, capsize=0, label=lab, markeredgecolor="white", markeredgewidth=1.2)
        ch = summary[summary.case.str.contains(f"digits_{task}_")]["chance_mean"].iloc[0]
        ax.axhline(ch, color=GRID, linewidth=1.5, zorder=1)
        ax.text(weights["L8"] + 1, ch, "chance", color=MUTED, fontsize=8, va="center")
        ax.set_xticks(list(weights.values()))
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        _style(ax, "number of hand-written weights")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("test accuracy", fontsize=9, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "scaling")


# ---------------------------------------------------------------- tables

def _fmt(m, s=None, digits=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    return f"{m:.{digits}f}" if s is None or np.isnan(s) else f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def table_qml(df: pd.DataFrame) -> str:
    free = agg(df[df.experiment == "qml_free"], ["acc_llm", "acc_random", "acc_trained_support", "acc_llm_finetuned", "logreg_support", "n_qubits", "n_gates", "n_weights"])
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Task & Model & Valid & LLM & Random w. & Optimizer & LLM$\to$opt. & LogReg \\", r"\midrule"]
    for case in [c for c in QML_FREE_LABEL if c in set(free.case)]:
        for model in MODEL_ORDER:
            r = free[(free.case == case) & (free.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{QML_FREE_LABEL[case]} & {MODEL_LABEL[model]} & {int(r.n_valid)}/{int(r.n_trials)} & "
                         f"{_fmt(r.acc_llm_mean, r.acc_llm_std)} & {_fmt(r.acc_random_mean, r.acc_random_std)} & "
                         f"{_fmt(r.acc_trained_support_mean, r.acc_trained_support_std)} & "
                         f"{_fmt(r.acc_llm_finetuned_mean, r.acc_llm_finetuned_std)} & {_fmt(r.logreg_support_mean, r.logreg_support_std)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_qml_fixed(df: pd.DataFrame) -> str:
    t = agg(df[df.experiment == "qml_fixed"], ["acc_llm", "acc_random", "acc_trained_support", "acc_llm_finetuned", "acc_trained_full"])
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Task, ansatz & Model & Valid & LLM & Random w. & Opt. (shown) & LLM$\to$opt. & Opt. (200 img.) \\", r"\midrule"]
    for case in [c for c in QML_FIXED_LABEL if c in set(t.case)]:
        for model in MODEL_ORDER:
            r = t[(t.case == case) & (t.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{QML_FIXED_LABEL[case]} & {MODEL_LABEL[model]} & {int(r.n_valid)}/{int(r.n_trials)} & "
                         f"{_fmt(r.acc_llm_mean, r.acc_llm_std)} & {_fmt(r.acc_random_mean, r.acc_random_std)} & "
                         f"{_fmt(r.acc_trained_support_mean, r.acc_trained_support_std)} & "
                         f"{_fmt(r.acc_llm_finetuned_mean, r.acc_llm_finetuned_std)} & {_fmt(r.acc_trained_full_mean, r.acc_trained_full_std)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_ratio(df: pd.DataFrame, experiment: str, labels: dict[str, str], header: str) -> str:
    t = agg(df[df.experiment == experiment], ["ratio_llm", "ratio_random", "ratio_optimizer", "ratio_llm_then_opt"])
    lines = [r"\begin{tabular}{llccccc}", r"\toprule", header, r"\midrule"]
    cases = [c for c in labels if c in set(t.case)] + sorted(set(t.case) - set(labels))
    for case in cases:
        for model in MODEL_ORDER:
            r = t[(t.case == case) & (t.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{labels.get(case, case)} & {MODEL_LABEL[model]} & {int(r.n_valid)}/{int(r.n_trials)} & "
                         f"{_fmt(r.ratio_llm_mean, r.ratio_llm_std)} & {_fmt(r.ratio_random_mean, r.ratio_random_std)} & "
                         f"{_fmt(r.ratio_optimizer_mean, r.ratio_optimizer_std)} & {_fmt(r.ratio_llm_then_opt_mean, r.ratio_llm_then_opt_std)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def qaoa_labels(df: pd.DataFrame) -> dict[str, str]:
    kinds = {"3reg": "3-regular", "er": "Erd\\H{o}s--R\\'enyi", "wring": "weighted ring"}
    out = {}
    for case in sorted(set(df[df.experiment == "qaoa"].case)):
        _, gid, p = case.split("_")
        kind = next(v for k, v in kinds.items() if gid.startswith(k))
        out[case] = f"{kind} \\#{gid[len(gid.rstrip('0123456789')):]}, p={p[1:]}"
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, help="models to include, in order")
    args = ap.parse_args()
    MODEL_ORDER[:] = args.models
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    df = df[df.model.isin(MODEL_ORDER)]
    if df.empty:
        print("no results")
        return 1
    df.to_csv(ROOT / "results" / "summary.csv", index=False)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.facecolor": "white"})
    fig_qml(df)
    fig_energy(df, "vqe", VQE_LABEL, "fraction of the spectrum captured: 1 = ground state, 0 = highest state", "vqe",
               "VQE: energy of the state prepared with LLM-written angles")
    ql = qaoa_labels(df)
    fig_energy(df, "qaoa", {k: v.replace("\\H{o}", "ő").replace("\\'e", "é").replace("\\#", "#") for k, v in ql.items()},
               "approximation ratio ⟨C⟩ / C_max", "qaoa", "QAOA MaxCut: expected cut with LLM-written angles")
    fig_warmstart(df)
    fig_scaling(df)
    (TAB_DIR / "qml_free.tex").write_text(table_qml(df) + "\n")
    (TAB_DIR / "qml_fixed.tex").write_text(table_qml_fixed(df) + "\n")
    (TAB_DIR / "vqe.tex").write_text(table_ratio(df, "vqe", VQE_LABEL, r"Problem, ansatz & Model & Valid & LLM & Random & Optimizer & LLM$\to$opt. \\") + "\n")
    (TAB_DIR / "qaoa.tex").write_text(table_ratio(df, "qaoa", ql, r"Graph, depth & Model & Valid & LLM & Random & Optimizer & LLM$\to$opt. \\") + "\n")
    counts = df.groupby(["experiment", "model"]).status.value_counts().unstack(fill_value=0)
    print(counts)
    print(f"figures -> {FIG_DIR}, tables -> {TAB_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
