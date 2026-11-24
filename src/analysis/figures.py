"""Collect audit records from results/ and draw the figures + hypothesis report."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from .hypotheses import h2_h3_link, test_h1, test_h2, test_h3


EXPLORATORY_PREFIX = "D6_"  # Amendment A3 (post-hoc) experiments; never pooled into the pre-registered tests


def load_records(results_root: str, exploratory: bool = False) -> List[Dict]:
    """Audit records, tagged with their experiment directory.  By default only the pre-registered experiments:
    A3 cells reuse method names at other hyperparameters and would silently contaminate H1-H3."""
    recs = []
    for p in glob.glob(os.path.join(results_root, "*", "*", "audit_record.json")):
        exp = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if exp.startswith(EXPLORATORY_PREFIX) != exploratory:
            continue
        with open(p) as f:
            r = json.load(f)
        r["_experiment"] = exp
        recs.append(r)
    return recs


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_attribution_drop(records: List[Dict], out: str) -> None:
    plt = _plt()
    import pandas as pd

    df = pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, (dict, list))}, **r.get("cell", {})} for r in records])
    if df.empty or "attr_drop" not in df:
        return
    g = df.groupby("method")["attr_drop"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(g.index, g.values)
    ax.axhline(0.9, ls="--", c="k", lw=0.8, label="retrain target (0.9)")
    ax.set_ylabel("attribution drop (1 = influence gone)")
    ax.set_title("H2: does forget-example influence survive?")
    ax.legend(fontsize=7)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fig_rank_vs_relearn(records: List[Dict], out: str) -> None:
    plt = _plt()
    import pandas as pd

    df = pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, (dict, list))}, **r.get("cell", {})} for r in records])
    if df.empty or "effective_rank_wmean" not in df:
        return
    df = df[np.isfinite(df.get("relearn_steps_to_90", np.nan))]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    sizes = 30 + 200 * (df["delta_norm_total"] / df["delta_norm_total"].max())
    sc = ax.scatter(df["effective_rank_wmean"], df["relearn_steps_to_90"], s=sizes, c=df.get("method").astype("category").cat.codes, alpha=0.7)
    ax.set_xlabel("effective rank of delta theta (norm-weighted)")
    ax.set_ylabel("relearn steps to 90%")
    ax.set_title("H3: marker size = ||delta theta||")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


METHOD_ORDER = ["do_nothing", "retain_finetune", "gradient_ascent", "gradient_difference", "label_annealing", "variational_ewc", "oracle"]


def fig_closeness_ladder(records: List[Dict], out: str) -> None:
    """Figure 1: oracle-normalised closeness E on each query set, per method (mean over seeds, min/max bars).
    E = 0 is 'did nothing', E = 1 is 'as close as an independent retrain'."""
    plt = _plt()
    import pandas as pd

    df = pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, (dict, list))}, **r.get("cell", {})} for r in records])
    if df.empty or "E_forget" not in df:
        return
    sets = [s for s in ("forget", "heldout", "near", "far") if f"E_{s}" in df]
    regimes = sorted(df["regime"].dropna().unique()) if "regime" in df else ["class"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.5 * len(regimes), 3.6), squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        g = df[df["regime"] == reg]
        methods = [m for m in METHOD_ORDER if m in set(g["method"])]
        w = 0.8 / len(sets)
        for j, sname in enumerate(sets):
            stats = g.groupby("method")[f"E_{sname}"].agg(["mean", "min", "max"]).reindex(methods)
            x = np.arange(len(methods)) + (j - (len(sets) - 1) / 2) * w
            ax.bar(x, stats["mean"], w, label=sname, yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]], capsize=2)
        ax.axhline(0, c="k", lw=0.6)
        ax.axhline(1, c="k", lw=0.6, ls="--")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("closeness to retrain E")
        ax.set_title(f"{reg} regime  (0 = did nothing, 1 = retrain)", fontsize=9)
        ax.legend(fontsize=7, title="query set", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fig_relearn_curves(records: List[Dict], out: str) -> None:
    plt = _plt()
    recs = [r for r in records if r.get("relearn_curve_steps") and r.get("regime", "class") == "class"]
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    cmap = {m: f"C{i}" for i, m in enumerate(METHOD_ORDER)}
    seen = set()
    for r in recs:
        m = r.get("cell", {}).get("method")
        ax.plot(r["relearn_curve_steps"], r["relearn_curve_values"], c=cmap.get(m, "grey"), alpha=0.8, label=None if m in seen else m)
        seen.add(m)
    pre = np.nanmean([r.get("pre_unlearn_forget_rate", np.nan) for r in recs])
    ax.axhline(0.9 * pre, c="k", ls="--", lw=0.8, label="90% of original")
    ax.set_xlabel("relearning steps (5% of forget set)")
    ax.set_ylabel("forget-class generation rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def analyze(results_root: str) -> Dict:
    figdir = os.path.join(results_root, "figures")
    os.makedirs(figdir, exist_ok=True)
    allrecs = load_records(results_root)
    # tracks are analysed separately (LLM cells carry no diffusion regime and different loss scales); the top-level
    # keys stay the diffusion report for backward compatibility, the LLM report goes under "llm"
    track = lambda r: r.get("cell", {}).get("track", "diffusion")
    recs = [r for r in allrecs if track(r) != "llm"]
    llm = [r for r in allrecs if track(r) == "llm"]

    def _report(rs):
        return {"n_records": len(rs), "H1": test_h1(rs), "H2": test_h2(rs), "H3": test_h3(rs), "H2_H3_link": h2_h3_link(rs)}

    report = _report(recs)
    if llm:
        report["llm"] = _report(llm)
    # the do-nothing control, every cell (regime x seed x track), not just the last one read
    from ..audit.report import control_assertions

    ctrl = {}
    for r in allrecs:
        c = r.get("cell", {})
        if c.get("method") == "do_nothing":
            ctrl[f"{track(r)}/{c.get('regime', c.get('forget_frac'))}/s{c.get('seed')}"] = control_assertions(r)
    if ctrl:
        report["control_do_nothing"] = {"verdict": "passed" if all(v["verdict"] == "passed" for v in ctrl.values()) else "FAILED in " + ", ".join(k for k, v in ctrl.items() if v["verdict"] != "passed"), "cells": ctrl}
    fig_attribution_drop(recs, os.path.join(figdir, "fig_attribution_drop.png"))
    fig_rank_vs_relearn(recs, os.path.join(figdir, "fig_rank_vs_relearn.png"))
    fig_closeness_ladder(recs, os.path.join(figdir, "fig1_closeness_ladder.png"))
    fig_relearn_curves(recs, os.path.join(figdir, "fig_relearn_curves.png"))
    with open(os.path.join(figdir, "hypothesis_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in report.items()}, indent=2, default=str))
    # Amendment A3 + P5 are scored separately (exploratory records never enter the H1-H3 report above)
    from .a3_report import a3_report

    try:
        a3_report(results_root)
    except Exception as e:  # missing A3 outputs must not break the pre-registered report
        print("a3_report skipped:", repr(e))
    return report
