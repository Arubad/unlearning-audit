"""Collect audit records from results/ and draw the figures + hypothesis report."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from .hypotheses import h2_h3_link, test_h1, test_h2, test_h3


def load_records(results_root: str) -> List[Dict]:
    recs = []
    for p in glob.glob(os.path.join(results_root, "*", "*", "audit_record.json")):
        with open(p) as f:
            recs.append(json.load(f))
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


def analyze(results_root: str) -> Dict:
    figdir = os.path.join(results_root, "figures")
    os.makedirs(figdir, exist_ok=True)
    recs = load_records(results_root)
    report = {"n_records": len(recs), "H1": test_h1(recs), "H2": test_h2(recs), "H3": test_h3(recs), "H2_H3_link": h2_h3_link(recs)}
    # the do-nothing control
    from ..audit.report import control_assertions

    for r in recs:
        if r.get("cell", {}).get("method") == "do_nothing":
            report["control_do_nothing"] = control_assertions(r)
    fig_attribution_drop(recs, os.path.join(figdir, "fig_attribution_drop.png"))
    fig_rank_vs_relearn(recs, os.path.join(figdir, "fig_rank_vs_relearn.png"))
    with open(os.path.join(figdir, "hypothesis_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in report.items()}, indent=2, default=str))
    return report
