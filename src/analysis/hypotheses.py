"""Pre-registered pass/fail tests (thresholds from PREREGISTRATION.md v0.2).

Records are the dicts produced by audit/report.assemble_record, one per cell,
each carrying its ``cell`` ({method, forget_frac, retain_available, seed, track}).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.stats import bootstrap_ci, ci_overlaps, partial_spearman, spearman

APPROX_METHODS = ("gradient_ascent", "gradient_difference", "label_annealing", "variational_ewc", "npo")


def _df(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        d.update(r.get("cell", {}))
        rows.append(d)
    return pd.DataFrame(rows)


def test_h1(records: List[Dict]) -> Dict:
    """H1: for GA and label-annealing, MIA AUC <= 0.55 while probe accuracy >= 80% of pre-unlearning
    and relearn steps <= 1/5 oracle, in >= 2 of 3 seeds.  Refuted if probe < 50% of original or relearn within 2x oracle."""
    df = _df(records)
    methods = [m for m in ("gradient_ascent", "label_annealing") if m in set(df.get("method", []))]
    if df.empty or not methods or "mia_loss_auc" not in df:
        return {"verdict": "inconclusive", "reason": "need MIA + GA/label_annealing cells"}
    base = df[df.get("method") == "do_nothing"]
    probe0 = base["probe_accuracy"].mean() if not base.empty else df["probe_accuracy"].max()
    out = {}
    for m in methods:
        sub = df[df["method"] == m]
        if sub["mia_loss_auc"].isna().all():
            out[m] = {"verdict": "inconclusive", "reason": "no finite MIA AUC for this method's cells"}
            continue
        passes = 0
        for _, row in sub.iterrows():
            ok = row.get("mia_loss_auc", 1) <= 0.55 and row.get("probe_accuracy", 0) >= 0.8 * probe0 and row.get("relearn_ratio_to_oracle", 1) <= 0.2
            passes += int(ok)
        refuted = (sub["probe_accuracy"].mean() < 0.5 * probe0) or (sub["relearn_ratio_to_oracle"].mean() > 0.5)
        out[m] = {"verdict": "confirmed" if passes >= 2 else ("refuted" if refuted else "under-powered"), "n_seeds_pass": int(passes)}
    return {"verdict": out, "probe_baseline": float(probe0)}


def test_h2(records: List[Dict]) -> Dict:
    """H2: mean attribution of forget examples on retain outputs drops < 50% under approximate unlearning
    vs > 90% under retraining, with non-overlapping CIs.  Refuted if the approximate drop is within 10 pts of retrain."""
    df = _df(records)
    appr = df[df.get("method").isin(APPROX_METHODS)] if "method" in df else df.iloc[0:0]
    if appr.empty or "attr_drop" not in df:
        return {"verdict": "inconclusive"}
    appr_drop = bootstrap_ci(appr["attr_drop"].values)
    orc = df[df.get("method") == "oracle"]
    retrain_drop = None
    if "attr_drop_oracle" in df and df["attr_drop_oracle"].notna().any():
        retrain_drop = bootstrap_ci(df["attr_drop_oracle"].dropna().values)
    elif not orc.empty:
        retrain_drop = bootstrap_ci(orc["attr_drop"].values)
    if retrain_drop is None:
        return {"verdict": "inconclusive", "approx_drop": appr_drop}
    if abs(appr_drop["mean"] - retrain_drop["mean"]) < 0.10:
        verdict = "refuted"
    elif appr_drop["mean"] < 0.5 and retrain_drop["mean"] > 0.9 and not ci_overlaps(appr_drop, retrain_drop):
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    return {"verdict": verdict, "approx_drop": appr_drop, "retrain_drop": retrain_drop}


def test_h3(records: List[Dict]) -> Dict:
    """H3: top-5 energy > 70% for approximate vs < 30% for the oracle delta, and the partial Spearman of
    effective rank with relearn steps controlling for ||delta|| >= 0.7 across cells.  Full-parameter cells only."""
    df = _df(records)
    df = df[df.get("full_ft", True) != False] if "full_ft" in df else df  # keep diffusion (no flag) and full-FT LLM
    appr = df[df.get("method").isin(APPROX_METHODS)]
    orc = df[df.get("method") == "oracle"]
    if appr.empty or "top5_energy_wmean" not in df:
        return {"verdict": "inconclusive"}
    t5_appr = float(appr["top5_energy_wmean"].mean())
    t5_orc = float(orc["top5_energy_wmean"].mean()) if not orc.empty else float("nan")
    cells = df[df["method"].isin(APPROX_METHODS) & df.get("relearn_steps_to_90").notna()]
    cells = cells[np.isfinite(cells["relearn_steps_to_90"])]
    rho = partial_spearman(cells["effective_rank_wmean"].values, cells["relearn_steps_to_90"].values, cells["delta_norm_total"].values) if len(cells) >= 4 else float("nan")
    rho_raw = spearman(cells["effective_rank_wmean"].values, cells["relearn_steps_to_90"].values) if len(cells) >= 3 else float("nan")
    if np.isnan(rho) or (not np.isnan(t5_orc) and abs(t5_appr - t5_orc) < 0.1):
        verdict = "inconclusive" if np.isnan(rho) else "refuted"
    elif t5_appr > 0.7 and (np.isnan(t5_orc) or t5_orc < 0.3) and rho >= 0.7:
        verdict = "confirmed"
    elif abs(rho) < 0.3:
        verdict = "refuted"
    else:
        verdict = "under-powered"
    return {"verdict": verdict, "top5_energy_approx": t5_appr, "top5_energy_oracle": t5_orc, "partial_spearman_rank_relearn_given_norm": rho, "spearman_raw": rho_raw, "n_cells": int(len(cells))}


def h2_h3_link(records: List[Dict]) -> Dict:
    """Secondary analysis: does surviving retain-influence predict relearning speed?"""
    df = _df(records)
    sub = df[df.get("method").isin(APPROX_METHODS)]
    sub = sub[np.isfinite(sub.get("relearn_steps_to_90", np.nan)) & sub.get("attr_drop").notna()]
    if len(sub) < 4:
        return {"verdict": "inconclusive"}
    surviving = 1.0 - sub["attr_drop"].values
    rho = spearman(surviving, -sub["relearn_steps_to_90"].values)  # more surviving influence -> fewer steps
    return {"spearman_surviving_influence_vs_relearn_speed": rho, "n": int(len(sub))}
