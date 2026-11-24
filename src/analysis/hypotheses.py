"""Pre-registered pass/fail tests (PREREGISTRATION.md v0.2 + amendment 2026-11-18).

Records are the dicts produced by audit/report.assemble_record, one per cell, each carrying its ``cell``
({method, forget_frac, seed, track, ...}) and ``regime`` ('class' = the whole forget class removed, 'subset' =
a random part of it).  H1 and H3's relearning parts are evaluated in the class regime only: in the subset regime
the oracle still generates the class, so "relearning" the class is undefined (its steps-to-90% are 0).

Relearning times that never reached the target are right-censored (inf).  Rank statistics treat them as tied
at the maximum instead of dropping them (dropping them selected on the outcome).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.stats import bootstrap_ci, ci_overlaps, partial_spearman, spearman

APPROX_METHODS = ("gradient_ascent", "gradient_difference", "label_annealing", "variational_ewc", "npo")
H1_METHODS = ("gradient_ascent", "label_annealing")


def _df(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        d.update(r.get("cell", {}))
        rows.append(d)
    df = pd.DataFrame(rows)
    if "regime" not in df and not df.empty:
        df["regime"] = "class"
    return df


def _censor(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    fin = np.isfinite(x)
    cap = (x[fin].max() if fin.any() else 0.0) + 1.0
    x[~fin] = cap
    return x


def _ratio(steps: float, oracle: float) -> float:
    """Relearn ratio with censoring: approx finite & oracle never relearns -> 0 (much faster than oracle)."""
    if steps is None or oracle is None or (isinstance(steps, float) and np.isnan(steps)) or (isinstance(oracle, float) and np.isnan(oracle)):
        return float("nan")
    if not np.isfinite(oracle):
        return 0.0 if np.isfinite(steps) else float("nan")
    if float(oracle) == 0.0:
        return float("nan")  # the oracle is already at the target: relearning is undefined, not a pass
    if not np.isfinite(steps):
        return float("inf")
    return float(steps) / max(float(oracle), 1e-9)


def _ref(df: pd.DataFrame, method: str, col: str) -> float:
    sub = df[df["method"] == method]
    return float(sub[col].mean()) if (not sub.empty and col in sub) else float("nan")


def test_h1(records: List[Dict]) -> Dict:
    """H1 (class regime).  Per seed, for GA and label annealing: MIA AUC <= 0.55 AND probe balanced accuracy >=
    80% of the do-nothing value AND above the oracle's probe (amendment: oracle reference) AND relearn steps <= 1/5
    of the same-seed oracle's.  Confirmed in >= 2 of 3 seeds (>= 2 of 2 at minimum scale).  Refuted if probe < 50% of
    the do-nothing value, or relearning is within 2x of the oracle."""
    df = _df(records)
    if df.empty or "method" not in df:
        return {"verdict": "inconclusive", "reason": "no records"}
    df = df[df["regime"] == "class"]
    pcol = "probe_balanced_accuracy" if "probe_balanced_accuracy" in df else "probe_accuracy"
    probe0, probe_or = _ref(df, "do_nothing", pcol), _ref(df, "oracle", pcol)
    orc = {int(r["seed"]): r.get("relearn_steps_to_90") for _, r in df[df["method"] == "oracle"].iterrows()}
    out = {}
    for m in H1_METHODS:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        passes, ratios, probes = 0, [], []
        for _, row in sub.iterrows():
            ratio = _ratio(row.get("relearn_steps_to_90"), orc.get(int(row["seed"])))
            ratios.append(ratio)
            probes.append(row.get(pcol, np.nan))
            ok = (row.get("mia_loss_auc", 1.0) <= 0.55) and (row.get(pcol, 0) >= 0.8 * probe0) and (np.isnan(probe_or) or row.get(pcol, 0) > probe_or) and (ratio <= 0.2)
            passes += int(bool(ok))
        need = 2 if len(sub) >= 2 else 1
        refuted = (np.nanmean(probes) < 0.5 * probe0) or (np.nanmean([r for r in ratios if np.isfinite(r)] or [np.inf]) > 0.5)
        out[m] = {"verdict": "confirmed" if passes >= need else ("refuted" if refuted else "under-powered"), "n_seeds_pass": int(passes), "n_seeds": int(len(sub)),
                  "relearn_ratios": ratios, "probe": float(np.nanmean(probes)), "mia_loss_auc": float(sub["mia_loss_auc"].mean()) if "mia_loss_auc" in sub else float("nan")}
    return {"verdict": out, "probe_do_nothing": probe0, "probe_oracle": probe_or, "oracle_relearn_steps": orc}


def test_h2(records: List[Dict]) -> Dict:
    """H2 (redesigned 2026-11-18): oracle-normalised closeness E.  Per regime and approximate method:
    confirmed if E_near < 0.5 with its 95% CI below 0.5, while E_forget exceeds E_near by >= 0.25 with
    non-overlapping CIs (the certificate passes, the downstream fingerprint does not).  Refuted if E_near >= 0.8
    (the method reproduces the oracle's retain-side change).  E = 1 is an independent retrain by construction."""
    df = _df(records)
    if df.empty or "E_near" not in df:
        return {"verdict": "inconclusive", "reason": "no fingerprint (needs >= 1 oracle; floor needs 2)"}
    out = {}
    for regime, g in df.groupby("regime"):
        res = {}
        for m in [x for x in APPROX_METHODS + ("retain_finetune",) if x in set(g["method"])]:
            sub = g[g["method"] == m]
            en, ef = sub["E_near"].astype(float).values, sub["E_forget"].astype(float).values
            ci_n = {"mean": float(np.nanmean(en)) if np.isfinite(en).any() else float("nan"), "lo": float(np.nanmin(sub["E_near_lo"])) if np.isfinite(en).any() else float("nan"), "hi": float(np.nanmax(sub["E_near_hi"])) if np.isfinite(en).any() else float("nan")}
            ci_f = {"mean": float(np.nanmean(ef)) if np.isfinite(ef).any() else float("nan"), "lo": float(np.nanmin(sub["E_forget_lo"])) if np.isfinite(ef).any() else float("nan"), "hi": float(np.nanmax(sub["E_forget_hi"])) if np.isfinite(ef).any() else float("nan")}
            if np.isnan(ci_n["mean"]) or np.isnan(ci_f["mean"]):
                v = "inconclusive (no measurable removal trace: snr < 1.2)"
            elif ci_n["mean"] >= 0.8:
                v = "refuted"
            elif ci_n["hi"] < 0.5 and ci_f["mean"] - ci_n["mean"] >= 0.25 and not ci_overlaps(ci_f, ci_n):
                v = "confirmed"
            else:
                v = "under-powered"
            res[m] = {"verdict": v if m != "retain_finetune" else "control", "E_near": ci_n, "E_forget": ci_f,
                      "E_heldout": float(sub["E_heldout"].mean()) if "E_heldout" in sub else float("nan"), "E_far": float(sub["E_far"].mean()) if "E_far" in sub else float("nan"),
                      "fp_corr_pred_after": float(sub["fp_corr_pred_after"].mean()) if "fp_corr_pred_after" in sub else float("nan")}
        res["_attribution_validity_corr_pred_oracle"] = float(g["fp_corr_pred_oracle"].mean()) if "fp_corr_pred_oracle" in g else float("nan")
        res["_floor_estimated"] = bool(g.get("floor_estimated_forget", pd.Series([False])).fillna(False).astype(bool).all())
        out[regime] = res
    # legacy statistic, reported only
    legacy = {}
    if "attr_drop" in df:
        for m, g in df.groupby("method"):
            legacy[m] = float(g["attr_drop"].mean())
    return {"verdict": out, "legacy_attr_drop": legacy}


def test_h3(records: List[Dict]) -> Dict:
    """H3 (full-parameter cells, class regime for the relearning part).  Confirmed if top-5 energy > 0.7 for
    approximate methods vs < 0.3 for the oracle delta AND exceeds the matched-steps retain_finetune control by >= 0.2
    (amendment), AND partial Spearman(effective rank, relearn steps | ||Delta||) >= 0.7.  Refuted if |rho| < 0.3 or the
    spectra are indistinguishable from the control.  Alignment cos(g_f, Delta) is reported with the same partial
    correlation (secondary)."""
    df = _df(records)
    if "full_ft" in df:
        df = df[df["full_ft"] != False]  # noqa: E712  (LoRA cells: rank bounded by construction)
    if df.empty or "top5_energy_wmean" not in df:
        return {"verdict": "inconclusive"}
    appr = df[df["method"].isin(APPROX_METHODS)]
    t5_appr = float(appr["top5_energy_wmean"].mean()) if not appr.empty else float("nan")
    t5_orc, t5_ctl = _ref(df, "oracle", "top5_energy_wmean"), _ref(df, "retain_finetune", "top5_energy_wmean")
    cells = df[df["method"].isin(APPROX_METHODS) & (df["regime"] == "class")]
    cells = cells[cells["relearn_steps_to_90"].notna()] if "relearn_steps_to_90" in cells else cells.iloc[0:0]
    y = _censor(cells["relearn_steps_to_90"].values) if len(cells) else np.array([])
    rho = partial_spearman(cells["effective_rank_wmean"].values, y, cells["delta_norm_total"].values) if len(cells) >= 4 else float("nan")
    rho_raw = spearman(cells["effective_rank_wmean"].values, y) if len(cells) >= 3 else float("nan")
    rho_align = partial_spearman(cells["relearn_alignment"].values, y, cells["delta_norm_total"].values) if (len(cells) >= 4 and "relearn_alignment" in cells) else float("nan")
    spectra_differ = (t5_appr > 0.7) and (np.isnan(t5_orc) or t5_orc < 0.3) and (np.isnan(t5_ctl) or t5_appr - t5_ctl >= 0.2)
    if np.isnan(rho):
        verdict = "inconclusive"
    elif (not np.isnan(t5_ctl) and abs(t5_appr - t5_ctl) < 0.1) or abs(rho) < 0.3:
        verdict = "refuted"
    elif spectra_differ and rho >= 0.7:
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    if len(cells) < 9 and verdict == "confirmed":
        verdict = "confirmed (preliminary: < 9 cells)"
    return {"verdict": verdict, "top5_energy_approx": t5_appr, "top5_energy_oracle": t5_orc, "top5_energy_retain_finetune": t5_ctl,
            "partial_spearman_rank_relearn_given_norm": rho, "spearman_raw": rho_raw, "partial_spearman_alignment_relearn_given_norm": rho_align, "n_cells": int(len(cells))}


def h2_h3_link(records: List[Dict]) -> Dict:
    """Secondary: does surviving retain-side influence (1 - E_near) predict relearning speed?  Class regime."""
    df = _df(records)
    if df.empty or "E_near" not in df:
        return {"verdict": "inconclusive"}
    sub = df[df["method"].isin(APPROX_METHODS) & (df["regime"] == "class") & df["E_near"].notna()]
    if len(sub) < 4:
        return {"verdict": "inconclusive", "n": int(len(sub))}
    rho = spearman(1.0 - sub["E_near"].values, -_censor(sub["relearn_steps_to_90"].values))
    return {"spearman_surviving_influence_vs_relearn_speed": rho, "n": int(len(sub))}
