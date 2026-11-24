"""Scoring of P5 (pre-registered) and A3-a ... A3-e (exploratory) from run-v5 outputs.

The operationalisations below were fixed on 2026-11-21, while run v5 was running and before any of its outputs
existed (paper/RESEARCH_JOURNAL.md section 3).  Each prediction gets a verdict in {held, failed, untestable} plus the
numbers it was decided on.  With 2 seeds there is no meaningful seed-level CI; per-seed values are reported side by
side and a prediction "holds" only if it holds in every available seed.

  P5    for each gradient-ascent seed: forgetting fraction f_k = (pre - rate_k) / (pre - rate_full) >= 0.8 for some
        k <= 4, AND relearn steps do not decrease with k (Spearman(k, relearn) >= 0 over k in 1,4,16,64,full)
  A3-a  gradient ascent: among its grid entries the target (rate <= 0.10, retain >= 0.90) is met at some step only
        for lr < 3e-5, and window = first(retain < 0.90) - first(rate <= 0.10) <= 400 steps for every entry that
        forgets; gradient_difference and variational_ewc selections have reached_target = True
  A3-b  every selected-point record: |AUC - 0.5| <= 0.05 for loss, SecMI and LiRA
  A3-c  every gradient-method selected record: d(u,O)/d_floor on the forget set > 3 AND relearn ratio vs the same-seed
        oracle < 0.5
  A3-d  embedding_reset: cat rate <= 0.15 AND retain accuracy >= original - 0.02 AND |mean near-class loss shift vs
        original| <= d_floor_near (proxy for "retain-class distance at the floor")
  A3-e  embedding_reset relearn ratio < 0.2 AND it is the smallest relearn ratio among all audited A3 records
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..common.config import cell_name

FRAC = 0.1
GRAD_METHODS = ("gradient_ascent", "gradient_difference", "variational_ewc")


def _load(p: str):
    return json.load(open(p)) if os.path.exists(p) else None


def _d3(root: str, method: str, seed: int) -> Optional[Dict]:
    return _load(os.path.join(root, "D3_ddpm_unlearn", cell_name({"method": method, "forget_frac": FRAC, "seed": seed}), "audit_record.json"))


def _ratio(steps, oracle) -> float:
    from .hypotheses import _ratio as r

    return r(steps, oracle)


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(y == y[0]):
        return 0.0
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _verdict(flags: List[bool]) -> str:
    return "untestable" if not flags else ("held" if all(flags) else "failed")


def score_p5(root: str) -> Dict:
    per = {}
    for s in (0, 1):
        t = _load(os.path.join(root, "D3_ddpm_unlearn", cell_name({"method": "gradient_ascent", "forget_frac": FRAC, "seed": s}), "truncation.json"))
        if not t:
            continue
        pre = float(t["pre_unlearn_forget_rate"])
        rows = {r["rank"]: r for r in t["rows"]}
        full = rows["full"]["forget_rate"]
        denom = pre - full
        frac = {k: (pre - r["forget_rate"]) / denom if denom > 1e-9 else float("nan") for k, r in rows.items()}
        ks = [1, 4, 16, 64, "full"]
        order = [k if k != "full" else 10 ** 6 for k in ks if k in rows]
        rel = [rows[k]["relearn_steps_to_90"] for k in ks if k in rows]
        rel = [x if (x is not None and np.isfinite(x)) else 1e9 for x in rel]
        rho = _spearman(order, rel)
        if not np.isfinite(frac.get("full", float("nan"))):
            per[s] = {"pre": pre, "untestable": "the full update does not forget (pre - rate_full ~ 0)"}
            continue
        low_k = any(frac.get(k, float("nan")) >= 0.8 for k in (1, 4))
        per[s] = {"pre": pre, "forget_fraction": frac, "rows": {str(k): {kk: v for kk, v in r.items() if kk != "relearn_curve_values"} for k, r in rows.items()},
                  "spearman_k_relearn": rho, "low_rank_reproduces_forgetting": low_k, "holds": bool(low_k and rho >= 0)}
    return {"verdict": _verdict([v["holds"] for v in per.values() if "holds" in v]), "per_seed": per}


def _a3_records(root: str) -> List[Dict]:
    out = []
    for p in glob.glob(os.path.join(root, "D6_oppoint_audit", "*", "audit_record.json")):
        r = json.load(open(p))
        r["_dir"] = os.path.basename(os.path.dirname(p))
        out.append(r)
    return out


def _method(r: Dict) -> str:
    return r.get("cell", {}).get("method") or r["_dir"].split("method")[1].split("_seed")[0]


def _seed(r: Dict) -> int:
    c = r.get("cell", {})
    return int(c["seed"]) if "seed" in c else int(r["_dir"].split("_seed")[1].split("_")[0])


def score_a3a(root: str) -> Dict:
    from ..diffusion.oppoint import traj_name  # noqa: F401  (documented naming)

    trajs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "D6_oppoint", "traj_*.json")))]
    sel = _load(os.path.join(root, "D6_oppoint", "selected.json")) or {}
    if not trajs:
        return {"verdict": "untestable", "reason": "no trajectories"}
    rows, flags = [], []
    for t in trajs:
        e, pts = t["entry"], [p for p in t["points"] if p["step"] >= 1]
        f = next((p["step"] for p in pts if p["forget_rate"] <= 0.10), None)
        c = next((p["step"] for p in pts if p["retain_acc"] < 0.90), None)
        hit = any(p["forget_rate"] <= 0.10 and p["retain_acc"] >= 0.90 for p in pts)
        row = {**e, "first_forget_step": f, "first_collapse_step": c, "target_met": hit,
               "window": (None if f is None else ((c if c is not None else 10 ** 9) - f))}
        rows.append(row)
        if e["method"] == "gradient_ascent":
            if float(e["lr"]) >= 3e-5:
                flags.append(not hit)
            if f is not None:
                flags.append(row["window"] <= 400)
    for m in ("gradient_difference", "variational_ewc"):
        flags.append(bool(sel.get(m) and sel[m].get("reached_target")))
    return {"verdict": _verdict(flags), "trajectories": rows, "selected": sel}


def score_records(root: str) -> Dict:
    recs = _a3_records(root)
    if not recs:
        return {k: {"verdict": "untestable"} for k in ("A3-b", "A3-c", "A3-d", "A3-e")}
    table, b, c, d, e_ratio = [], [], [], [], {}
    for r in recs:
        m, s = _method(r), _seed(r)
        orc, orig = _d3(root, "oracle", s), _d3(root, "do_nothing", s)
        ratio = _ratio(r.get("relearn_steps_to_90"), orc.get("relearn_steps_to_90") if orc else None)
        dn = (r.get("d_u_oracle_forget") or float("nan")) / max(r.get("d_floor_forget") or float("nan"), 1e-12)
        row = {"dir": r["_dir"], "method": m, "seed": s, "reached_target": r.get("oppoint_reached_target"),
               "cat_rate": r.get("forget_class_rate_conditional"), "retain_acc": r.get("retain_sample_accuracy"),
               "mia_loss": r.get("mia_loss_auc"), "mia_secmi": r.get("mia_secmi_auc"), "mia_lira": r.get("mia_lira_auc"),
               "probe": r.get("probe_accuracy"), "relearn_steps": r.get("relearn_steps_to_90"), "relearn_ratio": ratio,
               "d_over_floor_forget": dn, "top5_energy": r.get("top5_energy_wmean"), "alignment": r.get("relearn_alignment"),
               "delta_norm": r.get("delta_norm_total")}
        table.append(row)
        b.append(all(abs(float(r[k]) - 0.5) <= 0.05 for k in ("mia_loss_auc", "mia_secmi_auc", "mia_lira_auc") if r.get(k) is not None))
        if m in GRAD_METHODS and np.isfinite(dn) and not np.isnan(ratio):  # undefined ratio/distance -> not scored
            c.append(bool(dn > 3 and ratio < 0.5))
        if m == "embedding_reset":
            near_shift = abs((r.get("loss_mean_near_after") or 0) - (r.get("loss_mean_near_original") or 0))
            ok_ret = orig is not None and r.get("retain_sample_accuracy") is not None and r["retain_sample_accuracy"] >= orig["retain_sample_accuracy"] - 0.02
            row["near_loss_shift"], row["d_floor_near"] = near_shift, r.get("d_floor_near")
            d.append(bool(r.get("forget_class_rate_conditional") is not None and r["forget_class_rate_conditional"] <= 0.15 and ok_ret
                          and near_shift <= (r.get("d_floor_near") or 0)))
        e_ratio.setdefault(s, []).append((ratio, m))
    e = []
    for s, lst in e_ratio.items():
        er = [x for x, m in lst if m == "embedding_reset"]
        others = [x for x, m in lst if m != "embedding_reset" and np.isfinite(x)]
        if er and not np.isnan(er[0]):
            e.append(bool(er[0] < 0.2 and all(er[0] <= o for o in others)))
    table.sort(key=lambda x: (x["method"], x["seed"]))
    return {"table": table, "A3-b": {"verdict": _verdict(b)}, "A3-c": {"verdict": _verdict(c)},
            "A3-d": {"verdict": _verdict(d)}, "A3-e": {"verdict": _verdict(e)}}


def a3_report(root: str) -> Dict:
    rep = {"P5": score_p5(root), "A3-a": score_a3a(root), **score_records(root)}
    d = os.path.join(root, "figures")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "a3_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(json.dumps({k: v.get("verdict") for k, v in rep.items() if isinstance(v, dict) and "verdict" in v}, indent=2))
    return rep


if __name__ == "__main__":
    import sys

    a3_report(sys.argv[1] if len(sys.argv) > 1 else "results")
