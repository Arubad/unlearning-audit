"""Study C4 (registered in PREREGISTRATION.md before any C4 data): C4-R row-swap design and C4-X cross-class control.

C4-R  For class k, network n in {retrain s0,s1,s2} carries retrain r's EMA row k (3x3 grid; diagonal = real retrains).
      Removed-label losses on label_diag's forget_train queries decide whether the row or the network drives the
      heterogeneity (R1), whether a common row restores homogeneity (R2) and calibration (R3).
C4-X  Retrains that removed a *different* class are independently trained networks that kept class k; their
      null-label E against the class-k retrains must stay < 0.5 (X1).

Usage:  python -m src.analysis.c4_controls [results_root]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

T_LOW = (10, 50, 100, 200, 300, 400)
GRID_CLASSES = (1, 3, 8, 9)
X_CLASSES = (1, 8, 9)


def _rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


def row_swap_stats(L: Dict, timesteps: Sequence[int], t_low: Sequence[int] = T_LOW) -> Dict:
    """L[(n, r)] = (n_queries, T) losses of network n carrying row r. Pure numpy (unit-tested)."""
    N = sorted({n for n, _ in L}); R = sorted({r for _, r in L})
    ti = [list(timesteps).index(t) for t in t_low]
    d_row = np.mean([_rms(L[(n, r)], L[(n, r2)]) for n in N for r, r2 in itertools.combinations(R, 2)], axis=0)
    d_net = np.mean([_rms(L[(n, r)], L[(n2, r)]) for r in R for n, n2 in itertools.combinations(N, 2)], axis=0)

    def hetero(models: List) -> float:
        pw = np.array([_rms(a, b) for a, b in itertools.combinations(models, 2)])
        return float(max((pw.max(0) / pw.min(0))[i] for i in ti))

    H_r = {str(r): hetero([L[(n, r)] for n in N]) for r in R}
    H_indep = hetero([L[(n, n)] for n in N]) if all((n, n) in L for n in N) else float("nan")
    return {"D_row_t": d_row.tolist(), "D_net_t": d_net.tolist(),
            "row_wins_low_t": int(sum(d_row[i] > d_net[i] for i in ti)),
            "H_r": H_r, "H_common": float(np.median(list(H_r.values()))), "H_indep": H_indep}


def score_c4r(per_class: Dict[str, Dict]) -> Dict:
    """per_class[c] = row_swap_stats(...) plus 'loo_common': list of plain-E point estimates (NaN = undefined)."""
    v = {"R1": {}, "R2": {}, "R3": {}}
    for c, s in per_class.items():
        v["R1"][c] = "held" if s["row_wins_low_t"] >= 4 else "failed"
        v["R2"][c] = "held" if s["H_common"] <= 0.5 * s["H_indep"] else "failed"
        e = [x for x in s["loo_common"] if x is not None and np.isfinite(x)]
        v["R3"][c] = "untestable" if len(e) < 3 else ("held" if np.mean([0.6 <= x <= 1.4 for x in e]) >= 2 / 3 else "failed")
    n1 = sum(x == "held" for x in v["R1"].values()); n2 = sum(x == "held" for x in v["R2"].values())
    testable = [x for x in v["R3"].values() if x != "untestable"]
    v["R1_overall"] = "held" if n1 >= 3 else "failed"
    v["R2_overall"] = "held" if n2 >= 3 else "failed"
    v["R3_overall"] = ("untestable" if len(testable) < 2 else ("held" if all(x == "held" for x in testable) else "failed"))
    return v


def score_c4x(E: Dict[str, List[float]]) -> Dict:
    """E[class] = 9 null-label E point estimates of cross-class retrains. X1: all 27 < 0.5."""
    allv = [x for vs in E.values() for x in vs]
    return {"X1": "held" if all(np.isfinite(x) and x < 0.5 for x in allv) else "failed",
            "max_E": float(np.nanmax(allv)), "n": len(allv)}


def main(root: str = "results", n: int = 256, seed: int = 0, n_boot: int = 1000, grid_classes=GRID_CLASSES,
         x_classes=X_CLASSES, out_name: str = "c4_controls.json") -> Dict:
    import copy

    import torch

    from ..audit.fingerprint import closeness_np
    from ..common.config import load_yaml
    from ..diffusion.data import cifar_labels, load_cifar
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.timestep_decomp import AUDIT_T, TIMESTEPS, _device, losses
    from ..diffusion.train import load_model
    from .label_diag import ORACLE_SEEDS, _ckpt

    dev = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    tr = load_cifar(cfg["data"].get("root", "data"))
    ltr = cifar_labels(tr)
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=dev)
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    orig = load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=dev)[0]
    null = int(orig.null_class)

    def queries(fc):  # label_diag / c3_consequence forget_train draw (first draw of default_rng(seed))
        idx = np.random.default_rng(seed).choice(np.where(ltr == fc)[0], n, replace=False)
        return torch.stack([tr[int(i)][0] for i in idx])

    def load(fc, s):
        return load_model(_ckpt(root, fc, s), cfg, use_ema=True, device=dev)[0]

    out = {"registered": "PREREGISTRATION.md, Study C4", "timesteps": TIMESTEPS, "audit_t": AUDIT_T, "R": {}, "X": {}}
    # ---- C4-R
    for fc in grid_classes:
        x = queries(fc)
        y = torch.full((n,), fc, dtype=torch.long)
        nets = {s: load(fc, s) for s in ORACLE_SEEDS}
        rows = {s: nets[s].class_emb.weight[fc].detach().clone() for s in ORACLE_SEEDS}
        L0 = losses(orig, diff, x, y, TIMESTEPS, dev)
        L = {}
        for ns, r in itertools.product(ORACLE_SEEDS, ORACLE_SEEDS):
            m = copy.deepcopy(nets[ns])
            with torch.no_grad():
                m.class_emb.weight[fc] = rows[r]
            L[(ns, r)] = losses(m, diff, x, y, TIMESTEPS, dev)
            del m
        st = row_swap_stats(L, TIMESTEPS)
        loo = []
        for r in ORACLE_SEEDS:
            for h in ORACLE_SEEDS:
                refs = [L[(o, r)][:, ai] for o in ORACLE_SEEDS if o != h]
                e = closeness_np(L[(h, r)][:, ai], L0[:, ai], refs, n_boot=n_boot, seed=seed)
                loo.append({"row": r, "held_out": h, **{k: e[k] for k in ("E", "E_lo", "E_hi", "snr")}})
        st["loo_common"] = [d["E"] for d in loo]
        st["loo_detail"] = loo
        out["R"][f"class{fc}"] = st
        print(f"[C4-R c{fc}] row wins at {st['row_wins_low_t']}/6 low t | H_indep {st['H_indep']:.2f} H_common {st['H_common']:.2f} "
              f"(H_r {st['H_r']}) | LOO common-row E " + " ".join("nan" if not np.isfinite(e) else f"{e:.2f}" for e in st["loo_common"]), flush=True)
        del nets
    out["R_verdicts"] = score_c4r(out["R"])
    print("[C4-R verdicts]", json.dumps(out["R_verdicts"]), flush=True)
    # ---- C4-X
    Ex = {}
    for fc in x_classes:
        x = queries(fc)
        y = torch.full((n,), null, dtype=torch.long)
        L0 = losses(orig, diff, x, y, TIMESTEPS, dev)[:, ai]
        refs = [losses(load(fc, s), diff, x, y, TIMESTEPS, dev)[:, ai] for s in ORACLE_SEEDS]
        rows = []
        for j in GRID_CLASSES:
            if j == fc:
                continue
            for s in ORACLE_SEEDS:
                Lu = losses(load(j, s), diff, x, y, TIMESTEPS, dev)[:, ai]
                e = closeness_np(Lu, L0, refs, n_boot=n_boot, seed=seed)
                rows.append({"removed_class": j, "seed": s, **{k: e[k] for k in ("E", "E_lo", "E_hi", "snr")}})
                print(f"[C4-X c{fc}] retrain(-{j}, s{s}) null E {e['E']:.3f} [{e['E_lo']:.3f},{e['E_hi']:.3f}]", flush=True)
        out["X"][f"class{fc}"] = rows
        Ex[f"class{fc}"] = [r["E"] for r in rows]
    out["X_verdicts"] = score_c4x(Ex)
    print("[C4-X verdicts]", json.dumps(out["X_verdicts"]), flush=True)
    p = os.path.join(root, "figures", out_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", p, flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
