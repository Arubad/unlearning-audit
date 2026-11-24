"""Confirmatory study C3-A (registered in PREREGISTRATION.md before data): does the choice of reference -- removed
label vs null label -- change the audit verdict on unlearned models?

For forget class k with three oracle retrains, on the forget-train query set of ``label_diag`` (n = 256, same RNG
draw), compute for each audited model:
  removed-label closeness  forget-label standardised E (plain E where the oracles' std SNR < 1.2; amendment)
  null-label closeness     null-label plain E
  behaviour                gated forget-class rate of 200 class-k samples (DDIM 50, guidance 2.0, seed 4242)
Audited models: do_nothing, 20 embedding_reset draws (seeds 100-119), and the C3 unlearning cells on disk.

Usage:  python -m src.analysis.c3_consequence [results_root] [class ...]
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
from typing import Dict

import numpy as np

RESET_SEEDS = tuple(range(100, 120))
METHODS = ("gradient_ascent", "gradient_difference", "label_annealing", "variational_ewc", "retain_finetune")


def _e(d: Dict) -> Dict:
    return {k: d.get(k) for k in ("E", "E_lo", "E_hi", "snr")}


def audit_class(root: str, fc: int, exp: str = "C3_unlearn", n: int = 256, seed: int = 0, n_samples: int = 200,
                sample_seed: int = 4242, n_boot: int = 1000) -> Dict:
    import torch

    from ..audit.fingerprint import closeness_np, standardized_closeness_np
    from ..common.config import load_yaml
    from ..diffusion.classifier import load_classifier
    from ..diffusion.data import cifar_labels, load_cifar
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.sample import class_rates, sample_class
    from ..diffusion.timestep_decomp import AUDIT_T, TIMESTEPS, _device, losses
    from ..diffusion.train import load_model
    from .label_diag import ORACLE_SEEDS, _ckpt

    dev = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    tr = load_cifar(cfg["data"].get("root", "data"))
    ltr = cifar_labels(tr)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.where(ltr == fc)[0], n, replace=False)  # = label_diag's forget_train draw (first draw)
    x = torch.stack([tr[int(i)][0] for i in idx])
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=dev)
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    clf = load_classifier(dev)
    orig = load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=dev)[0]
    null = int(orig.null_class)
    Y = {"forget_label": torch.full((n,), fc, dtype=torch.long), "null_label": torch.full((n,), null, dtype=torch.long)}

    def L(m):
        return {c: losses(m, diff, x, y, TIMESTEPS, dev)[:, ai] for c, y in Y.items()}

    def behaviour(m):
        imgs = sample_class(m, diff, fc, n_samples, 50, 2.0, sample_seed)
        r = class_rates(imgs, clf, fc)
        return {"rate_valid": r["valid"], "rate_raw": r["raw"], "degenerate": r["degenerate"]}

    L0 = L(orig)
    O, ob = [], {}
    for s in ORACLE_SEEDS:
        om = load_model(_ckpt(root, fc, s), cfg, use_ema=True, device=dev)[0]
        O.append(L(om))
        ob[f"o{s}"] = behaviour(om)
        del om
    refs = {c: [o[c] for o in O] for c in Y}
    std_snr = standardized_closeness_np(L0["forget_label"], L0["forget_label"], refs["forget_label"], n_boot=10)["snr"]
    removed_stat = "std" if std_snr >= 1.2 else "plain"
    print(f"[C3 c{fc}] oracle forget-label std SNR {std_snr:.2f} -> removed-label statistic: {removed_stat}; oracle gated rates "
          + " ".join(f"{k} {v['rate_valid']:.3f}" for k, v in ob.items()), flush=True)

    def score(Lu):
        fl_fn = standardized_closeness_np if removed_stat == "std" else closeness_np
        return {"removed_label": _e(fl_fn(Lu["forget_label"], L0["forget_label"], refs["forget_label"], n_boot=n_boot, seed=seed)),
                "removed_label_plain": _e(closeness_np(Lu["forget_label"], L0["forget_label"], refs["forget_label"], n_boot=n_boot, seed=seed)),
                "null_label": _e(closeness_np(Lu["null_label"], L0["null_label"], refs["null_label"], n_boot=n_boot, seed=seed))}

    models = {}
    models["do_nothing"] = {**score(L0), **behaviour(orig)}
    for rs in RESET_SEEDS:
        m = copy.deepcopy(orig)
        g = torch.Generator(device="cpu").manual_seed(rs)
        with torch.no_grad():
            row = m.class_emb.weight[fc]
            m.class_emb.weight[fc] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
        models[f"reset_{rs}"] = {**score(L(m)), **behaviour(m)}
        del m
    for d in sorted(glob.glob(os.path.join(root, exp, f"forget_class{fc}_method*_seed*"))):
        ck = os.path.join(d, "unlearned.pt")
        if os.path.exists(ck):
            m = load_model(ck, cfg, use_ema=False, device=dev)[0]
            models[os.path.basename(d)] = {**score(L(m)), **behaviour(m)}
            del m
    for k, v in models.items():
        r, nl = v["removed_label"], v["null_label"]
        print(f"[C3 c{fc}] {k:52} removed E {r['E']:8.2f} | null E {nl['E']:8.2f} [{nl['E_lo']:.2f},{nl['E_hi']:.2f}] | gated rate {v['rate_valid']:.3f}", flush=True)
    return {"forget_class": fc, "removed_label_statistic": removed_stat, "oracle_std_snr": std_snr, "oracle_behaviour": ob, "models": models}


def score_c3a(res: Dict) -> Dict:
    """Registered C3-A verdicts (PREREGISTRATION.md, 'Confirmatory study C3')."""
    M = res["models"]
    thr = max(v["rate_valid"] for v in res["oracle_behaviour"].values()) + 0.05
    resets = [v for k, v in M.items() if k.startswith("reset_")]
    E = np.array([v["removed_label"]["E"] for v in resets], dtype=float)
    passes = [bool(np.isfinite(e) and 0.5 <= e <= 1.5 and v["rate_valid"] <= thr) for e, v in zip(E, resets)]
    v: Dict = {"n_reset_pass": int(sum(passes)), "reset_E_sd": float(np.nanstd(E, ddof=1)) if np.isfinite(E).sum() > 1 else float("nan"),
               "behaviour_threshold": thr}
    v["C3-A1"] = "held" if sum(passes) >= 1 else "failed"
    v["C3-A2"] = "held" if np.isfinite(v["reset_E_sd"]) and v["reset_E_sd"] >= 0.3 else "failed"
    meth = {k: x for k, x in M.items() if "_method" in k}
    if meth:
        lo = {k: x["null_label"]["E_lo"] for k, x in meth.items()}
        v["C3-A3"] = "held" if all(not (l is not None and np.isfinite(l) and l >= 0.8) for l in lo.values()) else "failed"
        la = [x["null_label"]["E"] for k, x in meth.items() if "label_annealing" in k]
        v["C3-A4"] = ("held" if len(la) == 2 and all(np.isfinite(e) and e < 0 for e in la) else "failed") if la else "untestable"
    else:
        v["C3-A3"] = v["C3-A4"] = "untestable"
    return v


def main(root: str = "results", classes=(8, 1)) -> Dict:
    p = os.path.join(root, "figures", "c3a_consequence.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    for fc in classes:
        res = audit_class(root, fc)
        res["c3a_verdicts"] = score_c3a(res)
        out[f"class{fc}"] = res
        print(f"[C3-A rules] class {fc}:", json.dumps(res["c3a_verdicts"]), flush=True)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=1, default=float)
    print("wrote", p)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(root, tuple(int(c) for c in sys.argv[2:]) or (8, 1))
