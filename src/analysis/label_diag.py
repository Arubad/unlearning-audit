"""Amendment A6 (post-hoc, 2026-11-22) diagnostics and confirmatory study C2 scoring (registered in PREREGISTRATION.md
before any C2 data).

An exact retrain without class k never sees label k, so its class-embedding row k keeps its random initial value
(no gradient; weight decay 0).  Querying the retrain with label k therefore measures its response to a seed-specific
random vector, not a counterfactual.  The null (unconditional, CFG) label is trained in every model and gives a
well-posed counterfactual.  This module measures, for a forget class k with three oracle retrains:

  embedding_check   how far row k (and every other row) moved from the seed's initial value
  label_controls    per-timestep signal/floor ratios, plain / standardised audit SNR, the max/min oracle-pair distance
                    ratio, and leave-one-oracle-out closeness, on the four query sets of ``timestep_decomp.controls``
                    (same RNG order), under the forget label and under the null label
  score_c2          the registered C2 verdicts

Usage:  python -m src.analysis.label_diag [results_root] [class ...]     (default classes: 8)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np

ORACLE_SEEDS = (0, 1, 2)


def _ckpt(root: str, fc: int, s: int) -> str:
    cls = "" if fc == 3 else f"_c{fc}"
    return os.path.join(root, "checkpoints", f"ddpm_cifar10_retain{cls}_f0.1_s{s}.pt")


def embedding_check(root: str, fc: int, oracle_seeds=ORACLE_SEEDS) -> Dict:
    import torch

    from ..common.config import load_yaml
    from ..common.seed import derive_seed, seed_everything
    from ..diffusion.unet import build_unet

    cfg = load_yaml("configs/diffusion_base.yaml")
    out = {}
    for s in oracle_seeds:
        seed_everything(derive_seed(s, "train_retain"))  # the oracle's init stream (train.train_ddpm)
        init = build_unet(cfg).class_emb.weight.detach().clone()
        d = torch.load(_ckpt(root, fc, s), map_location="cpu", weights_only=False)
        W, We = d["model"]["class_emb.weight"].float(), d["ema"]["class_emb.weight"].float()
        moved = (W - init).norm(dim=1)
        out[f"s{s}"] = {"row_moved_from_init": moved.tolist(), "forget_row_moved": float(moved[fc]),
                        "forget_row_moved_rel": float(moved[fc] / init[fc].norm()),
                        "forget_row_moved_ema": float((We - init).norm(dim=1)[fc]),
                        "other_rows_moved_mean": float(np.mean([moved[c].item() for c in range(10) if c != fc]))}
        print(f"[emb c{fc}] oracle s{s}: forget row moved {moved[fc]:.2e} (ema {out[f's{s}']['forget_row_moved_ema']:.2e}); "
              f"other rows mean {out[f's{s}']['other_rows_moved_mean']:.3f}", flush=True)
    return out


def label_controls(root: str, fc: int, oracle_seeds=ORACLE_SEEDS, n: int = 256, seed: int = 0, n_boot: int = 1000) -> Dict:
    import torch

    from ..audit.fingerprint import closeness_np, standardized_closeness_np
    from ..common.config import load_yaml
    from ..diffusion.data import cifar_labels, load_cifar
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.timestep_decomp import AUDIT_T, FC, NEAR, TIMESTEPS, _device, losses, rms
    from ..diffusion.train import load_model

    dev = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    tr, te = load_cifar(cfg["data"].get("root", "data")), load_cifar(cfg["data"].get("root", "data"), train=False)
    ltr, lte = cifar_labels(tr), cifar_labels(te)
    # identical query sets to timestep_decomp.controls (same RNG draws, same order)
    rng = np.random.default_rng(seed)
    others = [c for c in range(10) if c != fc]
    near = [c for c in NEAR if c != fc] if fc in NEAR or fc == FC else [c for c in (0, 1, 8, 9) if c != fc]
    far = [c for c in others if c not in near]
    sets = {"forget_train": (tr, rng.choice(np.where(ltr == fc)[0], n, replace=False)),
            "forget_test": (te, rng.choice(np.where(lte == fc)[0], n, replace=False)),
            "near": (tr, rng.choice(np.where(np.isin(ltr, near))[0], n, replace=False)),
            "far": (tr, rng.choice(np.where(np.isin(ltr, far))[0], n, replace=False))}
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=dev)
    M = {"orig": load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=dev)[0]}
    for s in oracle_seeds:
        M[f"o{s}"] = load_model(_ckpt(root, fc, s), cfg, use_ema=True, device=dev)[0]
    on = [f"o{s}" for s in oracle_seeds]
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    null = int(M["orig"].null_class)
    out = {"timesteps": TIMESTEPS, "audit_t": AUDIT_T, "forget_class": fc, "near": near, "far": far, "oracle_seeds": list(oracle_seeds), "labels": {}}
    for cond in ("forget_label", "null_label"):
        res = {}
        for qn, (ds, idx) in sets.items():
            x = torch.stack([ds[int(i)][0] for i in idx])
            lab = (ltr if ds is tr else lte)[idx]
            # forget_label: each image's own label (= fc on the forget sets, its retain label on near/far)
            y = torch.tensor(lab, dtype=torch.long) if cond == "forget_label" else torch.full((len(idx),), null, dtype=torch.long)
            L = {k: losses(m, diff, x, y, TIMESTEPS, dev) for k, m in M.items()}
            O = [L[k] for k in on]
            pw = np.array([rms(a, b) for a, b in itertools.combinations(O, 2)])
            fl, sg = pw.mean(0), np.mean([rms(L["orig"], o) for o in O], axis=0)
            pl = closeness_np(L["orig"][:, ai], L["orig"][:, ai], [o[:, ai] for o in O], n_boot=10)
            st = standardized_closeness_np(L["orig"][:, ai], L["orig"][:, ai], [o[:, ai] for o in O], n_boot=10)
            r = {"snr_t": (sg / fl).tolist(), "snr_plain": pl["snr"], "snr_std": st["snr"], "pair_ratio_t": (pw.max(0) / pw.min(0)).tolist(),
                 "loss_gap_t": (np.mean([o.mean(0) for o in O], axis=0) - L["orig"].mean(0)).tolist()}
            if qn == "forget_train":
                r["loo"] = []
                for h in range(len(O)):
                    refs = [O[j][:, ai] for j in range(len(O)) if j != h]
                    r["loo"].append({"held_out": oracle_seeds[h], "plain": closeness_np(O[h][:, ai], L["orig"][:, ai], refs, n_boot=n_boot, seed=seed),
                                     "std": standardized_closeness_np(O[h][:, ai], L["orig"][:, ai], refs, n_boot=n_boot, seed=seed)})
            res[qn] = r
            print(f"[{cond} c{fc}] {qn:12} snr_t " + " ".join(f"{v:.2f}" for v in r["snr_t"]) + f" | plain {r['snr_plain']:.2f} std {r['snr_std']:.2f}"
                  f" | max pair ratio t<=400 {max(r['pair_ratio_t'][i] for i, t in enumerate(TIMESTEPS) if t <= 400):.1f}", flush=True)
            for lo in r.get("loo", []):
                print(f"      LOO held-out s{lo['held_out']}: plain E {lo['plain']['E']:.2f} [{lo['plain']['E_lo']:.2f},{lo['plain']['E_hi']:.2f}] "
                      f"| std E {lo['std']['E']:.2f} [{lo['std']['E_lo']:.2f},{lo['std']['E_hi']:.2f}]", flush=True)
        out["labels"][cond] = res
    return out


def score_c2(ctrl: Dict, emb: Dict) -> Dict:
    """Registered C2 verdicts (PREREGISTRATION.md, 'Confirmatory study C2')."""
    T = ctrl["timesteps"]
    nl, fl = ctrl["labels"]["null_label"], ctrl["labels"]["forget_label"]
    ft = nl["forget_train"]
    low = [i for i, t in enumerate(T) if t <= 400]
    v: Dict[str, str] = {}
    c0 = ft["snr_plain"] >= 1.5
    v["C2-0"] = "held" if c0 else "failed"
    if c0:
        v["C2-A"] = "held" if all(np.isfinite(r["plain"]["E"]) and 0.8 <= r["plain"]["E"] <= 1.2 for r in ft["loo"]) else "failed"
        te = nl["forget_test"]["snr_plain"]
        v["C2-C"] = "held" if (te >= 1.5 and abs(te - ft["snr_plain"]) <= 0.2 * ft["snr_plain"]) else "failed"
    else:
        v["C2-A"] = v["C2-C"] = "untestable"
    v["C2-B"] = "held" if all(ft["pair_ratio_t"][i] <= 1.3 for i in low) else "failed"
    v["C2-D"] = "held" if (nl["near"]["snr_plain"] <= 1.25 and nl["far"]["snr_plain"] <= 1.25) else "failed"
    mid = [i for i, t in enumerate(T) if 50 <= t <= 400]
    hi = [i for i, t in enumerate(T) if t >= 700]
    v["C2-E"] = "held" if max(ft["snr_t"][i] for i in mid) >= 1.5 * max(ft["snr_t"][i] for i in hi) else "failed"
    v["C2-F"] = "held" if any(fl["forget_train"]["pair_ratio_t"][i] >= 2.0 for i in low) else "failed"
    v["C2-G"] = "held" if any((not np.isfinite(r["std"]["E"])) or not (0.6 <= r["std"]["E"] <= 1.4) for r in fl["forget_train"]["loo"]) else "failed"
    # registered as relative movement < 1e-5 (amended before C2 data from an absolute 1e-6: float32 init round-off
    # between torch builds is ~1e-7 relative)
    v["sanity_forget_row_untrained"] = all(e["forget_row_moved_rel"] < 1e-5 for e in emb.values())
    return v


def score_c3b(ctrl: Dict, emb: Dict) -> Dict:
    """Registered C3-B verdicts (PREREGISTRATION.md, 'Confirmatory study C3'): the C2 rules, with C2-B replaced by
    C3-B-B (revised after C2, tested only on the fresh class): at every t <= 400 the null-label max/min oracle-pair
    ratio is <= 1.5 AND below the forget-label ratio at the same t."""
    v = score_c2(ctrl, emb)
    T = ctrl["timesteps"]
    nl = ctrl["labels"]["null_label"]["forget_train"]["pair_ratio_t"]
    fl = ctrl["labels"]["forget_label"]["forget_train"]["pair_ratio_t"]
    low = [i for i, t in enumerate(T) if t <= 400]
    v["C3-B-B"] = "held" if all(nl[i] <= 1.5 and nl[i] < fl[i] for i in low) else "failed"
    v["claim_b_confirmed"] = all(v[k] == "held" for k in ("C2-0", "C2-A", "C3-B-B", "C2-D"))
    return v


def main(root: str = "results", classes=(8,)) -> Dict:
    p = os.path.join(root, "figures", "label_diag.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    for fc in classes:
        emb = embedding_check(root, fc)
        ctrl = label_controls(root, fc)
        out[f"class{fc}"] = {"embedding": emb, "controls": ctrl, "c2_verdicts": score_c2(ctrl, emb)}
        if fc == 9:  # C3-B (truck) uses the revised homogeneity rule
            out[f"class{fc}"]["c3b_verdicts"] = score_c3b(ctrl, emb)
            print(f"[C3-B rules] class {fc}:", json.dumps(out[f"class{fc}"]["c3b_verdicts"]), flush=True)
        print(f"[C2-rules] class {fc}:", json.dumps(out[f'class{fc}']['c2_verdicts']), flush=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", p)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(root, tuple(int(c) for c in sys.argv[2:]) or (8,))
