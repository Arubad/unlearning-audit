"""Confirmatory study C1 (registered 2026-11-21 in PREREGISTRATION.md, before any C1 data): scoring.

For a forget class k with >= 3 oracle retrains (CIFAR-10 minus class k, independent seeds):
  per-timestep signal-to-floor ratios on four query sets (``timestep_decomp.controls``), and a leave-one-oracle-out
  calibration: each oracle in turn is held out and scored, with plain E and timestep-standardised E, against the
  remaining oracles as the reference.  An exchangeable held-out retrain should score E ~ 1; the original scores 0
  by construction.

Verdict rules (verbatim from the registration):
  C1-1 localisation  forget_train snr_t <= 1.25 at every t <= 200 AND >= 1.5 at >= 3 of t in {500,600,700,800,900}
  C1-2 dilution      forget_train plain audit-5 SNR < 1.2
  C1-3 recovery      forget_train standardised audit-5 SNR >= 1.4
  C1-4 class-level   forget_test standardised SNR >= 1.4 AND |forget_test - forget_train| <= 0.3
  C1-5 controls      near and far standardised SNR <= 1.2
  C1-6 calibration   in every rotation the held-out oracle's standardised E lies in [0.6, 1.4], while plain E is
                     undefined (SNR < 1.2) in a majority of rotations
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np


def loo_calibration(root: str, forget_class: int, oracle_seeds: List[int], n: int = 256, seed: int = 0) -> Dict:
    import torch

    from ..audit.fingerprint import closeness_np, standardized_closeness_np
    from ..common.config import load_yaml
    from ..diffusion.data import cifar_labels, load_cifar
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.timestep_decomp import AUDIT_T, _device, losses
    from ..diffusion.train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    ds = load_cifar(cfg["data"].get("root", "data"))
    lab = cifar_labels(ds)
    idx = np.random.default_rng(seed).choice(np.where(lab == forget_class)[0], n, replace=False)
    x = torch.stack([ds[int(i)][0] for i in idx])
    y = torch.tensor(lab[idx], dtype=torch.long)
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    ck = os.path.join(root, "checkpoints")
    cls = "" if forget_class == 3 else f"_c{forget_class}"
    l0 = losses(load_model(os.path.join(ck, "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=device)[0], diff, x, y, AUDIT_T, device)
    lo = {s: losses(load_model(os.path.join(ck, f"ddpm_cifar10_retain{cls}_f0.1_s{s}.pt"), cfg, use_ema=True, device=device)[0], diff, x, y, AUDIT_T, device)
          for s in oracle_seeds}
    rot = []
    for held in oracle_seeds:
        refs = [lo[s] for s in oracle_seeds if s != held]
        pl = closeness_np(lo[held], l0, refs, n_boot=1000, seed=seed)
        st = standardized_closeness_np(lo[held], l0, refs, n_boot=1000, seed=seed)
        rot.append({"held_out": held, "plain": pl, "std": st})
        print(f"[C1 LOO c{forget_class}] held-out oracle s{held}: plain E {pl['E']:.2f} (snr {pl['snr']:.2f}) | "
              f"std E {st['E']:.2f} [{st['E_lo']:.2f},{st['E_hi']:.2f}] (snr {st['snr']:.2f})", flush=True)
    return {"forget_class": forget_class, "oracle_seeds": list(oracle_seeds), "rotations": rot}


def score(ctrl: Dict, loo: Dict) -> Dict:
    T = ctrl["timesteps"]
    s = ctrl["sets"]
    ft = s["forget_train"]
    lo_t = [ft["snr_t"][T.index(t)] for t in T if t <= 200]
    hi_t = [ft["snr_t"][T.index(t)] for t in (500, 600, 700, 800, 900)]
    v = {
        "C1-1": all(x <= 1.25 for x in lo_t) and sum(x >= 1.5 for x in hi_t) >= 3,
        "C1-2": ft["snr_plain_audit"] < 1.2,
        "C1-3": ft["snr_std_audit"] >= 1.4,
        "C1-4": s["forget_test"]["snr_std_audit"] >= 1.4 and abs(s["forget_test"]["snr_std_audit"] - ft["snr_std_audit"]) <= 0.3,
        "C1-5": s["near"]["snr_std_audit"] <= 1.2 and s["far"]["snr_std_audit"] <= 1.2,
    }
    rs = loo["rotations"]
    v["C1-6"] = all(0.6 <= r["std"]["E"] <= 1.4 for r in rs) and sum(not np.isfinite(r["plain"]["E"]) for r in rs) > len(rs) / 2
    return {k: ("held" if ok else "failed") for k, ok in v.items()}


def main(root: str = "results", classes=(1, 3), oracle_seeds=(0, 1, 2)) -> Dict:
    from ..diffusion.timestep_decomp import controls

    out = {}
    for fc in classes:
        ctrl = controls(root, forget_class=fc, oracle_seeds=oracle_seeds)
        loo = loo_calibration(root, fc, list(oracle_seeds))
        out[f"class{fc}"] = {"controls": ctrl, "loo": loo, "verdicts": score(ctrl, loo)}
        print(f"[C1] class {fc} verdicts:", json.dumps(out[f'class{fc}']['verdicts']), flush=True)
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    with open(os.path.join(root, "figures", "c1_report.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
