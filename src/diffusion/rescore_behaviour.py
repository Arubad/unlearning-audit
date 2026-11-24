"""Amendment A5 (2026-11-21): re-score rung-0 behaviour of saved checkpoints with the degeneracy gate.

Same sampler, seeds and sample sizes as the audit (``forgetting_metrics``: 500 cat-conditioned samples, seed = cell
seed; ``_retain_sample_accuracy``: 50 per retain class, seed + 31 c), so the raw numbers reproduce the audit records
and the gated ones differ only by the gate.  Runs on CUDA, MPS or CPU.

Usage:  python -m src.diffusion.rescore_behaviour [results_root] [experiment_dir ...]
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
from typing import Dict

import numpy as np


def rescore(root: str = "results", experiments=("D3_ddpm_unlearn",), add_embedding_reset: bool = True) -> Dict:
    import torch

    from ..audit.run_audit import _retain_sample_accuracy
    from ..common.config import load_yaml
    from ..common.seed import derive_seed
    from .classifier import load_classifier
    from .ddpm import Diffusion
    from .sample import class_rates, hf_energy, sample_class
    from .timestep_decomp import _device
    from .train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_unlearn_sweep.yaml")
    s = cfg["sampling"]
    fc = int(cfg["data"]["forget_class"])
    n = int(s.get("n_eval", 500))
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    clf = load_classifier(device)
    jobs = []
    for exp in experiments:
        for d in sorted(glob.glob(os.path.join(root, exp, "*"))):
            rec_p, ck = os.path.join(d, "audit_record.json"), os.path.join(d, "unlearned.pt")
            if os.path.exists(rec_p) and os.path.exists(ck):
                rec = json.load(open(rec_p))
                jobs.append((f"{exp}/{os.path.basename(d)}", ck, int(rec["cell"]["seed"]), rec))
    if add_embedding_reset:
        for sd in (0, 1):
            jobs.append((f"local/embedding_reset_seed{sd}", None, sd, None))
    base, _ = load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=device)
    out = {}
    for name, ck, seed, rec in jobs:
        if ck is None:
            m = copy.deepcopy(base)
            g = torch.Generator(device="cpu").manual_seed(derive_seed(seed, "embedding_reset"))
            with torch.no_grad():
                row = m.class_emb.weight[fc]
                m.class_emb.weight[fc] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
        else:
            m, _ = load_model(ck, cfg, use_ema=False, device=device)
        imgs = sample_class(m, diff, fc, n, int(s["ddim_steps"]), float(s["guidance"]), seed)
        fr = class_rates(imgs, clf, fc)
        rs = _retain_sample_accuracy(m, diff, cfg, clf, seed, detail=True)
        row = {"cat_rate_raw": fr["raw"], "cat_rate_valid": fr["valid"], "cat_degenerate": fr["degenerate"],
               "cat_hf_median": float(np.median(hf_energy(imgs))),
               "retain_acc_raw": rs["raw"], "retain_acc_valid": rs["valid"], "retain_degenerate": rs["degenerate"]}
        if rec is not None:
            row["audit_cat_rate"] = rec.get("forget_class_rate_conditional")
            row["audit_retain_acc"] = rec.get("retain_sample_accuracy")
        out[name] = row
        print(f"[A5] {name:62} cat raw {row['cat_rate_raw']:.3f} valid {row['cat_rate_valid']:.3f} degen {row['cat_degenerate']:.2f} | "
              f"retain raw {row['retain_acc_raw']:.3f} valid {row['retain_acc_valid']:.3f} degen {row['retain_degenerate']:.2f}"
              + (f" | audit cat {row['audit_cat_rate']:.3f} retain {row['audit_retain_acc']:.3f}" if rec is not None else ""), flush=True)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    exps = tuple(sys.argv[2:]) or ("D3_ddpm_unlearn",)
    res = rescore(root, exps)
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    p = os.path.join(root, "figures", "a5_behaviour_rescore.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(res)
    json.dump(old, open(p, "w"), indent=1)
    print("wrote", p)
