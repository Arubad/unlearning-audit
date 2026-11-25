"""Study C4-F (registered in PREREGISTRATION.md before any C4 data): removed-label Frechet distance, an analogue of
FID_O (Zheng, Tai & Yeh, CVPR 2026: FID between the unlearned model's and the oracle's generated images "for all the
classes") restricted to the forgotten label.

Per class k in {8, 1, 9}: original, 3 retrains and the 20 C3-A re-draws each sample 1,000 images under label k
(50 DDIM steps, guidance 2.0, independent sampler seed per model); original and retrains also sample label 0.
Features: pytorch-fid InceptionV3 768-d block.  Per-model feature means/covariances are cached so the run resumes.

Usage:  python -m src.analysis.c4_fd [results_root] [n_samples] [class ...]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np

RESET_SEEDS = tuple(range(100, 120))
CLASSES = (8, 1, 9)
CONTRAST_LABEL = 0


def frechet(mu1, s1, mu2, s2) -> float:
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))  # scipy >= 1.16 has no `disp` argument
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def score_c4f(per_class: Dict[str, Dict]) -> Dict:
    """per_class[c] = {'retrain_pairs_k': [3 FDs], 'retrain_pairs_0': [3 FDs], 'redraw_vs_retrain': [60 FDs],
    'orig_vs_retrain': [3 FDs]}.  Pure (unit-tested)."""
    v = {"F1": {}, "F2": {}, "F3": {}, "F1_fraction": {}}
    for c, d in per_class.items():
        mx = max(d["retrain_pairs_k"])
        frac = float(np.mean([x <= mx for x in d["redraw_vs_retrain"]]))
        v["F1_fraction"][c] = frac
        v["F1"][c] = "held" if frac >= 0.25 else "failed"
        v["F2"][c] = "held" if np.mean(d["retrain_pairs_k"]) > 1.5 * np.mean(d["retrain_pairs_0"]) else "failed"
        v["F3"][c] = "held" if min(d["orig_vs_retrain"]) > mx else "failed"
    v["F1_overall"] = "held" if sum(x == "held" for x in v["F1"].values()) >= 2 else "failed"
    v["F2_overall"] = "held" if sum(x == "held" for x in v["F2"].values()) >= 2 else "failed"
    v["F3_overall"] = "held" if all(x == "held" for x in v["F3"].values()) else "failed"
    return v


def main(root: str = "results", n_samples: int = 1000, classes=CLASSES, resets=RESET_SEEDS, out_name: str = "c4_fd.json",
         cache_dir: str = None) -> Dict:
    import copy

    import torch
    from pytorch_fid.inception import InceptionV3

    from ..common.config import load_yaml
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.sample import sample_class
    from ..diffusion.timestep_decomp import _device
    from ..diffusion.train import load_model
    from .label_diag import ORACLE_SEEDS, _ckpt

    dev = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=dev)
    inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[768]]).to(dev).eval()
    cache_dir = cache_dir or os.path.join(root, "c4_fd_cache")
    os.makedirs(cache_dir, exist_ok=True)
    orig = load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=dev)[0]

    def _free():  # 2026-11-25: MPS caching allocator grew to 16 GB on a 16 GB laptop; release between models (no numerical effect)
        import gc
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    def stats(name, model, label, seed):
        p = os.path.join(cache_dir, f"{name}_y{label}_n{n_samples}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["mu"], z["sigma"]
        imgs = sample_class(model, diff, label, n_samples, 50, 2.0, seed, batch=250)
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 50):  # 2026-11-25: 250 -> 50 under the memory cap; eval-mode Inception is per-image
                xb = ((imgs[i:i + 50].clamp(-1, 1) + 1) / 2).to(dev)
                fm = torch.nn.functional.adaptive_avg_pool2d(inc(xb)[0], (1, 1))  # as pytorch_fid.fid_score for dims < 2048
                feats.append(fm.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float64))
        f = np.concatenate(feats)
        mu, sigma = f.mean(0), np.cov(f, rowvar=False)
        np.savez(p, mu=mu, sigma=sigma)
        print(f"  [C4-F] sampled+featurised {name} y={label} (seed {seed})", flush=True)
        del imgs, feats, f, xb, fm
        _free()
        return mu, sigma

    out = {"registered": "PREREGISTRATION.md, Study C4 (C4-F)", "n_samples": n_samples, "feature_dim": 768, "classes": {}}
    for fc in classes:
        names = ["original"] + [f"retrain_s{s}" for s in ORACLE_SEEDS] + [f"reset_{r}" for r in resets]
        seed_of = {nm: 50000 + 1000 * fc + i for i, nm in enumerate(names)}
        S, S0 = {}, {}
        S["original"] = stats(f"c{fc}_original", orig, fc, seed_of["original"])
        S0["original"] = stats(f"c{fc}_original", orig, CONTRAST_LABEL, seed_of["original"] + 500)
        for s in ORACLE_SEEDS:
            om = load_model(_ckpt(root, fc, s), cfg, use_ema=True, device=dev)[0]
            S[f"retrain_s{s}"] = stats(f"c{fc}_retrain_s{s}", om, fc, seed_of[f"retrain_s{s}"])
            S0[f"retrain_s{s}"] = stats(f"c{fc}_retrain_s{s}", om, CONTRAST_LABEL, seed_of[f"retrain_s{s}"] + 500)
            del om
            _free()
        for rs in resets:
            m = copy.deepcopy(orig)
            g = torch.Generator(device="cpu").manual_seed(rs)  # identical construction to C3-A
            with torch.no_grad():
                row = m.class_emb.weight[fc]
                m.class_emb.weight[fc] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
            S[f"reset_{rs}"] = stats(f"c{fc}_reset_{rs}", m, fc, seed_of[f"reset_{rs}"])
            del m
            _free()
        R = [f"retrain_s{s}" for s in ORACLE_SEEDS]
        fd = lambda a, b, T=S: frechet(*T[a], *T[b])
        d = {"retrain_pairs_k": [fd(a, b) for a, b in itertools.combinations(R, 2)],
             "retrain_pairs_0": [frechet(*S0[a], *S0[b]) for a, b in itertools.combinations(R, 2)],
             "orig_vs_retrain": [fd("original", r) for r in R],
             "orig_vs_retrain_0": [frechet(*S0["original"], *S0[r]) for r in R],
             "redraw_vs_retrain": [fd(f"reset_{rs}", r) for rs in resets for r in R],
             "redraw_vs_retrain_by_seed": {str(rs): [fd(f"reset_{rs}", r) for r in R] for rs in resets}}
        out["classes"][f"class{fc}"] = d
        print(f"[C4-F c{fc}] retrain pairs (label k) {np.round(d['retrain_pairs_k'], 1).tolist()} | (label 0) "
              f"{np.round(d['retrain_pairs_0'], 1).tolist()} | orig-retrain {np.round(d['orig_vs_retrain'], 1).tolist()} | "
              f"re-draw-retrain median {np.median(d['redraw_vs_retrain']):.1f} [{min(d['redraw_vs_retrain']):.1f}, {max(d['redraw_vs_retrain']):.1f}]", flush=True)
    out["verdicts"] = score_c4f(out["classes"])
    print("[C4-F verdicts]", json.dumps(out["verdicts"]), flush=True)
    p = os.path.join(root, "figures", out_name)
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", p, flush=True)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    cls = tuple(int(c) for c in sys.argv[3:]) or CLASSES
    main(root, n, cls)
