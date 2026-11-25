"""Study C5 (registered in PREREGISTRATION.md, 2026-11-25 13:55 IST, before any C5 data or code): FID_O against one
retrain, as defined by Zheng, Tai & Yeh (CVPR 2026): FID between the unlearned model's and the oracle's generated
images "for all the classes" -- here with a retrain-to-retrain floor and a same-model sampling floor.

Per (model, label): 1,000 samples (DDIM 50, guidance 2.0), InceptionV3 2048-d pool3 features, cached as float32.
A re-draw differs from the original only in row k, so for labels != k its samples are the original copy-A samples
(same seeds -> identical images); only label k is sampled for re-draws.

Usage:  python -m src.analysis.c5_fido [results_root] [n_per_label] [class ...]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

from .c4_fd import frechet

RESET_SEEDS = tuple(range(100, 120))
CLASSES = (8, 1, 9)
LABELS = tuple(range(10))


def seed_for(kind: str, label: int, k: int = 0, s: int = 0, r: int = 100) -> int:
    """Registered sampler seeds."""
    if kind == "origA":
        return 70000 + label
    if kind == "origB":
        return 71000 + label
    if kind == "retrain":
        return 72000 + 100 * k + 10 * s + label
    if kind == "redraw":
        return 73000 + 100 * k + (r - 100)
    raise ValueError(kind)


def pooled_stats(feats: Sequence[np.ndarray]):
    """Gaussian fit to the concatenation of per-label feature blocks. Pure (unit-tested)."""
    f = np.concatenate([np.asarray(x, dtype=np.float64) for x in feats])
    return f.mean(0), np.cov(f, rowvar=False)


def fid_o(feats_a: Dict[int, np.ndarray], feats_b: Dict[int, np.ndarray], labels: Sequence[int] = LABELS) -> float:
    """FD between the label-pooled sample distributions of two models. Pure (unit-tested)."""
    return frechet(*pooled_stats([feats_a[l] for l in labels]), *pooled_stats([feats_b[l] for l in labels]))


def score_c5(per_class: Dict[str, Dict]) -> Dict:
    """per_class[c] = {'retrain_pairs': [3], 'same_model_floor': float, 'orig_vs_retrain': [3],
    'redraw_vs_retrain': [60], 'retrain_pairs_retained': [3]}.  Pure (unit-tested)."""
    v = {"O1": {}, "O2": {}, "O3": {}, "O4": {}, "O1_fraction": {}}
    for c, d in per_class.items():
        mx = max(d["retrain_pairs"])
        frac = float(np.mean([x <= mx for x in d["redraw_vs_retrain"]]))
        v["O1_fraction"][c] = frac
        v["O1"][c] = "held" if frac >= 0.25 else "failed"
        v["O2"][c] = "held" if np.mean(d["retrain_pairs"]) > 1.5 * d["same_model_floor"] else "failed"
        v["O3"][c] = "held" if min(d["orig_vs_retrain"]) > mx else "failed"
        v["O4"][c] = "held" if np.mean(d["retrain_pairs"]) > 1.25 * np.mean(d["retrain_pairs_retained"]) else "failed"
    for r in ("O1", "O2", "O4"):
        v[f"{r}_overall"] = "held" if sum(x == "held" for x in v[r].values()) >= 2 else "failed"
    v["O3_overall"] = "held" if all(x == "held" for x in v["O3"].values()) else "failed"
    return v


def main(root: str = "results", n: int = 1000, classes=CLASSES, resets=RESET_SEEDS, out_name: str = "c5_fido.json",
         cache_dir: str = None) -> Dict:
    import copy
    import gc

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
    inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    cache_dir = cache_dir or os.path.join(root, "c5_fido_cache")
    os.makedirs(cache_dir, exist_ok=True)
    orig = load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=dev)[0]

    def free():
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    def feats(name: str, model, label: int, seed: int) -> np.ndarray:
        p = os.path.join(cache_dir, f"{name}_y{label}_n{n}.npy")
        if os.path.exists(p):
            return np.load(p)
        imgs = sample_class(model, diff, label, n, 50, 2.0, seed, batch=250)
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 50):
                xb = ((imgs[i:i + 50].clamp(-1, 1) + 1) / 2).to(dev)
                fm = torch.nn.functional.adaptive_avg_pool2d(inc(xb)[0], (1, 1))
                out.append(fm.flatten(1).cpu().numpy().astype(np.float32))
        f = np.concatenate(out)
        np.save(p, f)
        print(f"  [C5] sampled+featurised {name} y={label} (seed {seed})", flush=True)
        del imgs, out, xb, fm
        free()
        return f

    FA = {l: feats("origA", orig, l, seed_for("origA", l)) for l in LABELS}
    FB = {l: feats("origB", orig, l, seed_for("origB", l)) for l in LABELS}
    floor = fid_o(FA, FB)
    print(f"[C5] same-model floor FID_O(origA, origB) = {floor:.3f}", flush=True)
    out = {"registered": "PREREGISTRATION.md, Study C5", "n_per_label": n, "feature_dim": 2048,
           "same_model_floor": floor, "classes": {}}
    for k in classes:
        R = {}
        for s in ORACLE_SEEDS:
            om = load_model(_ckpt(root, k, s), cfg, use_ema=True, device=dev)[0]
            R[s] = {l: feats(f"c{k}_retrain_s{s}", om, l, seed_for("retrain", l, k=k, s=s)) for l in LABELS}
            del om
            free()
        D = {}
        for r in resets:
            m = copy.deepcopy(orig)
            g = torch.Generator(device="cpu").manual_seed(r)  # identical construction to C3-A / C4-F
            with torch.no_grad():
                row = m.class_emb.weight[k]
                m.class_emb.weight[k] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
            D[r] = dict(FA)
            D[r][k] = feats(f"c{k}_reset_{r}", m, k, seed_for("redraw", k, k=k, r=r))
            del m
            free()
        retained = [l for l in LABELS if l != k]
        pairs = list(itertools.combinations(ORACLE_SEEDS, 2))
        d = {"retrain_pairs": [fid_o(R[a], R[b]) for a, b in pairs],
             "retrain_pairs_retained": [fid_o(R[a], R[b], retained) for a, b in pairs],
             "same_model_floor": floor,
             "orig_vs_retrain": [fid_o(FA, R[s]) for s in ORACLE_SEEDS],
             "origB_vs_retrain": [fid_o(FB, R[s]) for s in ORACLE_SEEDS],
             "redraw_vs_retrain_by_seed": {str(r): [fid_o(D[r], R[s]) for s in ORACLE_SEEDS] for r in resets}}
        d["redraw_vs_retrain"] = [x for r in resets for x in d["redraw_vs_retrain_by_seed"][str(r)]]
        out["classes"][f"class{k}"] = d
        print(f"[C5 c{k}] retrain pairs {np.round(d['retrain_pairs'], 2).tolist()} (retained-only "
              f"{np.round(d['retrain_pairs_retained'], 2).tolist()}) | floor {floor:.2f} | orig-retrain "
              f"{np.round(d['orig_vs_retrain'], 2).tolist()} | re-draw-retrain median {np.median(d['redraw_vs_retrain']):.2f} "
              f"[{min(d['redraw_vs_retrain']):.2f}, {max(d['redraw_vs_retrain']):.2f}]", flush=True)
        del R, D
        free()
    out["verdicts"] = score_c5(out["classes"])
    print("[C5 verdicts]", json.dumps(out["verdicts"]), flush=True)
    p = os.path.join(root, "figures", out_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", p, flush=True)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    cls = tuple(int(c) for c in sys.argv[3:]) or CLASSES
    main(root, n, cls)
