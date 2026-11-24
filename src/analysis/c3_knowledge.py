"""Exploratory follow-up to C3-A (NOT registered; predictions written in EXPERIMENT_LOG before running).

Question: a zero-step embedding_reset passes the removed-label audit for ship, while the null-label audit scores it 0
by construction. Is the forget-class knowledge still in the network? This test is not tautological: we ask whether
the class can be re-attached by optimising ONLY the forget-label embedding row (256 numbers; network frozen).
  - reset models:   the network is the original's, so a re-attached row should restore class-k generation;
  - oracle retrains: the network never saw class k, so a row alone should restore it much less.
Also reported: the unconditional (null-label) class-k rate of the original vs the oracles.

Usage:  python -m src.analysis.c3_knowledge [results_root] [class] [steps]
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

RESET_SEEDS = (102, 109, 110)  # three of the six resets that passed the removed-label audit for ship (C3-A1)
PASSING = {8: (102, 109, 110), 1: (100, 104, 106), 9: (100, 101, 104)}  # first three C3-A1 passers per class (c3a_consequence.json)


def main(root: str = "results", fc: int = 8, steps: int = 400, lr: float = 0.05, batch: int = 64,
         n_relearn: int = 1000, n_samples: int = 200, n_uncond: int = 1000, sample_seed: int = 4242) -> dict:
    import torch

    from ..common.config import load_yaml
    from ..diffusion.classifier import load_classifier
    from ..diffusion.data import cifar_labels, load_cifar
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.sample import class_rates, sample_class
    from ..diffusion.timestep_decomp import _device
    from ..diffusion.train import load_model
    from .label_diag import ORACLE_SEEDS, _ckpt

    dev = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    tr = load_cifar(cfg["data"].get("root", "data"))
    ltr = cifar_labels(tr)
    rng = np.random.default_rng(0)
    cls_idx = np.where(ltr == fc)[0]
    audit_idx = rng.choice(cls_idx, 256, replace=False)  # same first draw as label_diag / c3_consequence
    pool = np.setdiff1d(cls_idx, audit_idx)
    ridx = np.random.default_rng(1).choice(pool, n_relearn, replace=False)  # relearning images, disjoint from audit set
    xr = torch.stack([tr[int(i)][0] for i in ridx])
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=dev)
    clf = load_classifier(dev)
    orig = load_model(os.path.join(root, "checkpoints", "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=dev)[0]

    def rate(m, y):
        return class_rates(sample_class(m, diff, y, n_samples if y is not None else n_uncond, 50, 2.0, sample_seed), clf, fc)

    def relearn_row(m, seed):
        """Optimise only row `fc` of the label embedding (all other weights frozen; other rows get zero gradient,
        so Adam leaves them exactly unchanged)."""
        m = copy.deepcopy(m).eval()  # eval: no CFG dropout inside the model; labels are passed explicitly
        for p in m.parameters():
            p.requires_grad_(False)
        W = m.class_emb.weight
        W.requires_grad_(True)
        W_before = W.detach().clone()
        mask = torch.zeros_like(W); mask[fc] = 1.0
        opt = torch.optim.Adam([W], lr=lr)
        g = torch.Generator(device="cpu").manual_seed(seed)
        hist = []
        for s in range(steps):
            bi = torch.randint(0, len(xr), (batch,), generator=g)
            x0 = xr[bi].to(dev)
            y = torch.full((batch,), fc, device=dev, dtype=torch.long)
            loss = diff.loss(m, x0, y)
            opt.zero_grad(); loss.backward(); W.grad.mul_(mask); opt.step()
            if s % 50 == 0 or s == steps - 1:
                hist.append((s, float(loss)))
        W.requires_grad_(False)
        other = torch.cat([W[:fc], W[fc + 1:]]) - torch.cat([W_before[:fc], W_before[fc + 1:]])
        assert float(other.abs().max()) == 0.0, "non-forget rows changed"
        return m, hist

    out = {"forget_class": fc, "steps": steps, "lr": lr, "batch": batch, "n_relearn": n_relearn, "models": {}}
    t0 = time.time()
    out["uncond_rate"] = {"original": rate(orig, None)}
    for s in ORACLE_SEEDS:
        om = load_model(_ckpt(root, fc, s), cfg, use_ema=True, device=dev)[0]
        out["uncond_rate"][f"oracle_s{s}"] = rate(om, None)
        r0 = rate(om, fc)
        mr, h = relearn_row(om, 10 + s)
        out["models"][f"oracle_s{s}"] = {"before": r0, "after": rate(mr, fc), "loss_hist": h}
        print(f"[knowledge c{fc}] oracle_s{s}: rate before {r0['valid']:.3f} after row-relearn {out['models'][f'oracle_s{s}']['after']['valid']:.3f} "
              f"| uncond {out['uncond_rate'][f'oracle_s{s}']['valid']:.3f} | {time.time()-t0:.0f}s", flush=True)
        del om, mr
    print(f"[knowledge c{fc}] original uncond rate {out['uncond_rate']['original']['valid']:.3f}", flush=True)
    for rs in PASSING.get(fc, RESET_SEEDS):
        m = copy.deepcopy(orig)
        g = torch.Generator(device="cpu").manual_seed(rs)
        with torch.no_grad():
            row = m.class_emb.weight[fc]
            m.class_emb.weight[fc] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
        r0 = rate(m, fc)
        mr, h = relearn_row(m, 20 + rs)
        out["models"][f"reset_{rs}"] = {"before": r0, "after": rate(mr, fc), "loss_hist": h}
        print(f"[knowledge c{fc}] reset_{rs}: rate before {r0['valid']:.3f} after row-relearn {out['models'][f'reset_{rs}']['after']['valid']:.3f} | {time.time()-t0:.0f}s", flush=True)
        del m, mr
    p = os.path.join(root, "figures", f"c3_knowledge_c{fc}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", p, flush=True)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    fc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    main(root, fc, steps)
