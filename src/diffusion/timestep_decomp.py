"""Amendment A4 (2026-11-21, post-hoc / exploratory): decompose the original-vs-oracle loss difference by timestep and
by conditioning.  Runs from existing checkpoints (CPU / MPS is enough: a few minutes for 6 models).

For each model m, query set Q (training cats; near-class training images), timestep t on a dense grid, and
conditioning c in {true label, null}:  l_m(x, c, t) = eps-MSE with noise fixed per (example, t) -- common random
numbers, identical for every model and both conditionings.  Outputs per t:

  snr_t        d_t(original, O) / d_t(O_0, O_1)                  (A4-a; d_t = RMS over x of the loss difference)
  B_m(t)       mean_x [ l_m(x, null, t) - l_m(x, label, t) ]     label benefit (A4-b, A4-d)
  snr_avg      the same SNR with the audit's 5-timestep average (50..800), i.e. what E sees (A4-c)

Usage:  python -m src.diffusion.timestep_decomp [results_root] [out_json]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

TIMESTEPS = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 990]
AUDIT_T = [50, 200, 400, 600, 800]
NEAR = (2, 4, 5, 6, 7)
FC = 3


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def losses(model, diff, x, y, timesteps, device, noise_seed=777, batch=128):
    """(n, T) eps-MSE; noise depends only on (example position, t) so all models / conditionings share it."""
    import torch

    out = np.zeros((len(x), len(timesteps)), np.float64)
    model.eval()
    with torch.no_grad():
        for ti, t in enumerate(timesteps):
            g = torch.Generator(device="cpu").manual_seed(noise_seed + 100003 * ti)
            noise_all = torch.randn((len(x),) + tuple(x.shape[1:]), generator=g)
            for i in range(0, len(x), batch):
                xb = x[i:i + batch].to(device)
                yb = y[i:i + batch].to(device)
                nb = noise_all[i:i + batch].to(device)
                tb = torch.full((xb.shape[0],), int(t), device=device, dtype=torch.long)
                out[i:i + batch, ti] = diff.loss(model, xb, yb, t=tb, noise=nb, reduction="none").float().cpu().numpy()
    return out


def rms(a, b, axis=0):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def run(root: str = "results", n: int = 256, seed: int = 0) -> Dict:
    import torch

    from ..common.config import cell_name, load_yaml
    from .data import cifar_labels, load_cifar
    from .ddpm import Diffusion
    from .train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    ds = load_cifar(cfg["data"].get("root", "data"))
    labels = cifar_labels(ds)
    rng = np.random.default_rng(seed)
    sets = {"forget": rng.choice(np.where(labels == FC)[0], n, replace=False),
            "near": rng.choice(np.where(np.isin(labels, NEAR))[0], n, replace=False)}
    ck = os.path.join(root, "checkpoints")
    d3 = lambda m, s: os.path.join(root, "D3_ddpm_unlearn", cell_name({"method": m, "forget_frac": 0.1, "seed": s}), "unlearned.pt")
    paths = {"original": (os.path.join(ck, "ddpm_cifar10_base.pt"), True),
             "oracle0": (os.path.join(ck, "ddpm_cifar10_retain_f0.1_s0.pt"), True),
             "oracle1": (os.path.join(ck, "ddpm_cifar10_retain_f0.1_s1.pt"), True),
             "gradient_ascent": (d3("gradient_ascent", 0), False),
             "label_annealing": (d3("label_annealing", 0), False)}
    models = {}
    for k, (p, ema) in paths.items():
        if os.path.exists(p):
            models[k], _ = load_model(p, cfg, use_ema=ema, device=device)
    # embedding_reset built exactly as src/diffusion/unlearn.py does (seed 0), from the original
    import copy

    from ..common.seed import derive_seed

    er = copy.deepcopy(models["original"])
    g = torch.Generator(device="cpu").manual_seed(derive_seed(0, "embedding_reset"))
    with torch.no_grad():
        row = er.class_emb.weight[FC]
        er.class_emb.weight[FC] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
    models["embedding_reset"] = er
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    null = models["original"].null_class

    L: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for qn, idx in sets.items():
        x = torch.stack([ds[int(i)][0] for i in idx])
        y = torch.tensor(labels[idx], dtype=torch.long)
        yn = torch.full_like(y, null)
        L[qn] = {}
        for mn, m in models.items():
            L[qn][mn] = {"label": losses(m, diff, x, y, TIMESTEPS, device), "null": losses(m, diff, x, yn, TIMESTEPS, device)}
            print(f"[A4] {qn} {mn} done", flush=True)

    res: Dict = {"timesteps": TIMESTEPS, "n": n, "device": str(device), "sets": {}}
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    for qn, Lq in L.items():
        o, o0, o1 = Lq["original"], Lq["oracle0"], Lq["oracle1"]
        r = {}
        for c in ("label", "null"):
            floor = rms(o0[c], o1[c])
            sig = 0.5 * (rms(o[c], o0[c]) + rms(o[c], o1[c]))
            fa = rms(o0[c][:, ai].mean(1), o1[c][:, ai].mean(1))
            sa = 0.5 * (rms(o[c][:, ai].mean(1), o0[c][:, ai].mean(1)) + rms(o[c][:, ai].mean(1), o1[c][:, ai].mean(1)))
            # the audit's d: RMS over (x, t) of the per-(x, t) difference (fingerprint.rms_dist)
            fd = float(np.sqrt(np.mean((o0[c][:, ai] - o1[c][:, ai]) ** 2)))
            sd = 0.5 * (float(np.sqrt(np.mean((o[c][:, ai] - o0[c][:, ai]) ** 2))) + float(np.sqrt(np.mean((o[c][:, ai] - o1[c][:, ai]) ** 2))))
            r[c] = {"floor_t": floor.tolist(), "signal_t": sig.tolist(), "snr_t": (sig / floor).tolist(),
                    "snr_timeavg_loss": float(sa / fa), "snr_audit_d": float(sd / fd)}
        r["label_benefit"] = {mn: (Lm["null"] - Lm["label"]).mean(0).tolist() for mn, Lm in Lq.items()}
        r["label_benefit_se"] = {mn: ((Lm["null"] - Lm["label"]).std(0) / np.sqrt(n)).tolist() for mn, Lm in Lq.items()}
        r["mean_loss_label"] = {mn: Lm["label"].mean(0).tolist() for mn, Lm in Lq.items()}
        # distance of every model to the original / to the oracles, per t (label conditioning)
        r["d_to_original_t"] = {mn: rms(Lm["label"], o["label"]).tolist() for mn, Lm in Lq.items()}
        r["d_to_oracle_t"] = {mn: (0.5 * (rms(Lm["label"], o0["label"]) + rms(Lm["label"], o1["label"]))).tolist() for mn, Lm in Lq.items()}
        r["d_null_to_original_t"] = {mn: rms(Lm["null"], o["null"]).tolist() for mn, Lm in Lq.items()}
        res["sets"][qn] = r
    return res


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "figures", "a4_timestep_decomp.json")
    res = run(root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


def closeness_all(root: str = "results", n: int = 256, seed: int = 0) -> Dict:
    """Plain and timestep-standardised E (A4) for every class-regime D3 checkpoint + embedding_reset, label
    conditioning, on the audit's 5 timesteps and on the dense 12-timestep grid."""
    import copy

    import torch

    from ..audit.fingerprint import closeness_np, standardized_closeness_np
    from ..common.config import cell_name, load_yaml
    from ..common.seed import derive_seed
    from .data import cifar_labels, load_cifar
    from .ddpm import Diffusion
    from .train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    ds = load_cifar(cfg["data"].get("root", "data"))
    labels = cifar_labels(ds)
    rng = np.random.default_rng(seed)
    sets = {"forget": rng.choice(np.where(labels == FC)[0], n, replace=False),
            "near": rng.choice(np.where(np.isin(labels, NEAR))[0], n, replace=False)}
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    ck = os.path.join(root, "checkpoints")
    orig, _ = load_model(os.path.join(ck, "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=device)
    oracles = [load_model(os.path.join(ck, f"ddpm_cifar10_retain_f0.1_s{s}.pt"), cfg, use_ema=True, device=device)[0] for s in (0, 1)]
    data = {}
    for qn, idx in sets.items():
        x = torch.stack([ds[int(i)][0] for i in idx])
        y = torch.tensor(labels[idx], dtype=torch.long)
        data[qn] = (x, y)
    base_L = {qn: (losses(orig, diff, x, y, TIMESTEPS, device), [losses(o, diff, x, y, TIMESTEPS, device) for o in oracles]) for qn, (x, y) in data.items()}
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    out: Dict = {"n": n, "timesteps": TIMESTEPS, "audit_timesteps": AUDIT_T, "cells": {}}
    methods = ["do_nothing", "oracle", "retain_finetune", "label_annealing", "gradient_difference", "variational_ewc", "gradient_ascent", "embedding_reset"]
    for m in methods:
        for s in (0, 1):
            if m == "embedding_reset":
                mu = copy.deepcopy(orig)
                g = torch.Generator(device="cpu").manual_seed(derive_seed(s, "embedding_reset"))
                with torch.no_grad():
                    row = mu.class_emb.weight[FC]
                    mu.class_emb.weight[FC] = torch.randn(row.shape, generator=g).to(row.device, row.dtype)
            else:
                p = os.path.join(root, "D3_ddpm_unlearn", cell_name({"method": m, "forget_frac": 0.1, "seed": s}), "unlearned.pt")
                if not os.path.exists(p):
                    continue
                mu, _ = load_model(p, cfg, use_ema=False, device=device)
            cell = {}
            for qn, (x, y) in data.items():
                lu = losses(mu, diff, x, y, TIMESTEPS, device)
                l0, lo = base_L[qn]
                for tag, cols in (("audit5", ai), ("dense12", list(range(len(TIMESTEPS))))):
                    pl = closeness_np(lu[:, cols], l0[:, cols], [o[:, cols] for o in lo], n_boot=1000, seed=seed)
                    st = standardized_closeness_np(lu[:, cols], l0[:, cols], [o[:, cols] for o in lo], n_boot=1000, seed=seed)
                    cell[f"{qn}/{tag}"] = {"plain": pl, "std": st}
            out["cells"][f"{m}/s{s}"] = cell
            print(f"[A4-E] {m} s{s}: " + "  ".join(f"{k} E_std={v['std']['E']:.3f} [{v['std']['E_lo']:.3f},{v['std']['E_hi']:.3f}] snr={v['std']['snr']:.2f}" for k, v in cell.items() if k.endswith("audit5")), flush=True)
    return out


def reset_draws(root: str = "results", seeds=range(100, 120), n_loss: int = 256, n_samples: int = 200, seed: int = 0) -> Dict:
    """A4-e: distribution over independent embedding_reset draws of E_std (cats, audit timesteps) and gated cat rate."""
    import copy

    import torch

    from ..audit.fingerprint import standardized_closeness_np
    from ..common.config import load_yaml
    from .classifier import load_classifier
    from .data import cifar_labels, load_cifar
    from .ddpm import Diffusion
    from .sample import class_rates, sample_class
    from .train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_unlearn_sweep.yaml")
    s = cfg["sampling"]
    ds = load_cifar(cfg["data"].get("root", "data"))
    labels = cifar_labels(ds)
    idx = np.random.default_rng(seed).choice(np.where(labels == FC)[0], n_loss, replace=False)
    x = torch.stack([ds[int(i)][0] for i in idx])
    y = torch.tensor(labels[idx], dtype=torch.long)
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    clf = load_classifier(device)
    ck = os.path.join(root, "checkpoints")
    orig, _ = load_model(os.path.join(ck, "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=device)
    oracles = [load_model(os.path.join(ck, f"ddpm_cifar10_retain_f0.1_s{k}.pt"), cfg, use_ema=True, device=device)[0] for k in (0, 1)]
    T = AUDIT_T
    l0 = losses(orig, diff, x, y, T, device)
    lo = [losses(o, diff, x, y, T, device) for o in oracles]
    ref = {}
    for name, m in [("original", orig), ("oracle0", oracles[0]), ("oracle1", oracles[1])]:
        ref[name] = class_rates(sample_class(m, diff, FC, n_samples, int(s["ddim_steps"]), float(s["guidance"]), 4242), clf, FC)
    rows = []
    for sd in seeds:
        m = copy.deepcopy(orig)
        g = torch.Generator(device="cpu").manual_seed(int(sd))
        with torch.no_grad():
            r = m.class_emb.weight[FC]
            m.class_emb.weight[FC] = torch.randn(r.shape, generator=g).to(r.device, r.dtype)
        e = standardized_closeness_np(losses(m, diff, x, y, T, device), l0, lo, n_boot=200, seed=seed)
        cr = class_rates(sample_class(m, diff, FC, n_samples, int(s["ddim_steps"]), float(s["guidance"]), 4242), clf, FC)
        w = m.class_emb.weight.detach().float().cpu()
        cos = torch.nn.functional.cosine_similarity(w[FC][None], torch.cat([w[:FC], w[FC + 1:]]), dim=1)
        rows.append({"draw": int(sd), "E_std": e["E"], "E_lo": e["E_lo"], "E_hi": e["E_hi"], "cat_valid": cr["valid"], "cat_raw": cr["raw"],
                     "degenerate": cr["degenerate"], "max_cos_to_trained_rows": float(cos[:-1].abs().max()), "cos_to_null": float(cos[-1])})
        print(f"[A4-e] draw {sd}: E_std {e['E']:.2f} [{e['E_lo']:.2f},{e['E_hi']:.2f}]  cat valid {cr['valid']:.3f} raw {cr['raw']:.3f}  cos(null) {float(cos[-1]):+.3f}", flush=True)
    return {"reference_rates": ref, "draws": rows, "n_samples": n_samples, "n_loss": n_loss}


def controls(root: str = "results", n: int = 256, seed: int = 0, forget_class: int = FC, oracle_seeds=(0, 1)) -> Dict:
    """Per-timestep signal-to-floor ratio (label conditioning) of original vs oracles on four query sets:
    train images of the forgotten class (members), TEST images of that class (non-members: is the trace about the
    class or about memorised examples?), near and far retain classes (negative controls).  With k >= 3 oracles the
    floor is the mean pairwise distance and the signal the mean distance to every oracle."""
    import itertools

    import torch

    from ..common.config import load_yaml
    from .data import cifar_labels, load_cifar
    from .ddpm import Diffusion
    from .train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    tr, te = load_cifar(cfg["data"].get("root", "data")), load_cifar(cfg["data"].get("root", "data"), train=False)
    ltr, lte = cifar_labels(tr), cifar_labels(te)
    rng = np.random.default_rng(seed)
    others = [c for c in range(10) if c != forget_class]
    near = [c for c in NEAR if c != forget_class] if forget_class in NEAR or forget_class == FC else [c for c in (0, 1, 8, 9) if c != forget_class]
    far = [c for c in others if c not in near]
    sets = {"forget_train": (tr, rng.choice(np.where(ltr == forget_class)[0], n, replace=False)),
            "forget_test": (te, rng.choice(np.where(lte == forget_class)[0], n, replace=False)),
            "near": (tr, rng.choice(np.where(np.isin(ltr, near))[0], n, replace=False)),
            "far": (tr, rng.choice(np.where(np.isin(ltr, far))[0], n, replace=False))}
    ck = os.path.join(root, "checkpoints")
    cls = "" if forget_class == FC else f"_c{forget_class}"
    orig, _ = load_model(os.path.join(ck, "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=device)
    oracles = [load_model(os.path.join(ck, f"ddpm_cifar10_retain{cls}_f0.1_s{s}.pt"), cfg, use_ema=True, device=device)[0] for s in oracle_seeds]
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    out = {"timesteps": TIMESTEPS, "forget_class": forget_class, "oracle_seeds": list(oracle_seeds), "near": near, "far": far, "sets": {}}
    for qn, (ds, idx) in sets.items():
        x = torch.stack([ds[int(i)][0] for i in idx])
        y = torch.tensor((ltr if ds is tr else lte)[idx], dtype=torch.long)
        lo_ = losses(orig, diff, x, y, TIMESTEPS, device)
        lO = [losses(o, diff, x, y, TIMESTEPS, device) for o in oracles]
        pairs = list(itertools.combinations(range(len(lO)), 2))
        floor = np.mean([rms(lO[i], lO[j]) for i, j in pairs], axis=0)
        sig = np.mean([rms(lo_, o) for o in lO], axis=0)
        sc = np.where(floor > 1e-12, floor, 1.0)[None, :]
        fl_std = np.mean([np.sqrt(np.mean(((lO[i] - lO[j]) / sc)[:, ai] ** 2)) for i, j in pairs])
        sg_std = np.mean([np.sqrt(np.mean(((lo_ - o) / sc)[:, ai] ** 2)) for o in lO])
        fl_pl = np.mean([np.sqrt(np.mean((lO[i] - lO[j])[:, ai] ** 2)) for i, j in pairs])
        sg_pl = np.mean([np.sqrt(np.mean((lo_ - o)[:, ai] ** 2)) for o in lO])
        out["sets"][qn] = {"snr_t": (sig / floor).tolist(), "snr_plain_audit": float(sg_pl / fl_pl), "snr_std_audit": float(sg_std / fl_std)}
        print(f"[ctrl c{forget_class}] {qn:13} snr_t " + " ".join(f"{v:.2f}" for v in sig / floor) + f" | plain {sg_pl / fl_pl:.2f} std {sg_std / fl_std:.2f}", flush=True)
    return out


def subset_controls(root: str = "results", frac: float = 0.05, n: int = 256, seed: int = 0) -> Dict:
    """A6: per-timestep original-vs-oracle ratio in the subset regime on the removed examples (members of the
    original only), retained examples of the same class (members of both), and test images of the class (members
    of neither)."""
    import json as _json

    import torch

    from ..common.config import load_yaml
    from .data import cifar_labels, load_cifar
    from .ddpm import Diffusion
    from .train import load_model

    device = _device()
    cfg = load_yaml("configs/diffusion_base.yaml")
    tr, te = load_cifar(cfg["data"].get("root", "data")), load_cifar(cfg["data"].get("root", "data"), train=False)
    ltr, lte = cifar_labels(tr), cifar_labels(te)
    sp = _json.load(open(os.path.join(root, "splits", f"cifar10_forget{FC}_frac{frac:g}_split0.json")))
    removed = np.array(sp["forget"])
    kept = np.setdiff1d(np.where(ltr == FC)[0], removed)
    rng = np.random.default_rng(seed)
    sets = {"removed": (tr, rng.choice(removed, n, replace=False)), "retained_same_class": (tr, rng.choice(kept, n, replace=False)),
            "test_class": (te, rng.choice(np.where(lte == FC)[0], n, replace=False))}
    ck = os.path.join(root, "checkpoints")
    orig, _ = load_model(os.path.join(ck, "ddpm_cifar10_base.pt"), cfg, use_ema=True, device=device)
    oracles = [load_model(os.path.join(ck, f"ddpm_cifar10_retain_f{frac:g}_s{s}.pt"), cfg, use_ema=True, device=device)[0] for s in (0, 1)]
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    ai = [TIMESTEPS.index(t) for t in AUDIT_T]
    out = {"timesteps": TIMESTEPS, "frac": frac, "n_removed": int(len(removed)), "sets": {}}
    for qn, (ds, idx) in sets.items():
        x = torch.stack([ds[int(i)][0] for i in idx])
        y = torch.tensor((ltr if ds is tr else lte)[idx], dtype=torch.long)
        lo_ = losses(orig, diff, x, y, TIMESTEPS, device)
        o0, o1 = (losses(o, diff, x, y, TIMESTEPS, device) for o in oracles)
        floor, sig = rms(o0, o1), 0.5 * (rms(lo_, o0) + rms(lo_, o1))
        sc = np.where(floor > 1e-12, floor, 1.0)[None, :]
        pl = 0.5 * (np.sqrt(np.mean((lo_ - o0)[:, ai] ** 2)) + np.sqrt(np.mean((lo_ - o1)[:, ai] ** 2))) / np.sqrt(np.mean((o0 - o1)[:, ai] ** 2))
        st = 0.5 * (np.sqrt(np.mean(((lo_ - o0) / sc)[:, ai] ** 2)) + np.sqrt(np.mean(((lo_ - o1) / sc)[:, ai] ** 2))) / np.sqrt(np.mean(((o0 - o1) / sc)[:, ai] ** 2))
        # mean signed loss change (oracle - original): positive = the oracle finds these images harder
        dl = (0.5 * (o0 + o1) - lo_).mean(0)
        out["sets"][qn] = {"snr_t": (sig / floor).tolist(), "snr_plain_audit": float(pl), "snr_std_audit": float(st), "mean_loss_change_t": dl.tolist()}
        print(f"[A6] {qn:20} snr_t " + " ".join(f"{v:.2f}" for v in sig / floor) + f" | plain {pl:.2f} std {st:.2f}", flush=True)
    return out
