"""Unlearning methods for the class-conditional DDPM, all under matched gradient steps.

gradient_ascent     maximise the eps-MSE on forget examples; the per-step ascent loss is
                    clipped at ``ascent_bound`` so the model cannot diverge to NaN
gradient_difference ascent on forget + descent on retain (weight ``retain_weight``)
label_annealing     our reading of Panda et al. (UAI 2025): forget images are trained
                    *normally* but their class label is annealed linearly from the forget
                    class toward the null (unconditional) class over the steps, so the
                    conditional branch stops routing to them; plus retain descent
variational_ewc     our reading of the TMLR 2026 variational method: ascent on forget +
                    diagonal-Fisher-weighted quadratic to the original parameters
                    (Fisher estimated on retain), i.e. a Gaussian posterior prior
oracle              load the retain-only retrain (D2); no steps
do_nothing          the original checkpoint; the expected-to-fail control

Both "our reading" methods are interpretations of the papers, flagged as such
in PREREGISTRATION.md; replace them with the authors' code when available.
``retain_available: false`` removes every retain term (data-constraints setting).
Every method records ||delta theta||_F per parameter tensor and in total.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from ..common.config import run_dir
from ..common.kaggle_env import torch_device
from ..common.seed import derive_seed, seed_everything
from .data import cifar_labels, infinite, load_cifar, make_loader
from .ddpm import Diffusion
from .train import ckpt_path, get_split, load_model, save_ckpt

METHODS = ("gradient_ascent", "gradient_difference", "label_annealing", "variational_ewc", "oracle", "do_nothing")


def delta_norms(model, original_sd: Dict[str, torch.Tensor]) -> Dict[str, float]:
    per = {}
    total = 0.0
    sd = model.state_dict()
    for k, v in sd.items():
        if not v.dtype.is_floating_point:
            continue
        d = (v.detach().float().cpu() - original_sd[k].float().cpu())
        n2 = float((d**2).sum())
        per[k] = float(np.sqrt(n2))
        total += n2
    return {"delta_norm_total": float(np.sqrt(total)), "delta_norm_per_tensor": per}


@torch.no_grad()
def diagonal_fisher(model, diff: Diffusion, loader, n_batches: int, device) -> Dict[str, torch.Tensor]:
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    it = iter(loader)
    for _ in range(n_batches):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            diff.loss(model, x, y).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= n_batches
    model.zero_grad(set_to_none=True)
    return fisher


def unlearn_cell(cfg: Dict) -> Dict:
    u = cfg["unlearn"]
    method = u["method"]
    assert method in METHODS, method
    seed = int(u["seed"])
    seed_everything(derive_seed(seed, f"unlearn_{method}"))
    device = torch_device()
    out_dir = run_dir(cfg)
    forget_frac = float(u["forget_frac"])
    retain_available = bool(u.get("retain_available", True))

    base_path = cfg.get("base_checkpoint") or ckpt_path(cfg, "base", 0)
    model, ck = load_model(base_path, cfg, use_ema=True, device=device)
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    out_path = os.path.join(out_dir, "unlearned.pt")

    if method == "do_nothing":
        save_ckpt(out_path, model, None, cfg, 0, None)
        return {"checkpoint": out_path, "steps": 0, **delta_norms(model, original_sd), "unlearn_gpu_hours": 0.0}
    if method == "oracle":
        oracle_path = cfg.get("oracle_checkpoint") or ckpt_path(cfg, "retain", seed)
        omodel, _ = load_model(oracle_path, cfg, use_ema=True, device=device)
        save_ckpt(out_path, omodel, None, cfg, 0, None)
        return {"checkpoint": out_path, "steps": 0, "oracle_source": oracle_path, **delta_norms(omodel, original_sd), "unlearn_gpu_hours": 0.0}

    ds = load_cifar(cfg["data"].get("root", "data"))
    split = get_split(cfg, forget_frac, seed, cifar_labels(ds))
    bs = int(u.get("batch_size", 64))
    f_loader = make_loader(ds, split["forget"], min(bs, len(split["forget"])), True, derive_seed(seed, "forget"), 2)
    r_loader = make_loader(ds, split["retain"], bs, True, derive_seed(seed, "retain"), 2)
    f_it, r_it = infinite(f_loader), infinite(r_loader)
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    steps = int(u["steps"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(u["learning_rate"]), weight_decay=0.0)
    bound = float(u.get("ascent_bound", 1.0))
    rw = float(u.get("retain_weight", 1.0))
    fisher = None
    if method == "variational_ewc":
        if not retain_available:
            fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}  # isotropic prior when retain data are unavailable
        else:
            fisher = diagonal_fisher(model, diff, r_loader, int(u.get("fisher_batches", 20)), device)
        fmax = max(float(f.max()) for f in fisher.values()) or 1.0
        for n in fisher:
            fisher[n] = fisher[n] / fmax  # normalise so ewc_weight is scale-free
    null = model.null_class
    fc = int(cfg["data"]["forget_class"])
    hist = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        xf, yf = next(f_it)
        xf, yf = xf.to(device), yf.to(device)
        loss_terms = {}
        if method == "gradient_ascent" or method == "gradient_difference" or method == "variational_ewc":
            lf = diff.loss(model, xf, yf)
            # bounded ascent: once the forget loss exceeds `bound` the ascent term is switched off
            asc = -torch.clamp(lf, max=bound)
            loss = asc
            loss_terms["forget_mse"] = float(lf.item())
        elif method == "label_annealing":
            frac = step / steps
            # anneal the label: with probability frac use the null class, else the true forget class
            use_null = torch.rand(yf.shape[0], device=device) < frac
            y_ann = torch.where(use_null, torch.full_like(yf, null), yf)
            lf = diff.loss(model, xf, y_ann)
            loss = lf
            loss_terms["forget_annealed_mse"] = float(lf.item())
            loss_terms["anneal_frac"] = frac
        if retain_available and method in ("gradient_difference", "label_annealing"):
            xr, yr = next(r_it)
            xr, yr = xr.to(device), yr.to(device)
            yr = diff.train_step_labels(model, yr)
            lr_ = diff.loss(model, xr, yr)
            loss = loss + rw * lr_
            loss_terms["retain_mse"] = float(lr_.item())
        if method == "variational_ewc":
            ewc = 0.0
            for n, p in model.named_parameters():
                ewc = ewc + (fisher[n] * (p - original_sd[n]) ** 2).sum()
            loss = loss + float(u.get("ewc_weight", 1.0)) * ewc
            loss_terms["ewc"] = float(ewc.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(u.get("grad_clip", 1.0)))
        opt.step()
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss.item()), **loss_terms}
            hist.append(rec)
            if step % 200 == 0:
                print(json.dumps(rec))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss during unlearning; lower learning_rate or ascent_bound")
    save_ckpt(out_path, model, None, cfg, steps, split)
    with open(os.path.join(out_dir, "unlearn_history.json"), "w") as f:
        json.dump(hist, f)
    norms = delta_norms(model, original_sd)
    with open(os.path.join(out_dir, "delta_norms.json"), "w") as f:
        json.dump(norms, f)
    return {"checkpoint": out_path, "steps": steps, "delta_norm_total": norms["delta_norm_total"], "unlearn_gpu_hours": (time.time() - t0) / 3600}
