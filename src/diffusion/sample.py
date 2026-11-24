"""Sampling-based forgetting metrics for a DDPM checkpoint.

forget_class_rate(ckpt)   fraction of samples *conditioned on the forget label* that the
                          classifier still recognises as the forget class
uncond_forget_rate(ckpt)  fraction of *unconditional* samples classified as the forget
                          class (its drift from the base model is the "unconditional
                          density unchanged?" check in the reasoning section)
retain_fid(ckpt)          FID of samples on retain classes against CIFAR-10 test (optional;
                          torchmetrics or torch-fidelity; reports 'unavailable' otherwise)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from ..common.kaggle_env import torch_device
from .classifier import load_classifier
from .ddpm import Diffusion


@torch.no_grad()
def sample_class(model, diff: Diffusion, y: Optional[int], n: int, steps: int, guidance: float, seed: int, batch: int = 128) -> torch.Tensor:
    g = torch.Generator(device=diff.device).manual_seed(seed)
    outs = []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        yy = None if y is None else torch.full((b,), int(y), device=diff.device, dtype=torch.long)
        outs.append(diff.ddim_sample(model, b, yy, steps=steps, guidance=guidance, generator=g))
    return torch.cat(outs)


@torch.no_grad()
def classify(images: torch.Tensor, clf, batch: int = 512) -> np.ndarray:
    preds = []
    for s in range(0, len(images), batch):
        preds.append(clf(images[s : s + batch]).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# Amendment A5 (2026-11-21): validity gate for classifier-based rates.  A degenerate sample (noise, saturated colour)
# is still assigned a class by the CIFAR classifier: in run v4, gradient-difference cat-conditioned samples were pure
# noise for seed 0 (100 % degenerate) yet 65 % were labelled "cat" (seed 1 still made real cats), and gradient-ascent
# noise was labelled "frog".  hf_energy = mean squared
# neighbour difference in [-1, 1] pixel units; HF_DEGENERATE is its 99.9th percentile on 5,000 real CIFAR-10 train
# images (seed 0; median 0.034, max 0.158).  Degenerate samples measured so far sit at 0.36-0.46.
HF_DEGENERATE = 0.1267


def hf_energy(images: torch.Tensor) -> np.ndarray:
    x = images.float()
    dv = ((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2).mean((1, 2, 3))
    dh = ((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2).mean((1, 2, 3))
    return ((dv + dh) / 2).cpu().numpy()


def degenerate_mask(images: torch.Tensor, threshold: float = HF_DEGENERATE) -> np.ndarray:
    return hf_energy(images) > threshold


def class_rates(images: torch.Tensor, clf, target: int) -> Dict[str, float]:
    """raw: classified as target; valid: classified as target AND not degenerate; degenerate: flagged fraction."""
    pred, deg = classify(images, clf), degenerate_mask(images)
    return {"raw": float((pred == target).mean()), "valid": float(((pred == target) & ~deg).mean()), "degenerate": float(deg.mean())}


def forgetting_metrics(model, cfg: Dict, seed: int, n: int = 500) -> Dict[str, float]:
    device = torch_device()
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    clf = load_classifier(device)
    fc = int(cfg["data"]["forget_class"])
    s = cfg["sampling"]
    cond = sample_class(model, diff, fc, n, int(s["ddim_steps"]), float(s["guidance"]), seed)
    unc = sample_class(model, diff, None, n, int(s["ddim_steps"]), 1.0, seed + 1)
    rc, ru = class_rates(cond, clf, fc), class_rates(unc, clf, fc)
    return {"forget_class_rate_conditional": rc["raw"], "forget_class_rate_unconditional": ru["raw"], "n_samples": int(n),
            "forget_class_rate_conditional_valid": rc["valid"], "degenerate_rate_conditional": rc["degenerate"],
            "forget_class_rate_unconditional_valid": ru["valid"], "degenerate_rate_unconditional": ru["degenerate"]}


def retain_fid(model, cfg: Dict, seed: int, n: int = 2000) -> Dict[str, float]:
    """FID of retain-class conditional samples vs CIFAR-10 test images of those classes.  Optional."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return {"fid_retain": float("nan"), "fid_note": "torchmetrics unavailable"}
    from .data import load_cifar

    device = torch_device()
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    fc = int(cfg["data"]["forget_class"])
    classes = [c for c in range(10) if c != fc]
    per = max(1, n // len(classes))
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    s = cfg["sampling"]
    for c in classes:
        imgs = sample_class(model, diff, c, per, int(s["ddim_steps"]), float(s["guidance"]), seed + c)
        fid.update(((imgs + 1) / 2).clamp(0, 1), real=False)
    test = load_cifar(cfg["data"].get("root", "data"), train=False)
    labels = np.array(test.targets)
    idx = np.where(labels != fc)[0][:n]
    xs = torch.stack([test[i][0] for i in idx]).to(device)
    fid.update(((xs + 1) / 2).clamp(0, 1), real=True)
    return {"fid_retain": float(fid.compute().item())}
