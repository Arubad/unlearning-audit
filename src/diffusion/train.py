"""D1 (base) and D2 (retain-only oracle) training with one recipe.

``train_ddpm(cfg, retain_only=False, seed=...)`` trains for ``train.steps``
optimizer steps on CIFAR-10 (all classes) or on the retain split (oracle), with
fp16 autocast on GPU, EMA, and periodic checkpoints.  Checkpoint files hold
``{"model", "ema", "config", "step", "split"}``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from ..common.kaggle_env import results_root, torch_device
from ..common.seed import derive_seed, seed_everything
from .data import cifar_labels, cifar_split_indices, infinite, load_cifar, make_loader, save_split
from .ddpm import EMA, Diffusion
from .unet import build_unet, count_params


def ckpt_path(cfg: Dict, kind: str, seed: int) -> str:
    d = os.path.join(results_root(), "checkpoints")
    os.makedirs(d, exist_ok=True)
    tag = f"{kind}_s{seed}" if kind != "base" else "base"
    return os.path.join(d, f"ddpm_cifar10_{tag}.pt")


def split_path(cfg: Dict, forget_frac: float, seed: int) -> str:
    return os.path.join(results_root(), "splits", f"cifar10_forget{cfg['data']['forget_class']}_frac{forget_frac}_s{seed}.json")


def get_split(cfg: Dict, forget_frac: float, seed: int, labels: Optional[np.ndarray] = None) -> Dict:
    p = split_path(cfg, forget_frac, seed)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    if labels is None:
        labels = cifar_labels(load_cifar(cfg["data"].get("root", "data")))
    s = cifar_split_indices(labels, int(cfg["data"]["forget_class"]), forget_frac, derive_seed(seed, "split"))
    save_split(s, p)
    return s


def save_ckpt(path: str, model, ema: Optional[EMA], cfg: Dict, step: int, split: Optional[Dict]) -> None:
    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict() if ema else None, "config": cfg, "step": step, "split": split}, path)


def load_model(path: str, cfg: Optional[Dict] = None, use_ema: bool = True, device=None):
    ck = torch.load(path, map_location="cpu")
    cfg = cfg or ck["config"]
    model = build_unet(cfg)
    sd = ck["ema"] if (use_ema and ck.get("ema") is not None) else ck["model"]
    model.load_state_dict(sd)
    return model.to(device or torch_device()), ck


def train_ddpm(cfg: Dict, retain_only: bool = False, seed: int = 0, steps: Optional[int] = None, out_path: Optional[str] = None, init_from: Optional[str] = None) -> Dict:
    seed_everything(derive_seed(seed, "train"))
    device = torch_device()
    tr = cfg["train"]
    steps = int(steps or tr["steps"])
    ds = load_cifar(cfg["data"].get("root", "data"))
    labels = cifar_labels(ds)
    split = None
    indices = None
    if retain_only:
        split = get_split(cfg, float(cfg["data"]["forget_frac"]), seed, labels)
        indices = split["retain"]
    loader = make_loader(ds, indices, int(tr["batch_size"]), True, derive_seed(seed, "loader"), int(cfg["data"].get("num_workers", 2)))
    it = infinite(loader)

    model = build_unet(cfg).to(device)
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location="cpu")["model"])
    print(f"UNet parameters: {count_params(model)/1e6:.2f}M")
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    ema = EMA(model, float(cfg["model"].get("ema_decay", 0.9999)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["learning_rate"]), weight_decay=float(tr.get("weight_decay", 0.0)))
    use_amp = device.type == "cuda" and bool(tr.get("amp", True))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    out_path = out_path or ckpt_path(cfg, "retain" if retain_only else "base", seed)
    hist = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y = diff.train_step_labels(model, y)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = diff.loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        if step % 100 == 0:
            rec = {"step": step, "loss": float(loss.item()), "elapsed_s": time.time() - t0}
            hist.append(rec)
            if step % 1000 == 0:
                print(json.dumps(rec))
        if step % int(tr.get("ckpt_every", 5000)) == 0 or step == steps:
            save_ckpt(out_path, model, ema, cfg, step, split)
    with open(out_path.replace(".pt", "_history.json"), "w") as f:
        json.dump(hist, f)
    return {"checkpoint": out_path, "steps": steps, "params_M": count_params(model) / 1e6, "final_loss": hist[-1]["loss"] if hist else float("nan"), "train_gpu_hours": (time.time() - t0) / 3600}
