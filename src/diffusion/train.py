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
from ..common.deadline import DeadlineReached, past
from ..common.seed import derive_seed, seed_everything
from .data import cifar_labels, cifar_split_indices, infinite, load_cifar, make_loader, save_split
from .ddpm import EMA, Diffusion
from .unet import build_unet, count_params


def ckpt_path(cfg: Dict, kind: str, seed: int, forget_frac: Optional[float] = None) -> str:
    """base -> ddpm_cifar10_base.pt; retain (oracle) -> ddpm_cifar10_retain_f{frac}_s{seed}.pt.

    The oracle path carries the forget fraction (2026-11-18): before, oracles for different forget sets
    overwrote each other and the sweep could audit a 10% cell against a 5% oracle.
    """
    d = os.path.join(results_root(), "checkpoints")
    os.makedirs(d, exist_ok=True)
    if kind == "base":
        return os.path.join(d, "ddpm_cifar10_base.pt")
    ff = float(forget_frac if forget_frac is not None else cfg["data"]["forget_frac"])
    return os.path.join(d, f"ddpm_cifar10_{kind}{_cls_tag(cfg)}_f{ff:g}_s{seed}.pt")


def _cls_tag(cfg: Dict) -> str:
    """'' for the original forget class 3 (names unchanged), '_c{k}' otherwise (C1: oracles for other classes must
    not overwrite the cat oracles)."""
    fc = int(cfg.get("data", {}).get("forget_class", 3))
    return "" if fc == 3 else f"_c{fc}"


def oracle_paths(cfg: Dict, forget_frac: float) -> Dict[int, str]:
    """All oracle retrains on disk for this forget fraction, keyed by seed."""
    import glob
    import re

    d = os.path.join(results_root(), "checkpoints")
    out = {}
    for p in glob.glob(os.path.join(d, f"ddpm_cifar10_retain{_cls_tag(cfg)}_f{float(forget_frac):g}_s*.pt")):
        m = re.search(r"_s(\d+)\.pt$", p)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def base_path(cfg: Dict) -> str:
    p = cfg.get("base_checkpoint")
    return p if (p and os.path.exists(p)) else ckpt_path(cfg, "base", 0)


def split_seed(cfg: Dict) -> int:
    """The forget set is fixed per forget fraction (``data.split_seed``, default 0) and does NOT vary with the run
    seed (amendment 2026-11-18): run seeds vary training/unlearning randomness only, so oracle seeds are replicates
    of the same counterfactual and give the seed-to-seed noise floor."""
    return int(cfg.get("data", {}).get("split_seed", 0))


def split_path(cfg: Dict, forget_frac: float, seed: int) -> str:
    return os.path.join(results_root(), "splits", f"cifar10_forget{cfg['data']['forget_class']}_frac{forget_frac:g}_split{split_seed(cfg)}.json")


def get_split(cfg: Dict, forget_frac: float, seed: int = 0, labels: Optional[np.ndarray] = None) -> Dict:
    """``seed`` is accepted for backward compatibility and ignored; see ``split_seed``."""
    p = split_path(cfg, forget_frac, seed)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    if labels is None:
        labels = cifar_labels(load_cifar(cfg["data"].get("root", "data")))
    s = cifar_split_indices(labels, int(cfg["data"]["forget_class"]), forget_frac, derive_seed(split_seed(cfg), "split"))
    save_split(s, p)
    return s


def save_ckpt(path: str, model, ema: Optional[EMA], cfg: Dict, step: int, split: Optional[Dict], opt=None, scaler=None) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict() if ema else None, "config": cfg, "step": step, "split": split,
                "opt": opt.state_dict() if opt is not None else None, "scaler": scaler.state_dict() if scaler is not None else None,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, tmp)
    os.replace(tmp, path)  # atomic: a session reclaimed mid-save never leaves a truncated checkpoint


def load_model(path: str, cfg: Optional[Dict] = None, use_ema: bool = True, device=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = cfg or ck["config"]
    model = build_unet(cfg)
    sd = ck["ema"] if (use_ema and ck.get("ema") is not None) else ck["model"]
    model.load_state_dict(sd)
    return model.to(device or torch_device()), ck


def train_ddpm(cfg: Dict, retain_only: bool = False, seed: int = 0, steps: Optional[int] = None, out_path: Optional[str] = None, init_from: Optional[str] = None) -> Dict:
    # The oracle's init/data-order stream is distinct from the base model's (2026-11-18): before, oracle seed 0 shared
    # the base model's initialisation, so it was not an exchangeable replicate of oracle seed 1 for the noise floor.
    seed_everything(derive_seed(seed, "train_retain" if retain_only else "train"))
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
    start = 1
    if os.path.exists(out_path):  # resume (free Colab sessions are reclaimed without warning)
        ck = torch.load(out_path, map_location="cpu", weights_only=False)
        if int(ck.get("step", 0)) >= steps:
            print(f"already trained: {out_path} at step {ck['step']}")
            return {"checkpoint": out_path, "steps": steps, "params_M": count_params(model) / 1e6, "final_loss": float("nan"), "train_gpu_hours": 0.0, "resumed": True}
        model.load_state_dict(ck["model"])
        if ck.get("ema") is not None:
            ema.shadow.load_state_dict(ck["ema"])
        if ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start = int(ck["step"]) + 1
        hp = out_path.replace(".pt", "_history.json")
        if os.path.exists(hp):
            with open(hp) as f:
                hist = [h for h in json.load(f) if h["step"] < start]
        print(f"resuming {out_path} from step {start}")
    t0 = time.time()
    model.train()
    for step in range(start, steps + 1):
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
            save_ckpt(out_path, model, ema, cfg, step, split, opt, scaler)
            with open(out_path.replace(".pt", "_history.json"), "w") as f:
                json.dump(hist, f)
        elif step % 100 == 0 and past(600):  # 10 min margin for the save + session teardown
            save_ckpt(out_path, model, ema, cfg, step, split, opt, scaler)
            with open(out_path.replace(".pt", "_history.json"), "w") as f:
                json.dump(hist, f)
            raise DeadlineReached(f"{out_path} saved at step {step}")
    with open(out_path.replace(".pt", "_history.json"), "w") as f:
        json.dump(hist, f)
    return {"checkpoint": out_path, "steps": steps, "params_M": count_params(model) / 1e6, "final_loss": hist[-1]["loss"] if hist else float("nan"), "train_gpu_hours": (time.time() - t0) / 3600}
