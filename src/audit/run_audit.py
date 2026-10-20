"""Run the full audit for one diffusion cell: load the checkpoint pair, compute every axis, assemble the record.

The MIA axis is called through ``mia.mia_suite`` which is a stub in this code
drop (see src/audit/mia.py); the record carries NaN for it and H1 is marked
inconclusive until MIA is implemented.  Every other axis is complete.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from ..common.config import run_dir
from ..common.kaggle_env import torch_device
from ..diffusion.data import cifar_labels, load_cifar
from ..diffusion.ddpm import Diffusion
from ..diffusion.train import get_split, load_model
from .attribution import attribution_diffusion
from .mia import mia_suite
from .probe import probe_diffusion
from .relearn import relearn_diffusion
from .report import assemble_record, save_record
from .spectrum import spectrum_of_checkpoints


def audit_diffusion_cell(cfg: Dict, oracle_relearn_steps: Optional[float] = None) -> Dict:
    import torch

    from ..diffusion.sample import forgetting_metrics
    from ..diffusion.sample import sample_class

    u = cfg["unlearn"]
    seed = int(u["seed"])
    device = torch_device()
    out_dir = run_dir(cfg)
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    base_path = cfg.get("base_checkpoint")
    unlearned_path = os.path.join(out_dir, "unlearned.pt")
    base, _ = load_model(base_path, cfg, use_ema=True, device=device)
    unlearned, _ = load_model(unlearned_path, cfg, use_ema=False, device=device)
    oracle = None
    if cfg.get("oracle_checkpoint") and os.path.exists(cfg["oracle_checkpoint"]):
        oracle, _ = load_model(cfg["oracle_checkpoint"], cfg, use_ema=True, device=device)

    ds = load_cifar(cfg["data"].get("root", "data"))
    labels = cifar_labels(ds)
    split = get_split(cfg, float(u["forget_frac"]), seed, labels)
    fc = int(cfg["data"]["forget_class"])

    forgetting = forgetting_metrics(unlearned, cfg, seed, int(cfg["sampling"].get("n_eval", 300)))
    pre = forgetting_metrics(base, cfg, seed, int(cfg["sampling"].get("n_eval", 300)))["forget_class_rate_conditional"]
    mia = mia_suite(unlearned, base, [oracle] if oracle else [], (ds, split), cfg, device)
    attribution = attribution_diffusion(base, unlearned, oracle, diff, ds, split, cfg, device, seed)
    probe = probe_diffusion(unlearned, diff, ds, labels, fc, cfg, device, seed)
    spectrum = spectrum_of_checkpoints(unlearned.state_dict(), base.state_dict(), int(cfg["audit"]["spectrum"]["top_k"]))

    def metric_fn(m):
        from .relearn import steps_to_fraction  # noqa

        imgs = sample_class(m, diff, fc, int(cfg["sampling"].get("n_relearn_eval", 128)), int(cfg["sampling"]["ddim_steps"]), float(cfg["sampling"]["guidance"]), seed)
        from ..diffusion.sample import classify
        from ..diffusion.classifier import load_classifier

        return float((classify(imgs, load_classifier(device)) == fc).mean())

    relearn = relearn_diffusion(unlearned, cfg, split, pre, seed, metric_fn)
    record = assemble_record(cfg.get("_cell", {}) | {"track": "diffusion"}, {**forgetting, "pre_unlearn_forget_rate": pre}, mia, attribution, relearn, probe, spectrum, oracle_relearn_steps)
    save_record(record, out_dir)
    return record
