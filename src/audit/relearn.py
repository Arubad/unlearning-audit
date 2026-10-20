"""Few-shot relearning: fine-tune the unlearned checkpoint on a small part of the forget set.

Curve: forget-class generation rate (diffusion) or forget ROUGE-L (LLM) every
``eval_every`` steps; ``steps_to_fraction`` returns the first step at which the
curve reaches ``frac`` of the pre-unlearning value (inf if never).  The oracle,
which never saw the forget data, is the slow reference; "do nothing" must give 0.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Sequence

import numpy as np


def steps_to_fraction(steps: Sequence[int], values: Sequence[float], target_value: float, frac: float = 0.9) -> float:
    """First step whose value >= frac * target_value; 0 if already there at step 0; inf if never."""
    thr = frac * target_value
    for s, v in zip(steps, values):
        if v >= thr:
            return float(s)
    return math.inf


def relearn_diffusion(unlearned_model, cfg: Dict, split: Dict, pre_unlearn_rate: float, seed: int, metric_fn) -> Dict:
    """``metric_fn(model) -> float`` is the forget-class conditional generation rate."""
    import torch

    from ..common.kaggle_env import torch_device
    from ..common.seed import derive_seed
    from ..diffusion.data import infinite, load_cifar, make_loader
    from ..diffusion.ddpm import Diffusion

    r = cfg["audit"]["relearn"]
    device = torch_device()
    model = copy.deepcopy(unlearned_model).to(device)
    rng = np.random.default_rng(derive_seed(seed, "relearn"))
    n_sub = max(8, int(float(r["forget_subset_fraction"]) * len(split["forget"])))
    sub = rng.choice(split["forget"], size=min(n_sub, len(split["forget"])), replace=False).tolist()
    ds = load_cifar(cfg["data"].get("root", "data"))
    it = infinite(make_loader(ds, sub, min(int(r.get("batch_size", 32)), len(sub)), True, seed, 2))
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(r["learning_rate"]))
    steps_list: List[int] = [0]
    values: List[float] = [float(metric_fn(model))]
    max_steps, every = int(r["max_steps"]), int(r.get("eval_every", 50))
    model.train()
    for step in range(1, max_steps + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        loss = diff.loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % every == 0:
            model.eval()
            v = float(metric_fn(model))
            model.train()
            steps_list.append(step)
            values.append(v)
            if v >= float(r.get("recovery_target", 0.9)) * pre_unlearn_rate:
                break
    s90 = steps_to_fraction(steps_list, values, pre_unlearn_rate, float(r.get("recovery_target", 0.9)))
    return {"relearn_steps_to_90": s90, "relearn_curve_steps": steps_list, "relearn_curve_values": values, "relearn_n_examples": len(sub)}
