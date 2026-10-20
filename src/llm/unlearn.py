"""LLM unlearning methods under matched gradient steps.

gradient_ascent     ascent on forget NLL (bounded)
gradient_difference ascent on forget + descent on retain
npo                 Negative Preference Optimization (Zhang et al. 2024):
                    L = (2/beta) * E_forget[ log sigmoid( beta * (logp_ref - logp_theta) ) ] + retain NLL
                    (the reference is the pre-unlearning model with adapters disabled, or a frozen copy
                    for the full-FT arm)
oracle              load the retain-only retrain
do_nothing          the base checkpoint (control)
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..common.config import run_dir
from ..common.kaggle_env import resolve_dtype, torch_device
from ..common.seed import derive_seed, seed_everything
from .data import build_examples, collate, forget_retain_configs, load_tofu
from .finetune import adapter_dir, load_base, make_optimizer

METHODS = ("gradient_ascent", "gradient_difference", "npo", "oracle", "do_nothing")


def _seq_logp(model, batch) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = (labels != -100).float()
    tgt = labels.clamp(min=0)
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float() - torch.logsumexp(logits.float(), dim=-1)
    return (lp * mask).sum(-1)


def _nll(model, batch) -> torch.Tensor:
    return model(**batch).loss


def unlearn_llm(cfg: Dict) -> Dict:
    u = cfg["unlearn"]
    method = u["method"]
    assert method in METHODS, method
    seed = int(u["seed"])
    full_ft = bool(u.get("full_ft", False))
    seed_everything(derive_seed(seed, f"llm_unlearn_{method}"))
    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    out_dir = run_dir(cfg)
    forget_frac = float(cfg["data"]["forget_frac"])

    base_ckpt = cfg["base_checkpoint"]
    if method == "do_nothing":
        return {"checkpoint": base_ckpt, "steps": 0}
    if method == "oracle":
        return {"checkpoint": cfg["oracle_checkpoint"], "steps": 0}

    tok, model = load_base(cfg, device, dtype, full_ft, adapter=None if full_ft else base_ckpt)
    if full_ft:
        model.load_state_dict(torch.load(os.path.join(base_ckpt, "model.pt"), map_location="cpu"))
        model.to(device)
        ref = copy.deepcopy(model).eval() if method == "npo" else None
    else:
        ref = None  # NPO reference = adapters disabled

    forget_cfg, retain_cfg = forget_retain_configs(forget_frac)
    fex = build_examples(tok, load_tofu(forget_cfg), int(cfg["model"].get("max_length", 512)))
    rex = build_examples(tok, load_tofu(retain_cfg), int(cfg["model"].get("max_length", 512)))
    rng = np.random.default_rng(derive_seed(seed, "batches"))
    bs = int(u.get("batch_size", 8))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, float(u["learning_rate"]), full_ft)
    steps = int(u["steps"])
    beta = float(u.get("npo_beta", 0.1))
    bound = float(u.get("ascent_bound", 5.0))
    rw = float(u.get("retain_weight", 1.0))
    retain_available = bool(u.get("retain_available", True))
    hist: List[Dict] = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        fb = collate([fex[i] for i in rng.choice(len(fex), size=min(bs, len(fex)), replace=False)], tok.pad_token_id, device)
        if method == "gradient_ascent":
            loss = -torch.clamp(_nll(model, fb), max=bound)
        elif method == "gradient_difference":
            loss = -torch.clamp(_nll(model, fb), max=bound)
        elif method == "npo":
            lp = _seq_logp(model, fb)
            with torch.no_grad():
                if full_ft:
                    lp_ref = _seq_logp(ref, fb)
                else:
                    with model.disable_adapter():
                        lp_ref = _seq_logp(model, fb)
            # NPO (Zhang et al. 2024): minimise -(2/beta) E_forget[ log sigmoid( -beta (logp_theta - logp_ref) ) ]
            loss = -(2.0 / beta) * F.logsigmoid(-beta * (lp - lp_ref)).mean()
        if retain_available and method in ("gradient_difference", "npo"):
            rb = collate([rex[i] for i in rng.choice(len(rex), size=min(bs, len(rex)), replace=False)], tok.pad_token_id, device)
            loss = loss + rw * _nll(model, rb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item())})
        if not math.isfinite(float(loss.item())):
            raise RuntimeError("non-finite loss; lower learning_rate / ascent_bound")
    if full_ft:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    else:
        model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "unlearn_history.json"), "w") as f:
        json.dump(hist, f)
    return {"checkpoint": out_dir, "steps": steps, "full_ft": full_ft, "unlearn_gpu_hours": (time.time() - t0) / 3600}
