"""L1/L2: fine-tune Qwen2.5-0.5B on TOFU (LoRA r16 or full fine-tune), and the retain-only oracle.

The full-FT arm exists for H3 (LoRA deltas have rank <= r by construction, so the
rank-vs-relearning test is only meaningful on full-parameter updates).  8-bit
AdamW is used when bitsandbytes is importable, else torch AdamW in fp32.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..common.kaggle_env import resolve_dtype, results_root, torch_device
from ..common.seed import derive_seed, seed_everything
from .data import build_examples, collate, forget_retain_configs, load_tofu


def adapter_dir(cfg: Dict, tag: str) -> str:
    d = os.path.join(results_root(), "llm_checkpoints", tag)
    os.makedirs(d, exist_ok=True)
    return d


def load_base(cfg: Dict, device, dtype, full_ft: bool, adapter: Optional[str] = None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    elif not full_ft:
        l = cfg["model"]["lora"]
        model = get_peft_model(model, LoraConfig(r=int(l["r"]), lora_alpha=int(l["alpha"]), lora_dropout=float(l["dropout"]), target_modules=list(l["target_modules"]), task_type="CAUSAL_LM"))
    return tok, model.to(device)


def make_optimizer(params, lr: float, full_ft: bool):
    if full_ft:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr)


def _train(model, tok, examples: List[Dict], tr: Dict, device, seed: int, full_ft: bool) -> List[Dict]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, float(tr["learning_rate"]), full_ft)
    bs, accum, epochs = int(tr["batch_size"]), int(tr.get("grad_accum", 1)), int(tr["epochs"])
    rng = np.random.default_rng(derive_seed(seed, "order"))
    hist, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(examples))
        for s in range(0, len(order), bs):
            batch = collate([examples[i] for i in order[s : s + bs]], tok.pad_token_id, device)
            out = model(**batch)
            (out.loss / accum).backward()
            if (s // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if step and step % 20 == 0:
                hist.append({"step": step, "loss": float(out.loss.item())})
    return hist


def finetune(cfg: Dict, retain_only: bool = False, full_ft: bool = False, seed: int = 0) -> Dict:
    seed_everything(derive_seed(seed, "ft"))
    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    tok, model = load_base(cfg, device, dtype, full_ft)
    forget_frac = float(cfg["data"]["forget_frac"])
    if retain_only:
        _, retain_cfg = forget_retain_configs(forget_frac)
        ds = load_tofu(retain_cfg)
    else:
        ds = load_tofu("full")
    examples = build_examples(tok, ds, int(cfg["model"].get("max_length", 512)))
    hist = _train(model, tok, examples, cfg["train"], device, seed, full_ft)
    tag = f"{'oracle' if retain_only else 'base'}_{'full' if full_ft else 'lora'}_f{forget_frac}_s{seed}"
    d = adapter_dir(cfg, tag)
    if full_ft:
        torch.save(model.state_dict(), os.path.join(d, "model.pt"))
    else:
        model.save_pretrained(d)
    with open(os.path.join(d, "history.json"), "w") as f:
        json.dump(hist, f)
    return {"checkpoint": d, "full_ft": full_ft, "retain_only": retain_only, "final_loss": hist[-1]["loss"] if hist else float("nan")}
