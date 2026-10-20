"""L4: audit one LLM unlearning cell (eval, MIA, attribution, probe, relearn, spectrum).

Mirrors ``audit_diffusion_cell``.  Loads three checkpoints of the same architecture:
  base      fine-tuned on full TOFU (minus the held-out slice)       = the "original"
  unlearned the cell's checkpoint in run_dir                          = the audited model
  oracle    retrain on retain only                                    = the retrain reference
and the fixed held-out retain slice (MIA non-members) recorded by finetune.py.

H3 is only meaningful for full-parameter updates; the record carries ``lora`` so the
hypothesis test drops LoRA cells from the rank-vs-relearning correlation.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ..audit.attribution import attribution_llm
from ..audit.mia import mia_suite_llm
from ..audit.probe import probe_accuracy_np
from ..audit.relearn import steps_to_fraction
from ..audit.report import assemble_record, save_record
from ..audit.spectrum import lora_delta_spectrum, spectrum_of_state_dicts
from ..common.config import run_dir
from ..common.kaggle_env import resolve_dtype, results_root, torch_device
from ..common.seed import derive_seed, seed_everything
from .data import build_examples, collate, forget_retain_configs, heldout_split, load_tofu
from .eval import evaluate_llm, generate_answers, mean_rouge_l
from .finetune import load_base


def _load(cfg, ckpt, full_ft, device, dtype, trainable=False):
    import torch

    if full_ft:
        tok, model = load_base(cfg, device, dtype, full_ft=True)
        model.load_state_dict(torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu"))
        return tok, model.to(device)
    tok, model = load_base(cfg, device, dtype, full_ft=False, adapter=ckpt)
    return tok, model.to(device)


def _forget_retain_heldout_utility(cfg, tok, seed):
    frac = float(cfg["data"]["forget_frac"])
    forget_cfg, retain_cfg = forget_retain_configs(frac)
    ml = int(cfg["model"].get("max_length", 512))
    retain_ds = load_tofu(retain_cfg)
    split = heldout_split(len(retain_ds), float(cfg.get("data", {}).get("heldout_frac", 0.05)), derive_seed(seed, "heldout"))
    forget_ex = build_examples(tok, load_tofu(forget_cfg), ml)
    retain_train = build_examples(tok, retain_ds.select(split["train"]), ml)
    heldout_ex = build_examples(tok, retain_ds.select(split["heldout"]), ml)
    try:
        utility_ex = build_examples(tok, load_tofu("world_facts"), ml)
    except Exception:
        utility_ex = []
    return forget_ex, retain_train, heldout_ex, utility_ex


def relearn_llm(cfg, unlearned_model, tok, forget_ex, pre_forget_rouge, device, seed) -> Dict:
    """Fine-tune the unlearned model on a small slice of the forget set; forget ROUGE-L every N steps."""
    import torch

    r = cfg["audit"]["relearn"]
    model = copy.deepcopy(unlearned_model).to(device)
    rng = np.random.default_rng(derive_seed(seed, "relearn_llm"))
    n_sub = max(4, int(float(r.get("forget_subset_fraction", 0.05)) * len(forget_ex)))
    sub = [forget_ex[i] for i in rng.choice(len(forget_ex), size=min(n_sub, len(forget_ex)), replace=False)]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(r["learning_rate"]))
    bs = int(r.get("batch_size", 4))
    every, max_steps = int(r.get("eval_every", 20)), int(r.get("max_steps", 200))
    n_eval = int(cfg.get("eval", {}).get("n_eval", 100))

    def forget_rouge(m):
        pred = generate_answers(m, tok, forget_ex[:n_eval], device, int(cfg.get("eval", {}).get("max_new_tokens", 128)))
        return mean_rouge_l(pred, [x["answer"] for x in forget_ex[:n_eval]])

    steps_list, values = [0], [forget_rouge(model)]
    model.train()
    for step in range(1, max_steps + 1):
        idx = rng.choice(len(sub), size=min(bs, len(sub)), replace=False)
        b = collate([sub[i] for i in idx], tok.pad_token_id, device)
        loss = model(**b).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % every == 0:
            model.eval()
            values.append(forget_rouge(model))
            steps_list.append(step)
            model.train()
            if values[-1] >= float(r.get("recovery_target", 0.9)) * pre_forget_rouge:
                break
    s90 = steps_to_fraction(steps_list, values, pre_forget_rouge, float(r.get("recovery_target", 0.9)))
    return {"relearn_steps_to_90": s90, "relearn_curve_steps": steps_list, "relearn_curve_values": values, "relearn_n_examples": len(sub)}


def audit_llm_cell(cfg: Dict) -> Dict:
    u = cfg["unlearn"]
    seed = int(u["seed"])
    full_ft = bool(u.get("full_ft", False))
    method = u["method"]
    seed_everything(derive_seed(seed, f"llm_audit_{method}"))
    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    out_dir = run_dir(cfg)

    base_ckpt = cfg["base_checkpoint"]
    oracle_ckpt = cfg.get("oracle_checkpoint")
    unlearned_ckpt = base_ckpt if method == "do_nothing" else (oracle_ckpt if method == "oracle" else out_dir)

    tok, unlearned = _load(cfg, unlearned_ckpt, full_ft, device, dtype)
    forget_ex, retain_ex, heldout_ex, utility_ex = _forget_retain_heldout_utility(cfg, tok, seed)

    forgetting = evaluate_llm(unlearned, tok, forget_ex, retain_ex, utility_ex, device, cfg)

    _, base = _load(cfg, base_ckpt, full_ft, device, dtype)
    pre = evaluate_llm(base, tok, forget_ex, retain_ex, utility_ex, device, cfg)
    forgetting["pre_unlearn_forget_rouge_l"] = pre["forget_rouge_l"]

    oracle = None
    if oracle_ckpt and (os.path.exists(os.path.join(oracle_ckpt, "model.pt")) or os.path.exists(os.path.join(oracle_ckpt, "adapter_config.json"))):
        _, oracle = _load(cfg, oracle_ckpt, full_ft, device, dtype)

    mia = mia_suite_llm(unlearned, tok, forget_ex, heldout_ex, retain_ex, [oracle] if oracle else [], device, cfg)
    attribution = attribution_llm(base, unlearned, oracle, tok, forget_ex, retain_ex, cfg, device, seed)

    # probe: forget (1) vs retain (0) features from the unlearned model
    from ..audit.mia import llm_features

    layer = int(cfg["audit"]["probe"].get("feature_layer", -1))
    n_pr = int(cfg["audit"]["probe"].get("n_examples", 300))
    ff = llm_features(unlearned, tok, forget_ex[:n_pr], device, layer)
    fr = llm_features(unlearned, tok, retain_ex[:n_pr], device, layer)
    X = np.vstack([ff, fr])
    y = np.array([1] * len(ff) + [0] * len(fr))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = perm[:cut], perm[cut:]
    probe = probe_accuracy_np(X[tr], y[tr], X[te], y[te], seed=seed)

    # relearn (skip for oracle / do_nothing, which define the reference points)
    relearn = {"relearn_steps_to_90": 0.0} if method == "do_nothing" else relearn_llm(cfg, unlearned, tok, forget_ex, pre["forget_rouge_l"], device, seed)
    oracle_relearn_steps = None
    if method not in ("oracle", "do_nothing") and oracle is not None:
        oracle_relearn_steps = relearn_llm(cfg, oracle, tok, forget_ex, pre["forget_rouge_l"], device, seed)["relearn_steps_to_90"]

    # spectrum
    if full_ft:
        import torch

        spectrum = spectrum_of_state_dicts(unlearned.state_dict(), base.state_dict(), int(cfg["audit"].get("spectrum", {}).get("top_k", 5)))
        spectrum["lora"] = False
    else:
        spectrum = lora_delta_spectrum(unlearned, base, int(cfg["audit"].get("spectrum", {}).get("top_k", 5)))

    cell = dict(cfg.get("_cell", {}))
    cell.update({"track": "llm", "full_ft": full_ft})
    record = assemble_record(cell, forgetting, mia, attribution, relearn, probe, spectrum, oracle_relearn_steps)
    record["lora"] = bool(spectrum.get("lora", not full_ft))
    save_record(record, out_dir)
    print(json.dumps({k: v for k, v in record.items() if isinstance(v, (int, float, str, bool))}, indent=2, default=float))
    return record
