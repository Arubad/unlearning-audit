"""Tiny end-to-end check.  Diffusion: 200-step UNet on a 2k-image CIFAR subset, 2 unlearning
methods, audit on small subsets.  Exercises every code path; nothing it produces is a result.

Run on Kaggle (GPU).  The LLM smoke (32 TOFU examples, 10 steps) is gated behind
``LLM_SMOKE=1`` because it downloads Qwen2.5-0.5B.
"""
from __future__ import annotations

import json
import os
import tempfile

from .common.config import resolve, run_dir
from .common.explog import finish_entry, start_entry


def smoke_config(tmp: str) -> dict:
    return {
        "experiment_id": "SMOKE",
        "model": {"type": "ddpm_unet_small", "n_classes": 10, "base_channels": 32, "channel_mults": [1, 2], "res_blocks": 1, "attention_resolutions": [16], "image_size": 32, "timesteps": 1000, "ema_decay": 0.995, "dropout": 0.1, "cfg_dropout": 0.1},
        "data": {"root": os.path.join(tmp, "data"), "forget_class": 3, "forget_frac": 0.05, "num_workers": 2},
        "train": {"steps": 200, "batch_size": 64, "learning_rate": 2e-4, "grad_clip": 1.0, "amp": True, "ckpt_every": 200},
        "sampling": {"ddim_steps": 10, "guidance": 2.0, "n_eval": 32, "n_relearn_eval": 32},
        "unlearn": {"method": "gradient_ascent", "forget_frac": 0.05, "retain_available": True, "steps": 30, "learning_rate": 1e-5, "ascent_bound": 1.0, "retain_weight": 1.0, "ewc_weight": 1.0, "batch_size": 32, "seed": 0},
        "audit": {"attribution": {"projection_dim": 512, "projection_seed": 1234, "hessian_damping": 1e-2, "n_forget": 32, "train_subset": 128, "n_retain_queries": 32, "timesteps": [200, 500]}, "probe": {"n_per_class": 100, "timestep": 200, "C": 1.0}, "relearn": {"forget_subset_fraction": 0.1, "learning_rate": 1e-4, "max_steps": 40, "eval_every": 20, "recovery_target": 0.9, "batch_size": 16}, "spectrum": {"top_k": 5}},
        "logging": {"output_dir": os.path.join(tmp, "results")},
    }


def run_smoke() -> None:
    from .analysis.figures import analyze
    from .audit.run_audit import audit_diffusion_cell
    from .diffusion.classifier import train_classifier
    from .diffusion.train import train_ddpm
    from .diffusion.unlearn import unlearn_cell

    tmp = tempfile.mkdtemp(prefix="unlearn_smoke_")
    os.environ["RESULTS_ROOT"] = os.path.join(tmp, "results")
    cfg = smoke_config(tmp)
    base = train_ddpm(cfg, retain_only=False, seed=0)
    cfg["base_checkpoint"] = base["checkpoint"]
    train_ddpm(cfg, retain_only=True, seed=0)
    cfg["oracle_checkpoint"] = os.path.join(os.environ["RESULTS_ROOT"], "checkpoints", "ddpm_cifar10_retain_s0.pt")
    train_classifier(cfg, epochs=1)
    for method in ("gradient_ascent", "label_annealing", "do_nothing"):
        c = resolve(cfg, {"method": method})
        c["base_checkpoint"], c["oracle_checkpoint"] = cfg["base_checkpoint"], cfg["oracle_checkpoint"]
        d = run_dir(c)
        meta = start_entry(c, d, "smoke")
        unlearn_cell(c)
        rec = audit_diffusion_cell(c)
        finish_entry(meta, {"attr_drop": rec.get("attr_drop"), "relearn_steps_to_90": rec.get("relearn_steps_to_90")})
        print(method, json.dumps({k: v for k, v in rec.items() if isinstance(v, (int, float))}, default=float))
    analyze(os.environ["RESULTS_ROOT"])
    if os.environ.get("LLM_SMOKE") == "1":
        _llm_smoke(tmp)
    print("SMOKE OK ->", tmp)


def _llm_smoke(tmp: str) -> None:
    from .llm.unlearn import unlearn_llm
    from .llm.finetune import finetune

    cfg = {
        "experiment_id": "SMOKE_LLM",
        "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "auto", "max_length": 256, "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}},
        "data": {"forget_frac": 0.05},
        "train": {"learning_rate": 1e-4, "batch_size": 4, "grad_accum": 1, "epochs": 1},
        "unlearn": {"method": "npo", "forget_frac": 0.05, "retain_available": True, "steps": 10, "learning_rate": 1e-5, "npo_beta": 0.1, "batch_size": 4, "seed": 0, "full_ft": False},
        "logging": {"output_dir": os.path.join(tmp, "results")},
    }
    base = finetune(cfg, retain_only=False, full_ft=False, seed=0)
    cfg["base_checkpoint"] = base["checkpoint"]
    from .common.config import resolve, run_dir

    c = resolve(cfg, {})
    c["base_checkpoint"] = base["checkpoint"]
    print("llm unlearn:", unlearn_llm(c))
