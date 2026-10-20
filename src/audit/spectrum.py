"""Spectrum of the parameter update delta = theta_unlearned - theta_original (H3).

Per 2-D parameter tensor (conv kernels reshaped to out x (in*k*k); 1-D tensors skipped):
  singular values s_1 >= s_2 >= ...
  top5_energy     = sum_{i<=5} s_i^2 / sum_i s_i^2        (fraction of ||delta||_F^2 in 5 directions)
  effective_rank  = exp( H(p) ),  p_i = s_i / sum s      (Roy & Vetterli 2007)
  norm            = ||delta||_F
Aggregates: norm-weighted means over layers plus the total norm (the confound
H3 controls for).  numpy functions are unit-tested; ``spectrum_of_checkpoints``
is the torch wrapper.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def matrix_spectrum_np(delta: np.ndarray, top_k: int = 5) -> Dict[str, float]:
    m = delta.reshape(delta.shape[0], -1).astype(np.float64)
    if min(m.shape) < 2:
        return {"norm": float(np.linalg.norm(m)), "effective_rank": 1.0, "top5_energy": 1.0, "n_singular": 1}
    s = np.linalg.svd(m, compute_uv=False)
    e = s**2
    tot = e.sum()
    if tot <= 0:
        return {"norm": 0.0, "effective_rank": 0.0, "top5_energy": 0.0, "n_singular": int(len(s))}
    p = s / s.sum()
    p = p[p > 0]
    ent = -(p * np.log(p)).sum()
    return {"norm": float(np.sqrt(tot)), "effective_rank": float(np.exp(ent)), "top5_energy": float(e[:top_k].sum() / tot), "n_singular": int(len(s))}


def aggregate_spectra(per_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Norm^2-weighted means so that layers that moved more count more."""
    if not per_layer:
        return {"delta_norm_total": 0.0, "effective_rank_wmean": float("nan"), "top5_energy_wmean": float("nan"), "n_layers": 0}
    w = np.array([v["norm"] ** 2 for v in per_layer.values()])
    tot = w.sum()
    if tot <= 0:
        return {"delta_norm_total": 0.0, "effective_rank_wmean": 0.0, "top5_energy_wmean": 0.0, "n_layers": len(per_layer)}
    w = w / tot
    er = np.array([v["effective_rank"] for v in per_layer.values()])
    t5 = np.array([v["top5_energy"] for v in per_layer.values()])
    return {"delta_norm_total": float(np.sqrt(tot)), "effective_rank_wmean": float((w * er).sum()), "top5_energy_wmean": float((w * t5).sum()), "n_layers": int(len(per_layer)), "effective_rank_median": float(np.median(er))}


def spectrum_of_state_dicts(sd_unlearned: Dict, sd_original: Dict, top_k: int = 5, min_dim: int = 8) -> Dict:
    per: Dict[str, Dict[str, float]] = {}
    for k, v in sd_unlearned.items():
        if k not in sd_original:
            continue
        a = np.asarray(v.detach().float().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
        b = np.asarray(sd_original[k].detach().float().cpu().numpy() if hasattr(sd_original[k], "detach") else sd_original[k], dtype=np.float64)
        if a.ndim < 2 or a.shape[0] < min_dim or a.size // a.shape[0] < min_dim:
            continue
        per[k] = matrix_spectrum_np(a - b, top_k)
    return {"per_layer": per, **aggregate_spectra(per)}


def spectrum_of_checkpoints(unlearned_sd: Dict, original_sd: Dict, top_k: int = 5) -> Dict:
    return spectrum_of_state_dicts(unlearned_sd, original_sd, top_k)


def lora_delta_spectrum(model_after, model_before, top_k: int = 5) -> Dict:
    """Per-target-module LoRA delta = (alpha/r) * B @ A, differenced between two PEFT models.

    By construction each module delta has rank <= r, so the spectrum is reported for completeness but H3
    excludes LoRA cells (the record carries ``lora: true`` so the hypothesis test can drop it).
    """
    import torch

    def deltas(m):
        out = {}
        for name, mod in m.named_modules():
            A = getattr(mod, "lora_A", None)
            B = getattr(mod, "lora_B", None)
            if A is None or B is None:
                continue
            try:
                keys = list(A.keys()) if hasattr(A, "keys") else ["default"]
                for k in keys:
                    a = A[k].weight if hasattr(A, "__getitem__") else A.weight
                    b = B[k].weight if hasattr(B, "__getitem__") else B.weight
                    scaling = getattr(mod, "scaling", {k: 1.0})
                    sc = scaling[k] if isinstance(scaling, dict) else float(scaling)
                    out[f"{name}.{k}"] = (sc * (b @ a)).detach().float().cpu().numpy()
            except Exception:
                continue
        return out

    da, db = deltas(model_after), deltas(model_before)
    per = {}
    for k in da:
        d = da[k] - db.get(k, 0.0)
        if np.ndim(d) >= 2 and min(d.shape) >= 2:
            per[k] = matrix_spectrum_np(np.asarray(d), top_k)
    return {"per_layer": per, "lora": True, **aggregate_spectra(per)}
