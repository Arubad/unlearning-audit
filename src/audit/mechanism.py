"""Mechanism readouts for H3 (amendment 2026-11-18): gradient alignment and the rank-truncation intervention.

Why alignment, not only rank
----------------------------
Linearise the forget loss around the unlearned point theta_u = theta_0 + Delta.  One relearning step on the
forget set moves by -eta * g_f(theta_u).  The relearning distance needed is governed by how much of Delta lies
along directions that the forget gradient points back through:

    a(Delta) = cos( g_f(theta_u), Delta )

If unlearning moved the weights *up* the forget loss along a few directions (gradient ascent on one class),
g_f(theta_u) points straight back along Delta (a close to 1) and relearning is a short walk back.  A retrain
moves the weights to an unrelated basin; its Delta is not aligned with g_f (a close to 0).  Low rank is one way
to get high alignment, but not the only one, so we measure both and let the data say which predicts relearning
at matched norm.

Why an intervention
-------------------
Across-method correlations of rank with relearning (about 6-12 cells) confound rank with the method's
identity.  The intervention keeps the method and the norm fixed and varies only the rank: for each 2-D tensor,
replace Delta_l by its best rank-k approximation, then rescale all truncated tensors jointly so ||Delta||_F is
unchanged.  If H3's mechanism holds, small k already reproduces the forgetting (forget-class generation rate
stays low) and relearning time is flat or increasing in k.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ numpy cores

def flat_cosine(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    keys = [k for k in a if k in b]
    num = sum(float(np.sum(np.asarray(a[k], float) * np.asarray(b[k], float))) for k in keys)
    na = np.sqrt(sum(float(np.sum(np.asarray(a[k], float) ** 2)) for k in keys))
    nb = np.sqrt(sum(float(np.sum(np.asarray(b[k], float) ** 2)) for k in keys))
    return float(num / (na * nb)) if na > 0 and nb > 0 else float("nan")


def truncate_rank_np(delta: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation of a tensor viewed as out x (rest)."""
    m = delta.reshape(delta.shape[0], -1)
    if k >= min(m.shape):
        return delta.copy()
    U, s, Vt = np.linalg.svd(m, full_matrices=False)
    return ((U[:, :k] * s[:k]) @ Vt[:k]).reshape(delta.shape)


def truncated_delta_np(deltas: Dict[str, np.ndarray], k: int, min_dim: int = 8, match_norm: bool = True) -> Dict[str, np.ndarray]:
    """Rank-k truncation of every eligible 2-D (or conv) tensor; ineligible tensors are kept.  With
    ``match_norm`` the truncated tensors are rescaled jointly so the total ||Delta||_F equals the original."""
    out, trunc_keys = {}, []
    for name, d in deltas.items():
        d = np.asarray(d)
        if d.ndim >= 2 and d.shape[0] >= min_dim and d.size // d.shape[0] >= min_dim:
            out[name] = truncate_rank_np(d, k)
            trunc_keys.append(name)
        else:
            out[name] = d.copy()
    if match_norm and trunc_keys:
        before = np.sqrt(sum(float(np.sum(np.asarray(deltas[n], float) ** 2)) for n in trunc_keys))
        after = np.sqrt(sum(float(np.sum(out[n].astype(float) ** 2)) for n in trunc_keys))
        if after > 0:
            for n in trunc_keys:
                out[n] = out[n] * (before / after)
    return out


# ------------------------------------------------------------------ torch helpers

def forget_gradient(model, diff, dataset, indices: Sequence[int], device, n_batches: int = 4, batch: int = 64, seed: int = 0) -> Dict[str, np.ndarray]:
    """Mean gradient of the forget eps-loss at the model's current parameters, with fixed (t, noise)."""
    import torch
    from torch.utils.data import DataLoader, Subset

    rng = np.random.default_rng(seed)
    idx = rng.choice(np.asarray(indices), size=min(len(indices), n_batches * batch), replace=False).tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=batch, shuffle=False, num_workers=2)
    model.eval()  # no dropout: the gradient of the deterministic network
    model.zero_grad(set_to_none=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        t = torch.randint(0, diff.T, (x.shape[0],), generator=g).to(device)
        noise = torch.randn(x.shape, generator=g).to(device)
        diff.loss(model, x, y, t=t, noise=noise).backward()
        nb += 1
    grads = {n: (p.grad.detach().float().cpu().numpy() / nb) for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


def relearn_alignment(model_after, original_sd: Dict, diff, dataset, forget_indices: Sequence[int], device, seed: int = 0) -> Dict[str, float]:
    sd = model_after.state_dict()
    names = {n for n, _ in model_after.named_parameters()}
    delta = {n: (sd[n].detach().float().cpu().numpy() - original_sd[n].detach().float().cpu().numpy()) for n in names}
    gf = forget_gradient(model_after, diff, dataset, forget_indices, device, seed=seed)
    gnorm = float(np.sqrt(sum(float(np.sum(v.astype(float) ** 2)) for v in gf.values())))
    return {"relearn_alignment": flat_cosine(gf, delta), "forget_grad_norm_after": gnorm}


def apply_delta(model, original_sd: Dict, delta: Dict[str, np.ndarray]):
    import torch

    new = {}
    for k, v in original_sd.items():
        if k in delta:
            new[k] = (v.detach().float().cpu() + torch.as_tensor(delta[k], dtype=torch.float32)).to(v.dtype)
        else:
            new[k] = v
    model.load_state_dict(new)
    return model
