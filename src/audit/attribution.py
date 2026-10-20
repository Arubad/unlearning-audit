"""TRAK-style influence of forget examples on retain outputs (H2).

For a checkpoint theta we compute per-example gradients g(z) of the training loss
(eps-MSE at a few fixed timesteps for diffusion; per-example NLL for the LLM),
project them with a fixed-seed Rademacher matrix P to d = 4096 dimensions,
phi(z) = P g(z), and score

    score(z_forget, z_query) = phi(z_query)^T (Phi^T Phi + lambda I)^-1 phi(z_forget)

where Phi stacks the projected gradients of a training subset (TRAK, Park et al.
2023, without the multi-model ensemble).  The audit statistic is the mean
|score| of forget examples on retain queries, and its *drop* from the original
checkpoint to the unlearned one, compared with the drop to the oracle retrain.
Per-example gradients use torch.func (functional_call + vmap + grad).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..common.stats import bootstrap_ci


def rademacher_projection(d_in: int, d_out: int, seed: int, block: int = 2**16):
    """Generator of +-1/sqrt(d_out) blocks so a 10M x 4096 matrix never exists at once."""
    rng = np.random.default_rng(seed)
    for s in range(0, d_in, block):
        n = min(block, d_in - s)
        yield s, (rng.integers(0, 2, size=(n, d_out)).astype(np.float32) * 2 - 1) / np.sqrt(d_out)


def project_np(grads: np.ndarray, d_out: int, seed: int) -> np.ndarray:
    """grads: (n, d_in) -> (n, d_out) with the same fixed projection used everywhere."""
    out = np.zeros((grads.shape[0], d_out), dtype=np.float32)
    for s, P in rademacher_projection(grads.shape[1], d_out, seed):
        out += grads[:, s : s + P.shape[0]].astype(np.float32) @ P
    return out


def project_torch(flat, d_out: int, seed: int, block: int = 2**14):
    """Device-side projection with the same Rademacher structure as ``project_np`` (fixed seed -> fixed matrix).

    ``flat``: (n, d_in) torch tensor on the compute device.  Blocks of the projection matrix are generated on the
    device with a seeded generator each call, so the full 10M x 4096 matrix never exists and nothing is moved to
    the CPU.  The numpy and torch projections use different RNG streams; every phi within a run uses this one.
    """
    import torch

    device = flat.device
    g = torch.Generator(device=device).manual_seed(seed)
    out = torch.zeros((flat.shape[0], d_out), dtype=torch.float32, device=device)
    scale = 1.0 / float(np.sqrt(d_out))
    mm_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for s in range(0, flat.shape[1], block):
        n = min(block, flat.shape[1] - s)
        P = torch.randint(0, 2, (n, d_out), generator=g, device=device, dtype=torch.int8).to(mm_dtype) * 2 - 1
        out += (flat[:, s : s + n].to(mm_dtype) @ P).float() * scale
    return out


def trak_scores_np(phi_train: np.ndarray, phi_forget: np.ndarray, phi_query: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Return (n_forget, n_query) influence scores.  phi_* are projected gradients."""
    d = phi_train.shape[1]
    K = phi_train.T @ phi_train / phi_train.shape[0] + lam * np.eye(d, dtype=phi_train.dtype)
    Kinv_phi_f = np.linalg.solve(K, phi_forget.T)  # (d, n_forget)
    return (phi_query @ Kinv_phi_f).T


def attribution_summary(scores_original: np.ndarray, scores_after: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Mean |score| before and after, the relative drop, and a bootstrap CI over forget examples."""
    a0 = np.abs(scores_original).mean(axis=1)
    a1 = np.abs(scores_after).mean(axis=1)
    drop = 1.0 - a1 / np.maximum(a0, 1e-12)
    ci = bootstrap_ci(drop, np.mean, n_boot=2000, seed=seed)
    return {"attr_mean_abs_original": float(a0.mean()), "attr_mean_abs_after": float(a1.mean()), "attr_drop": float(np.mean(drop)), "attr_drop_lo": ci["lo"], "attr_drop_hi": ci["hi"], "attr_surviving_influence_per_example": (a1 / np.maximum(a0, 1e-12)).tolist()}


# ---------------------------------------------------------------- torch wrappers


def per_example_grads_torch(model, loss_fn: Callable, batches, n_params_mask: Optional[Dict] = None, device=None):
    """Yield (n_batch, d) flattened per-example gradients using torch.func.

    ``loss_fn(params, buffers, x, y, extra)`` must return a scalar for ONE example.
    """
    import torch
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def one(p, b, x, y, extra):
        return loss_fn(p, b, x.unsqueeze(0), y.unsqueeze(0), extra)

    g = vmap(grad(one), in_dims=(None, None, 0, 0, None))
    for x, y, extra in batches:
        gd = g(params, buffers, x, y, extra)
        flat = torch.cat([gd[k].reshape(x.shape[0], -1) for k in params], dim=1)
        yield flat  # stays on the device; callers project before moving to the CPU


def projected_grads_diffusion(model, diff, dataset, indices, timesteps, proj_dim: int, proj_seed: int, device, batch: int = 16) -> np.ndarray:
    """phi for a list of CIFAR indices: average of projected gradients over the fixed timesteps (fixed noise)."""
    import torch
    from torch.func import functional_call

    from ..diffusion.data import make_loader

    loader = make_loader(dataset, indices, batch, False, 0, 2, drop_last=False)
    out = []
    gen = torch.Generator(device="cpu").manual_seed(proj_seed)

    def loss_fn(p, b, x, y, extra):
        t, noise = extra
        xt = diff.q_sample(x, t, noise)
        pred = functional_call(model, (p, b), (xt, t, y))
        return ((pred - noise) ** 2).mean()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        acc = None
        for tval in timesteps:
            t = torch.full((x.shape[0],), int(tval), device=device, dtype=torch.long)
            noise = torch.randn(x.shape, generator=gen).to(device)
            for flat in per_example_grads_torch(model, loss_fn, [(x, y, (t, noise))], device=device):
                phi = project_torch(flat, proj_dim, proj_seed).cpu().numpy()
                acc = phi if acc is None else acc + phi
        out.append(acc / len(timesteps))
    return np.concatenate(out)


def attribution_diffusion(model_original, model_after, model_oracle, diff, dataset, split: Dict, cfg: Dict, device, seed: int = 0) -> Dict[str, float]:
    a = cfg["audit"]["attribution"]
    rng = np.random.default_rng(seed)
    forget = rng.choice(split["forget"], size=min(int(a.get("n_forget", 256)), len(split["forget"])), replace=False).tolist()
    retain = rng.choice(split["retain"], size=min(int(a.get("train_subset", 2000)), len(split["retain"])), replace=False).tolist()
    queries = retain[: int(a.get("n_retain_queries", 256))]
    ts = [int(t) for t in a.get("timesteps", [100, 400, 700])]
    d, ps, lam = int(a["projection_dim"]), int(a["projection_seed"]), float(a["hessian_damping"])
    res = {}
    scores = {}
    for name, m in (("original", model_original), ("after", model_after), ("oracle", model_oracle)):
        if m is None:
            continue
        m.eval()
        phi_tr = projected_grads_diffusion(m, diff, dataset, retain, ts, d, ps, device)
        phi_f = projected_grads_diffusion(m, diff, dataset, forget, ts, d, ps, device)
        phi_q = phi_tr[: len(queries)]
        scores[name] = trak_scores_np(phi_tr, phi_f, phi_q, lam)
    s = attribution_summary(scores["original"], scores["after"], seed)
    res.update({k: v for k, v in s.items() if not k.endswith("per_example")})
    res["attr_surviving_influence_per_example"] = s["attr_surviving_influence_per_example"]
    if "oracle" in scores:
        so = attribution_summary(scores["original"], scores["oracle"], seed)
        res.update({"attr_drop_oracle": so["attr_drop"], "attr_drop_oracle_lo": so["attr_drop_lo"], "attr_drop_oracle_hi": so["attr_drop_hi"]})
    return res


# ---------------------------------------------------------------- LLM attribution


def projected_grads_llm(model, tok, examples, proj_dim: int, proj_seed: int, device, batch: int = 4, param_subset_seed: int = 0, max_params: int = 2_000_000) -> np.ndarray:
    """phi for a list of LLM examples: per-example gradient of the answer NLL over the trainable parameters,
    projected to ``proj_dim``.  For full-FT models the parameter vector is huge, so a fixed random coordinate
    subset of size ``max_params`` is used (documented in the audit record); for LoRA all trainable params are used.
    """
    import torch
    from torch.func import functional_call, grad, vmap

    from ..llm.data import collate

    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    total = sum(v.numel() for v in params.values())
    coord_idx = None
    if total > max_params:
        rng = np.random.default_rng(param_subset_seed)
        coord_idx = np.sort(rng.choice(total, size=max_params, replace=False))

    def one(p, b, ids, attn, labels):
        out = functional_call(model, (p, b), (), {"input_ids": ids.unsqueeze(0), "attention_mask": attn.unsqueeze(0)})
        logits = out.logits[:, :-1]
        tgt = labels.unsqueeze(0)[:, 1:]
        mask = (tgt != -100).float()
        lp = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
        return -(lp * mask).sum() / mask.sum().clamp(min=1)

    out_rows = []
    for s in range(0, len(examples), batch):
        chunk = examples[s : s + batch]
        b = collate(chunk, tok.pad_token_id, device)
        try:
            g = vmap(grad(one), in_dims=(None, None, 0, 0, 0))(params, buffers, b["input_ids"], b["attention_mask"], b["labels"])
            flat = torch.cat([g[k].reshape(len(chunk), -1) for k in params], dim=1)
        except Exception:  # vmap over grad can fail for some ops; fall back to a per-example loop
            rows = []
            for i in range(len(chunk)):
                gi = grad(one)(params, buffers, b["input_ids"][i], b["attention_mask"][i], b["labels"][i])
                rows.append(torch.cat([gi[k].reshape(-1) for k in params]))
            flat = torch.stack(rows)
        if coord_idx is not None:
            flat = flat[:, torch.as_tensor(coord_idx, device=flat.device)]
        out_rows.append(project_torch(flat, proj_dim, proj_seed).cpu().numpy())
        del flat
    return np.concatenate(out_rows)


def attribution_llm(model_original, model_after, model_oracle, tok, forget_ex, retain_ex, cfg: Dict, device, seed: int = 0) -> Dict[str, float]:
    a = cfg["audit"]["attribution"]
    rng = np.random.default_rng(seed)
    nf = min(int(a.get("n_forget", 200)), len(forget_ex))
    nq = min(int(a.get("n_query", 200)), len(retain_ex))
    forget = [forget_ex[i] for i in rng.choice(len(forget_ex), size=nf, replace=False)]
    train = [retain_ex[i] for i in rng.choice(len(retain_ex), size=min(int(a.get("train_subset", 400)), len(retain_ex)), replace=False)]
    queries = train[:nq]
    d, ps, lam = int(a["projection_dim"]), int(a["projection_seed"]), float(a["hessian_damping"])
    mp = int(a.get("max_full_ft_params", 2_000_000))
    scores = {}
    for name, m in (("original", model_original), ("after", model_after), ("oracle", model_oracle)):
        if m is None:
            continue
        m.eval()
        phi_tr = projected_grads_llm(m, tok, train, d, ps, device, max_params=mp)
        phi_f = projected_grads_llm(m, tok, forget, d, ps, device, max_params=mp)
        phi_q = phi_tr[: len(queries)]
        scores[name] = trak_scores_np(phi_tr, phi_f, phi_q, lam)
    res = {}
    s = attribution_summary(scores["original"], scores["after"], seed)
    res.update({k: v for k, v in s.items() if not k.endswith("per_example")})
    res["attr_surviving_influence_per_example"] = s["attr_surviving_influence_per_example"]
    if "oracle" in scores:
        so = attribution_summary(scores["original"], scores["oracle"], seed)
        res.update({"attr_drop_oracle": so["attr_drop"], "attr_drop_oracle_lo": so["attr_drop_lo"], "attr_drop_oracle_hi": so["attr_drop_hi"]})
    return res
