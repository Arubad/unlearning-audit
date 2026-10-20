"""Membership inference: four attacks on one checkpoint, all reported as AUC (forget members vs held-out non-members).

Every attack is a *score per example*, larger = "member", and the AUC is rank
based (``auc_from_scores``).  Non-members are held-out examples **of the forget
class / forget distribution** (CIFAR-10 test images of the forget class; TOFU
examples never fine-tuned on), so the attacks measure memorisation of the
forgotten data, not class membership.

  loss      -(per-example loss)            diffusion: eps-MSE averaged over 5 fixed timesteps with fixed noise;
                                            LLM: mean token NLL of the answer.  Weakest, most common certificate.
  secmi     -(DDIM t-error at t*)           Duan et al. 2023: deterministic DDIM step forward then backward from
                                            x_t; the reconstruction error is smaller for members.  Diffusion only.
  lira      offline LiRA, Carlini et al.    OUT distribution = the per-example loss under the oracle retrain(s)
            2022, with the oracle as OUT    (models that never saw the forget set).  Score = (mean_OUT - loss_target)
                                            / sd, i.e. how much lower the target model's loss is than a model that
                                            never trained on the example.  With < 3 oracles the per-example sd is
                                            replaced by the global sd of OUT losses.
  cmia      class-geometry MIA              "Unlearning Isn't Forgetting" (ICML 2026): members of the forgotten
            (feature space)                 class sit closer to the forget-class centroid than non-members do;
                                            score = min distance to other centroids - distance to the forget-class
                                            centroid, in the *audited* model's feature space with centroids from
                                            retain data (a truly retrained model loses the geometry).

The numpy cores (``lira_scores_np``, ``cmia_scores_np``, ``auc_from_scores``) are
unit-tested; the torch helpers only produce per-example losses and features.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ numpy cores

def auc_from_scores(member_scores: Sequence[float], nonmember_scores: Sequence[float]) -> float:
    """Rank-based ROC AUC where larger scores mean 'member'; ties count one half."""
    m = np.asarray(member_scores, dtype=float)
    n = np.asarray(nonmember_scores, dtype=float)
    if len(m) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([m, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rm = ranks[: len(m)].sum()
    return float((rm - len(m) * (len(m) + 1) / 2) / (len(m) * len(n)))


def lira_scores_np(loss_target: np.ndarray, loss_out: np.ndarray) -> np.ndarray:
    """Offline LiRA score per example.

    ``loss_target``: (n,) loss under the audited model.  ``loss_out``: (k, n) losses under k models that never saw
    the examples (the oracle retrains).  Score = (mean_out - loss_target) / sd, sd = per-example std over the k OUT
    models when k >= 3, else the global std of OUT losses.  Larger = the target fits the example better than a
    non-member model would = member.
    """
    loss_target = np.asarray(loss_target, dtype=float)
    loss_out = np.atleast_2d(np.asarray(loss_out, dtype=float))
    mu = loss_out.mean(0)
    if loss_out.shape[0] >= 3:
        sd = loss_out.std(0) + 1e-8
    else:
        sd = np.full_like(mu, loss_out.std() + 1e-8)
    return (mu - loss_target) / sd


def cmia_scores_np(feats: np.ndarray, labels: Optional[np.ndarray], forget_class: int, centroids: np.ndarray) -> np.ndarray:
    """Class-geometry score for each row of ``feats`` (all assumed to be forget-class examples).

    ``centroids``: (n_classes, d) class centroids in the same feature space (from retain data, so members and
    non-members are scored against one geometry).  Score = min_{c != fc} ||f - mu_c|| - ||f - mu_fc||: positive when
    the example sits inside the forget class's region.  ``labels`` is unused and kept for interface symmetry.
    """
    feats = np.asarray(feats, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    d = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=-1)  # (n, C)
    own = d[:, forget_class]
    others = np.delete(d, forget_class, axis=1).min(1)
    return others - own


def class_centroids_np(feats: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    feats = np.asarray(feats, dtype=float)
    labels = np.asarray(labels)
    out = np.zeros((n_classes, feats.shape[1]))
    for c in range(n_classes):
        m = labels == c
        out[c] = feats[m].mean(0) if m.any() else 0.0
    return out


def mia_from_losses(loss_member: np.ndarray, loss_nonmember: np.ndarray) -> float:
    return auc_from_scores(-np.asarray(loss_member), -np.asarray(loss_nonmember))


# ------------------------------------------------------------------ torch helpers: diffusion

def _fixed_noise(shape, seed: int, device):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g).to(device)


def _subset_loader(dataset, indices: Sequence[int], batch: int):
    from torch.utils.data import DataLoader, Subset

    return DataLoader(Subset(dataset, [int(i) for i in indices]), batch_size=batch, shuffle=False, num_workers=2)


def diffusion_losses(model, diff, dataset, indices: Sequence[int], timesteps: Sequence[int], device, batch: int = 128, noise_seed: int = 1234) -> np.ndarray:
    """Per-example eps-MSE averaged over ``timesteps`` with fixed noise per (example, timestep).  (n,)"""
    import torch

    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        k = 0
        for x, y in _subset_loader(dataset, indices, batch):
            x, y = x.to(device), y.to(device)
            acc = torch.zeros(x.shape[0], device=device)
            for ti, t in enumerate(timesteps):
                noise = _fixed_noise(x.shape, noise_seed + 1000 * ti + k, device)
                tb = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
                acc += diff.loss(model, x, y, t=tb, noise=noise, reduction="none")
            out.append((acc / len(timesteps)).cpu().numpy())
            k += x.shape[0]
    return np.concatenate(out)


def secmi_t_error(model, diff, dataset, indices: Sequence[int], t_star: int, device, batch: int = 128, k_steps: int = 10) -> np.ndarray:
    """SecMI-style t-error (Duan et al. 2023).

    x_0 is mapped to x_{t*} by the deterministic DDIM forward map (6 hops using the model's own eps), then one DDIM
    step forward (t* -> t*+k) and one back; the squared reconstruction error of x_{t*} is smaller for members.  (n,)
    """
    import torch

    model.eval()
    out: List[np.ndarray] = []
    t0, t1 = int(t_star), int(min(t_star + k_steps, diff.T - 1))

    def ddim_move(x, t_from: int, t_to: int, y):
        tb = torch.full((x.shape[0],), int(t_from), device=device, dtype=torch.long)
        eps = model(x, tb, y)
        a_from, a_to = diff.abar[t_from], diff.abar[t_to]
        x0 = (x - torch.sqrt(1 - a_from) * eps) / torch.sqrt(a_from)
        return torch.sqrt(a_to) * x0 + torch.sqrt(1 - a_to) * eps

    with torch.no_grad():
        for x, y in _subset_loader(dataset, indices, batch):
            x, y = x.to(device), y.to(device)
            hops = [int(v) for v in np.linspace(0, t0, 6)]
            xt = x
            for a, b in zip(hops[:-1], hops[1:]):
                if b > a:
                    xt = ddim_move(xt, a, b, y)
            x_fwd = ddim_move(xt, t0, t1, y)
            x_back = ddim_move(x_fwd, t1, t0, y)
            out.append(((x_back - xt) ** 2).flatten(1).mean(1).cpu().numpy())
    return np.concatenate(out)


def diffusion_features(model, diff, dataset, indices: Sequence[int], timestep: int, device, batch: int = 128, noise_seed: int = 4321) -> np.ndarray:
    """Mid-block pooled features at a fixed timestep with fixed noise and the null label (for CMIA).  (n, d)"""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        k = 0
        for x, y in _subset_loader(dataset, indices, batch):
            x, y = x.to(device), y.to(device)
            noise = _fixed_noise(x.shape, noise_seed + k, device)
            tb = torch.full((x.shape[0],), int(timestep), device=device, dtype=torch.long)
            xt = diff.q_sample(x, tb, noise)
            out.append(model.features(xt, tb, torch.full_like(y, model.null_class)).float().cpu().numpy())
            k += x.shape[0]
    return np.concatenate(out)


def mia_suite(model, original_model, oracle_models, data, cfg: Dict, device) -> Dict[str, float]:
    """Diffusion track.  ``data`` = (cifar_train_dataset, split dict with 'forget' and 'retain' indices).

    Returns ``mia_loss_auc``, ``mia_secmi_auc``, ``mia_lira_auc`` (NaN without an oracle), ``mia_cmia_auc`` and counts.
    """
    from ..diffusion.data import cifar_labels, load_cifar
    from ..diffusion.ddpm import Diffusion

    ds, split = data
    a = cfg["audit"]["mia"]
    fc = int(cfg["data"]["forget_class"])
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    rng = np.random.default_rng(int(a.get("seed", 0)))
    n_max = int(a.get("n_examples", 1000))
    forget = np.array(split["forget"])
    forget = rng.choice(forget, size=min(n_max, len(forget)), replace=False)
    test = load_cifar(cfg["data"].get("root", "data"), train=False)
    test_labels = cifar_labels(test)
    held = np.where(test_labels == fc)[0]
    held = rng.choice(held, size=min(n_max, len(held)), replace=False)
    timesteps = [int(t) for t in a.get("timesteps", [50, 200, 400, 600, 800])]
    out: Dict[str, float] = {}

    lm = diffusion_losses(model, diff, ds, forget, timesteps, device)
    ln = diffusion_losses(model, diff, test, held, timesteps, device)
    out["mia_loss_auc"] = mia_from_losses(lm, ln)
    out["mia_loss_mean_forget"], out["mia_loss_mean_heldout"] = float(lm.mean()), float(ln.mean())

    t_star = int(a.get("secmi_t", 200))
    em = secmi_t_error(model, diff, ds, forget, t_star, device)
    en = secmi_t_error(model, diff, test, held, t_star, device)
    out["mia_secmi_auc"] = mia_from_losses(em, en)

    if oracle_models:
        om = np.stack([diffusion_losses(o, diff, ds, forget, timesteps, device) for o in oracle_models])
        on = np.stack([diffusion_losses(o, diff, test, held, timesteps, device) for o in oracle_models])
        out["mia_lira_auc"] = auc_from_scores(lira_scores_np(lm, om), lira_scores_np(ln, on))
    else:
        out["mia_lira_auc"] = float("nan")

    t_feat = int(a.get("feature_timestep", 100))
    labels = cifar_labels(ds)
    retain = np.array(split["retain"])
    retain = rng.choice(retain, size=min(int(a.get("n_centroid", 2000)), len(retain)), replace=False)
    f_ret = diffusion_features(model, diff, ds, retain, t_feat, device)
    cent = class_centroids_np(f_ret, labels[retain], int(cfg["model"].get("n_classes", 10)))
    f_m = diffusion_features(model, diff, ds, forget, t_feat, device)
    f_n = diffusion_features(model, diff, test, held, t_feat, device)
    out["mia_cmia_auc"] = auc_from_scores(cmia_scores_np(f_m, None, fc, cent), cmia_scores_np(f_n, None, fc, cent))
    out["mia_n_members"], out["mia_n_nonmembers"] = int(len(forget)), int(len(held))
    return out


# ------------------------------------------------------------------ torch helpers: LLM

def _answer_token_logprobs(model, tok, examples: List[Dict], device, batch: int):
    """Yield (logprobs [b, T-1], mask [b, T-1]) of answer tokens, batch by batch."""
    import torch

    from ..llm.data import collate

    model.eval()
    with torch.no_grad():
        for s in range(0, len(examples), batch):
            b = collate(examples[s : s + batch], tok.pad_token_id, device)
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1]
            tgt = b["labels"][:, 1:]
            mask = tgt != -100
            lp = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1).float() - torch.logsumexp(logits.float(), -1)
            yield lp, mask


def llm_answer_nll(model, tok, examples: List[Dict], device, batch: int = 8) -> np.ndarray:
    """Mean per-token NLL of the answer span for each example.  (n,)"""
    out: List[np.ndarray] = []
    for lp, mask in _answer_token_logprobs(model, tok, examples, device, batch):
        m = mask.float()
        out.append((-(lp * m).sum(-1) / m.sum(-1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def llm_mink_scores(model, tok, examples: List[Dict], device, k_frac: float = 0.2, batch: int = 8) -> np.ndarray:
    """Min-K% prob (Shi et al. 2024): mean log-prob of the k% least likely answer tokens; larger = member.  (n,)"""
    import torch

    out: List[float] = []
    for lp, mask in _answer_token_logprobs(model, tok, examples, device, batch):
        for i in range(lp.shape[0]):
            v = lp[i][mask[i]]
            k = max(1, int(k_frac * v.numel()))
            out.append(float(torch.sort(v).values[:k].mean().item()))
    return np.array(out)


def llm_features(model, tok, examples: List[Dict], device, layer: int = -1, batch: int = 8) -> np.ndarray:
    """Mean hidden state of the answer tokens at ``layer`` (for CMIA and probes).  (n, d)"""
    import torch

    from ..llm.data import collate

    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(examples), batch):
            b = collate(examples[s : s + batch], tok.pad_token_id, device)
            hs = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True).hidden_states[layer]
            mask = (b["labels"] != -100).float().unsqueeze(-1)
            out.append(((hs * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(out)


def mia_suite_llm(model, tok, forget_ex: List[Dict], heldout_ex: List[Dict], retain_ex: List[Dict], oracle_models: List, device, cfg: Dict) -> Dict[str, float]:
    """LLM track.  Members = forget examples; non-members = TOFU examples never fine-tuned on.

    CMIA uses a two-centroid geometry (centroid of the forget examples vs centroid of a matched number of retain
    examples) computed on the audited model; score = ||f - mu_retain|| - ||f - mu_forget||.
    """
    a = cfg["audit"]["mia"]
    layer = int(a.get("feature_layer", -1))
    out: Dict[str, float] = {}
    lm = llm_answer_nll(model, tok, forget_ex, device)
    ln = llm_answer_nll(model, tok, heldout_ex, device)
    out["mia_loss_auc"] = mia_from_losses(lm, ln)
    out["mia_loss_mean_forget"], out["mia_loss_mean_heldout"] = float(lm.mean()), float(ln.mean())
    out["mia_mink_auc"] = auc_from_scores(llm_mink_scores(model, tok, forget_ex, device), llm_mink_scores(model, tok, heldout_ex, device))
    out["mia_secmi_auc"] = float("nan")  # diffusion only
    if oracle_models:
        om = np.stack([llm_answer_nll(o, tok, forget_ex, device) for o in oracle_models])
        on = np.stack([llm_answer_nll(o, tok, heldout_ex, device) for o in oracle_models])
        out["mia_lira_auc"] = auc_from_scores(lira_scores_np(lm, om), lira_scores_np(ln, on))
    else:
        out["mia_lira_auc"] = float("nan")
    f_f = llm_features(model, tok, forget_ex, device, layer)
    f_h = llm_features(model, tok, heldout_ex, device, layer)
    f_r = llm_features(model, tok, retain_ex[: len(forget_ex)], device, layer)
    cent = np.stack([f_f.mean(0), f_r.mean(0)])  # class 0 = forget, 1 = retain
    out["mia_cmia_auc"] = auc_from_scores(cmia_scores_np(f_f, None, 0, cent), cmia_scores_np(f_h, None, 0, cent))
    out["mia_n_members"], out["mia_n_nonmembers"] = int(len(forget_ex)), int(len(heldout_ex))
    return out
