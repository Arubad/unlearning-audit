"""Oracle-normalised closeness: the redesigned H2 statistic (amendment 2026-11-18).

Why the old H2 statistic was replaced
-------------------------------------
The TRAK score tau(f, q) evaluated *at a checkpoint theta* estimates the effect on query q of up-weighting
(adding) training example f at theta.  It is defined, and generally non-zero, for a model that never saw f:
at the oracle it is the effect of adding f back.  Forget examples are fitted by the original model (small
gradients) and unfitted by the oracle (larger gradients), so mean |tau| is expected to *grow* under exact
retraining.  "Attribution drops by > 90% under retraining" was therefore not a prediction the oracle could
meet, and the test could not discriminate methods.

What replaces it
----------------
Unlearning is exact when the unlearned model is indistinguishable from the retrain.  For any *functional*
of the model -- here the vector of per-example eps-MSE losses at fixed (timestep, noise) on a query set Q --
measure the distance of each model to the oracle and normalise between the two reference points that are
always available:

    d(a, b)  = RMS over (q, t) of  l_a(q, t) - l_b(q, t)
    E(u)     = ( d(original, O) - d(u, O) ) / ( d(original, O) - d_floor )

with O the oracle retrain(s) and d_floor the seed-to-seed distance between two oracle retrains (two
replicates of the same counterfactual).  E = 0: no closer to the oracle than doing nothing.  E = 1: as close
as an independent retrain.  E < 0: moved *away* from the oracle (e.g. over-forgetting: forget losses blown
past where a retrain would put them).  E is computed separately on

    forget     the forget examples themselves (members)
    heldout    held-out examples of the forget class (non-members)
    near       retain examples of semantically near classes (the other animals)
    far        retain examples of far classes (vehicles)

The H2 claim "influence on retain outputs survives" becomes: approximate methods reach high E on `forget`
(the certificate they are tuned for) but low E on `near` (the downstream fingerprint of removing the class,
which a retrain carries and suppression does not).  Over-forgetting and collateral damage show up as negative
E, which the old statistic could not express.

A signed fingerprint is reported as well: the correlation over retain queries of (l_u - l_0) with the
influence-function prediction of the removal effect, sum_f tau_0(f, q), computed at the original model
(``attribution.py``).  corr(l_O - l_0, prediction) validates the attribution tool itself (an LDS-style
check); if it is near zero the influence axis is uninformative at this scale and the paper says so.

This is in the spirit of KLOM (Georgiev et al. 2024, "Attribute-to-Delete"), which compares output
distributions to many oracles; with a free-tier budget we use 2 oracle seeds and losses instead of margins.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

NEAR_CLASSES = (2, 4, 5, 6, 7)  # bird, deer, dog, frog, horse  (forget class 3 = cat)
FAR_CLASSES = (0, 1, 8, 9)  # airplane, automobile, ship, truck


# ------------------------------------------------------------------ numpy cores

def rms_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def closeness_np(l_u: np.ndarray, l_0: np.ndarray, l_oracles: Sequence[np.ndarray], n_boot: int = 2000, seed: int = 0, min_snr: float = 1.2) -> Dict[str, float]:
    """E(u) with a bootstrap CI over query examples (rows).  Arrays are (n_examples, n_timesteps).

    ``l_oracles``: list of loss arrays under k >= 1 oracle retrains.  With k >= 2 the floor is the mean pairwise
    oracle distance; with k == 1 the floor is 0 (reported, flagged ``floor_estimated=False``).
    """
    l_u, l_0 = np.atleast_2d(np.asarray(l_u, float)), np.atleast_2d(np.asarray(l_0, float))
    O = [np.atleast_2d(np.asarray(o, float)) for o in l_oracles]
    n = l_u.shape[0]

    def stat(idx):
        du = np.mean([rms_dist(l_u[idx], o[idx]) for o in O])
        d0 = np.mean([rms_dist(l_0[idx], o[idx]) for o in O])
        if len(O) >= 2:
            fl = np.mean([rms_dist(O[i][idx], O[j][idx]) for i in range(len(O)) for j in range(i + 1, len(O))])
        else:
            fl = 0.0
        den = d0 - fl
        return (d0 - du) / den if abs(den) > 1e-12 else np.nan, du, d0, fl

    E, du, d0, fl = stat(np.arange(n))
    snr = float(d0 / fl) if fl > 0 else float("inf")
    if len(O) >= 2 and snr < min_snr:
        # the original is already (almost) as close to the oracle as two oracles are to each other: removing the
        # forget set left no measurable trace on this query set, so there is nothing for E to normalise.
        return {"E": float("nan"), "E_lo": float("nan"), "E_hi": float("nan"), "d_u_oracle": float(du), "d_orig_oracle": float(d0), "d_floor": float(fl),
                "floor_estimated": True, "snr": snr, "n": int(n)}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        e = stat(rng.integers(0, n, n))[0]
        if np.isfinite(e):
            boots.append(e)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return {"E": float(E), "E_lo": lo, "E_hi": hi, "d_u_oracle": float(du), "d_orig_oracle": float(d0), "d_floor": float(fl), "floor_estimated": len(O) >= 2, "snr": snr, "n": int(n)}


def standardized_closeness_np(l_u: np.ndarray, l_0: np.ndarray, l_oracles: Sequence[np.ndarray], **kw) -> Dict[str, float]:
    """Amendment A4 (post-hoc): ``closeness_np`` after dividing every timestep column by its oracle-pair floor.

    The plain RMS over (example, timestep) is dominated by low-noise timesteps, where eps-MSE differences between
    any two models are largest in absolute terms, and where models trained with and without the forget class
    agree.  Standardising each column by its own seed-to-seed spread weights the timesteps equally in
    noise-floor units, so a difference confined to high noise is no longer averaged away.  Requires >= 2 oracles.
    """
    O = [np.atleast_2d(np.asarray(o, float)) for o in l_oracles]
    if len(O) < 2:
        raise ValueError("standardised closeness needs >= 2 oracles for the per-timestep floor")
    fl_t = np.mean([np.sqrt(np.mean((O[i] - O[j]) ** 2, axis=0)) for i in range(len(O)) for j in range(i + 1, len(O))], axis=0)
    sc = np.where(fl_t > 1e-12, fl_t, 1.0)[None, :]
    return closeness_np(np.asarray(l_u, float) / sc, np.asarray(l_0, float) / sc, [o / sc for o in O], **kw)


def signed_fingerprint_np(l_u: np.ndarray, l_0: np.ndarray, reference: np.ndarray) -> float:
    """Pearson correlation over queries of the mean (over timesteps) loss change u - 0 with a reference vector."""
    du = np.atleast_2d(np.asarray(l_u, float)).mean(1) - np.atleast_2d(np.asarray(l_0, float)).mean(1)
    r = np.asarray(reference, float).ravel()
    if du.std() < 1e-12 or r.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(du, r)[0, 1])


# ------------------------------------------------------------------ query sets

def query_sets(train_labels: np.ndarray, test_labels: np.ndarray, split: Dict, forget_class: int, n_per_set: int, seed: int) -> Dict[str, Dict]:
    """Fixed query sets: {'forget','heldout','near','far'} -> {'source': 'train'|'test', 'indices': [...]}.

    near/far are stratified over their classes and drawn from the *retain* split (members of every model).
    The sets depend only on (split, seed) so every checkpoint is evaluated on identical queries.
    """
    rng = np.random.default_rng(seed)
    retain = np.asarray(split["retain"])
    out = {"forget": {"source": "train", "indices": sorted(rng.choice(split["forget"], size=min(n_per_set, len(split["forget"])), replace=False).tolist())}}
    held = np.where(test_labels == forget_class)[0]
    out["heldout"] = {"source": "test", "indices": sorted(rng.choice(held, size=min(n_per_set, len(held)), replace=False).tolist())}
    for name, classes in (("near", NEAR_CLASSES), ("far", FAR_CLASSES)):
        per = max(1, n_per_set // len(classes))
        idx = []
        for c in classes:
            pool = retain[train_labels[retain] == c]
            idx += rng.choice(pool, size=min(per, len(pool)), replace=False).tolist()
        out[name] = {"source": "train", "indices": sorted(idx)}
    return out


# ------------------------------------------------------------------ torch helper

def losses_per_timestep(model, diff, dataset, indices: Sequence[int], timesteps: Sequence[int], device, batch: int = 128, noise_seed: int = 777) -> np.ndarray:
    """(n, len(timesteps)) eps-MSE with noise fixed per (example position, timestep) -- identical for every model."""
    import torch
    from torch.utils.data import DataLoader, Subset

    model.eval()
    rows: List[np.ndarray] = []
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]), batch_size=batch, shuffle=False, num_workers=2)
    k = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            cols = []
            for ti, t in enumerate(timesteps):
                g = torch.Generator(device="cpu").manual_seed(noise_seed + 100003 * ti + k)
                noise = torch.randn(x.shape, generator=g).to(device)
                tb = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
                cols.append(diff.loss(model, x, y, t=tb, noise=noise, reduction="none").float().cpu().numpy())
            rows.append(np.stack(cols, 1))
            k += x.shape[0]
    return np.concatenate(rows)


def fingerprint_suite(models: Dict[str, object], oracle_models: List, diff, train_ds, test_ds, qsets: Dict[str, Dict], timesteps: Sequence[int], device, seed: int = 0) -> Dict[str, float]:
    """``models`` = {'original': m0, 'after': mu}.  Returns E_<set> (+ CI, distances) for every query set."""
    out: Dict[str, float] = {}
    for name, q in qsets.items():
        ds = train_ds if q["source"] == "train" else test_ds
        l0 = losses_per_timestep(models["original"], diff, ds, q["indices"], timesteps, device)
        lu = losses_per_timestep(models["after"], diff, ds, q["indices"], timesteps, device)
        lo = [losses_per_timestep(o, diff, ds, q["indices"], timesteps, device) for o in oracle_models]
        out[f"loss_mean_{name}_original"], out[f"loss_mean_{name}_after"] = float(l0.mean()), float(lu.mean())
        if lo:
            out[f"loss_mean_{name}_oracle"] = float(np.mean([o.mean() for o in lo]))
            c = closeness_np(lu, l0, lo, seed=seed)
            for k, v in c.items():
                out[f"E_{name}" if k == "E" else f"E_{name}_{k[2:]}" if k.startswith("E_") else f"{k}_{name}"] = v
            if name == "near":
                out["_near_losses"] = {"after": lu, "original": l0, "oracles": lo}
    return out


# ------------------------------------------------------------------ LLM track (amendment 2026-11-19)
#
# Same statistic, different loss: the per-example mean answer-token NLL replaces the per-timestep denoising loss
# (one "timestep" column).  Query sets on TOFU:
#   forget   forget-set QA pairs                       (members of the original, not of the oracle)
#   heldout  held-out retain slice, never fine-tuned   (non-members of every model; the MIA non-members)
#   near     retain-train QA pairs (other fictitious authors, same template)   (members of every model)
#   far      world_facts                               (general knowledge from pre-training; members of none)
# 'near' is the analogue of the diffusion near-class set: same domain, retained, so any change of the oracle there is
# the downstream fingerprint of removing the forget set.  'far' is pre-training knowledge rather than retained
# fine-tuning data (TOFU has no retained far-domain fine-tuning set); it measures collateral damage to general
# ability, so the SNR guard will usually leave E_far undefined (the forget set left no trace there), as intended.


def llm_query_indices(sizes: Dict[str, int], n_per_set: int, seed: int) -> Dict[str, List[int]]:
    """Fixed, seeded index subsets per query set; depends only on (sizes, n_per_set, seed) so every
    checkpoint of a regime is evaluated on identical queries."""
    rng = np.random.default_rng(seed)
    return {k: sorted(rng.choice(n, size=min(n_per_set, n), replace=False).tolist()) for k, n in sizes.items() if n > 0}


def fingerprint_suite_llm(models: Dict[str, object], oracle_models: List, query_examples: Dict[str, List[Dict]], nll_fn, seed: int = 0) -> Dict[str, float]:
    """``models`` = {'original': m0, 'after': mu}; ``query_examples`` = {set_name: [example, ...]};
    ``nll_fn(model, examples) -> (n,)`` per-example answer-token NLL.  Output keys match ``fingerprint_suite``."""
    out: Dict[str, float] = {}
    for name, ex in query_examples.items():
        if not ex:
            continue
        l0 = np.asarray(nll_fn(models["original"], ex), float)[:, None]
        lu = np.asarray(nll_fn(models["after"], ex), float)[:, None]
        lo = [np.asarray(nll_fn(o, ex), float)[:, None] for o in oracle_models]
        out[f"loss_mean_{name}_original"], out[f"loss_mean_{name}_after"] = float(l0.mean()), float(lu.mean())
        if lo:
            out[f"loss_mean_{name}_oracle"] = float(np.mean([o.mean() for o in lo]))
            c = closeness_np(lu, l0, lo, seed=seed)
            for k, v in c.items():
                out[f"E_{name}" if k == "E" else f"E_{name}_{k[2:]}" if k.startswith("E_") else f"{k}_{name}"] = v
    return out
