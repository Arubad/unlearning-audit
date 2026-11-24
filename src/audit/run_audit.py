"""Run the full audit for one diffusion cell: load the checkpoint pair, compute every axis, assemble the record.

Axes (amendment 2026-11-18 in PREREGISTRATION.md):
  forgetting   forget-class generation rate (conditional / unconditional) + retain-class sample accuracy
  mia          loss, SecMI t-error, offline LiRA (OUT = oracle retrains), class-geometry MIA
  fingerprint  oracle-normalised closeness E on forget / heldout / near / far query sets (H2, redesigned)
               + signed fingerprint vs the influence-function removal prediction
  attribution  TRAK-style scores (legacy H2 statistic, secondary)
  probe        logistic regression on null-conditioned mid-block features
  spectrum     per-layer SVD of Delta, ||Delta||, and relearn alignment cos(g_f(theta_u), Delta)  (H3)
  relearn      few-shot fine-tune on 5% of the forget set; steps to 90% of the pre-unlearning rate

Oracle handling: every oracle retrain on disk for this forget fraction is used as a reference; when the audited
checkpoint IS the oracle of seed s, oracle s is excluded so it is compared with independent replicates only.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from ..common.config import run_dir
from ..common.kaggle_env import torch_device
from ..diffusion.data import cifar_labels, load_cifar
from ..diffusion.ddpm import Diffusion
from ..diffusion.train import base_path, get_split, load_model, oracle_paths
from .attribution import attribution_diffusion
from .fingerprint import fingerprint_suite, query_sets, signed_fingerprint_np
from .mechanism import relearn_alignment
from .mia import mia_suite
from .probe import probe_diffusion
from .relearn import relearn_diffusion
from .report import assemble_record, save_record
from .spectrum import spectrum_of_checkpoints

DEFAULT_MIA = {"n_examples": 1000, "seed": 0, "timesteps": [50, 200, 400, 600, 800], "secmi_t": 200, "feature_timestep": 100, "n_centroid": 2000}
DEFAULT_FP = {"n_per_set": 256, "timesteps": [50, 200, 400, 600, 800]}


def _retain_sample_accuracy(model, diff, cfg, clf, seed: int, per_class: int = 50, detail: bool = False):
    """Fraction of retain-class conditional samples classified as their class.  ``detail`` also returns the A5-gated
    accuracy (correct AND not degenerate) and the degenerate fraction; the float return is the raw accuracy."""
    from ..diffusion.sample import class_rates, sample_class

    fc = int(cfg["data"]["forget_class"])
    s = cfg["sampling"]
    tot = {"raw": 0.0, "valid": 0.0, "degenerate": 0.0}
    k = 0
    for c in range(int(cfg["model"].get("n_classes", 10))):
        if c == fc:
            continue
        imgs = sample_class(model, diff, c, per_class, int(s["ddim_steps"]), float(s["guidance"]), seed + 31 * c)
        r = class_rates(imgs, clf, c)
        for kk in tot:
            tot[kk] += r[kk]
        k += 1
    out = {kk: v / max(k, 1) for kk, v in tot.items()}
    return out if detail else out["raw"]


def audit_diffusion_cell(cfg: Dict, oracle_relearn_steps: Optional[float] = None) -> Dict:
    import torch

    from ..diffusion.classifier import load_classifier
    from ..diffusion.sample import class_rates, classify, forgetting_metrics, sample_class

    u = cfg["unlearn"]
    seed = int(u["seed"])
    method = u["method"]
    frac = float(u["forget_frac"])
    device = torch_device()
    out_dir = run_dir(cfg)
    import time as _time

    _t = [_time.time()]

    def _tick(what):
        now = _time.time()
        print(f"[audit] {method} s={seed}: {what} took {now - _t[0]:.0f}s", flush=True)
        _t[0] = now

    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    base, _ = load_model(base_path(cfg), cfg, use_ema=True, device=device)
    unlearned, _ = load_model(os.path.join(out_dir, "unlearned.pt"), cfg, use_ema=False, device=device)

    opaths = oracle_paths(cfg, frac)
    ref = {s: p for s, p in opaths.items() if not (method == "oracle" and s == seed)}
    oracles = [load_model(p, cfg, use_ema=True, device=device)[0] for p in ref.values()]
    print(f"[audit] {method} f={frac:g} s={seed}: {len(oracles)} reference oracle(s) {list(ref)}")

    ds = load_cifar(cfg["data"].get("root", "data"))
    test = load_cifar(cfg["data"].get("root", "data"), train=False)
    labels, test_labels = cifar_labels(ds), cifar_labels(test)
    split = get_split(cfg, frac, seed, labels)
    fc = int(cfg["data"]["forget_class"])
    clf = load_classifier(device)
    n_eval = int(cfg["sampling"].get("n_eval", 500))

    # ---- forgetting + utility
    forgetting = forgetting_metrics(unlearned, cfg, seed, n_eval)
    pre = forgetting_metrics(base, cfg, seed, n_eval)["forget_class_rate_conditional_valid"]
    rs = _retain_sample_accuracy(unlearned, diff, cfg, clf, seed, detail=True)
    forgetting["retain_sample_accuracy"], forgetting["retain_sample_accuracy_valid"], forgetting["retain_degenerate_rate"] = rs["raw"], rs["valid"], rs["degenerate"]
    forgetting["regime"] = "class" if split["n_forget"] >= int((labels == fc).sum()) else "subset"

    # ---- MIA
    cfg.setdefault("audit", {}).setdefault("mia", {})
    cfg["audit"]["mia"] = {**DEFAULT_MIA, **(cfg["audit"]["mia"] or {})}
    mia = mia_suite(unlearned, base, oracles, (ds, split), cfg, device)

    # ---- fingerprint (H2)
    fpc = {**DEFAULT_FP, **cfg["audit"].get("fingerprint", {})}
    qsets = query_sets(labels, test_labels, split, fc, int(fpc["n_per_set"]), seed=12345)  # fixed across cells
    fp = fingerprint_suite({"original": base, "after": unlearned}, oracles, diff, ds, test, qsets, [int(t) for t in fpc["timesteps"]], device, seed)
    near = fp.pop("_near_losses", None)

    _tick("forgetting + mia + fingerprint")

    # ---- attribution (legacy H2 + removal prediction).  Secondary readout: a failure here is recorded, not fatal,
    # so the primary axes of the cell are never lost to it.
    if not bool(cfg["audit"].get("attribution", {}).get("enabled", True)):
        # amendment A3 (2026-11-19): TRAK disabled after run v4 showed corr(prediction, oracle change) ~ 0 at 40 of 52
        # min per cell; the signed oracle-change correlation below needs no attribution and is kept
        attribution, pred = {"attr_disabled": True}, None
    else:
        try:
            attribution = attribution_diffusion(base, unlearned, oracles[0] if oracles else None, diff, ds, split, cfg, device, seed, query_indices=qsets["near"]["indices"])
            pred = np.asarray(attribution.pop("pred_removal_effect"))
            attribution.pop("pred_query_indices", None)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            torch.cuda.empty_cache()
            attribution, pred = {"attr_error": f"{type(e).__name__}: {str(e)[:300]}"}, None
    _tick("attribution")
    signed = {}
    if near is not None and pred is not None:
        signed["fp_corr_pred_after"] = signed_fingerprint_np(near["after"], near["original"], pred)
        signed["fp_corr_pred_oracle"] = float(np.nanmean([signed_fingerprint_np(o, near["original"], pred) for o in near["oracles"]]))
    if near is not None:
        omean = np.mean([o.mean(1) for o in near["oracles"]], 0) - near["original"].mean(1)
        signed["fp_corr_oracle_change"] = signed_fingerprint_np(near["after"], near["original"], omean)

    # ---- probe, spectrum, alignment
    probe = probe_diffusion(unlearned, diff, ds, labels, fc, cfg, device, seed)
    spectrum = spectrum_of_checkpoints(unlearned.state_dict(), base.state_dict(), int(cfg["audit"]["spectrum"]["top_k"]))
    align = relearn_alignment(unlearned, base.state_dict(), diff, ds, split["forget"], device, seed)
    _tick("probe + spectrum + alignment")

    # ---- relearning
    n_rel = int(cfg["sampling"].get("n_relearn_eval", 128))

    def metric_fn(m):
        imgs = sample_class(m, diff, fc, n_rel, int(cfg["sampling"]["ddim_steps"]), float(cfg["sampling"]["guidance"]), seed)
        return class_rates(imgs, clf, fc)["valid"]  # A5: noise labelled "cat" is not recovery

    relearn = relearn_diffusion(unlearned, cfg, split, pre, seed, metric_fn)
    _tick("relearn")

    record = assemble_record(cfg.get("_cell", {}) | {"track": "diffusion", "method": method, "forget_frac": frac, "seed": seed}, {**forgetting, "pre_unlearn_forget_rate": pre}, {**mia, **fp, **signed, **align}, attribution, relearn, probe, spectrum, oracle_relearn_steps)
    record["n_reference_oracles"] = len(oracles)
    record["relearn_curve_steps"], record["relearn_curve_values"] = relearn["relearn_curve_steps"], relearn["relearn_curve_values"]
    record["per_layer_spectrum"] = {k: v for k, v in spectrum.get("per_layer", {}).items()}
    save_record(record, out_dir)
    if near is not None:
        np.savez_compressed(os.path.join(out_dir, "near_losses.npz"), after=near["after"], original=near["original"], oracles=np.stack(near["oracles"]) if near["oracles"] else np.zeros(0), pred=pred)
    print(json.dumps({k: v for k, v in record.items() if isinstance(v, (int, float, str))}, default=float))
    return record


def truncation_intervention(cfg: Dict, ranks) -> Dict:
    """H3 intervention: replace the cell's Delta by its per-layer rank-k truncation at matched ||Delta||_F, then
    measure forgetting, retain accuracy and relearning time for each k (plus the untruncated Delta)."""
    from ..diffusion.classifier import load_classifier
    from ..diffusion.sample import class_rates, sample_class
    from .mechanism import apply_delta, truncated_delta_np

    u = cfg["unlearn"]
    seed, frac = int(u["seed"]), float(u["forget_frac"])
    device = torch_device()
    out_dir = run_dir(cfg)
    diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
    base, _ = load_model(base_path(cfg), cfg, use_ema=True, device=device)
    unlearned, _ = load_model(os.path.join(out_dir, "unlearned.pt"), cfg, use_ema=False, device=device)
    sd0 = {k: v.detach().clone() for k, v in base.state_dict().items()}
    sdu = unlearned.state_dict()
    deltas = {k: (sdu[k].float().cpu().numpy() - sd0[k].float().cpu().numpy()) for k in sd0 if sd0[k].dtype.is_floating_point}
    labels = cifar_labels(load_cifar(cfg["data"].get("root", "data")))
    split = get_split(cfg, frac, seed, labels)
    fc = int(cfg["data"]["forget_class"])
    clf = load_classifier(device)
    s = cfg["sampling"]
    n_rel = int(s.get("n_relearn_eval", 128))

    def rates(m, n):
        return class_rates(sample_class(m, diff, fc, n, int(s["ddim_steps"]), float(s["guidance"]), seed), clf, fc)

    def rate(m, n):  # A5: the decision metric is the gated rate
        return rates(m, n)["valid"]

    pre = rate(base, int(s.get("n_eval", 500)))
    rows = []
    for k in list(ranks) + [None]:
        dk = deltas if k is None else truncated_delta_np(deltas, int(k))
        m = apply_delta(build_like(base), sd0, dk).to(device)
        fr = rates(m, int(s.get("n_eval", 500)))
        rs = _retain_sample_accuracy(m, diff, cfg, clf, seed, detail=True)
        row = {"rank": "full" if k is None else int(k), "delta_norm": float(np.sqrt(sum(float((v.astype(float) ** 2).sum()) for v in dk.values()))),
               "forget_rate": fr["valid"], "forget_rate_raw": fr["raw"], "degenerate_rate": fr["degenerate"],
               "retain_sample_accuracy": rs["valid"], "retain_sample_accuracy_raw": rs["raw"]}
        rl = relearn_diffusion(m, cfg, split, pre, seed, lambda mm: rate(mm, n_rel))
        row["relearn_steps_to_90"] = rl["relearn_steps_to_90"]
        row["relearn_curve_values"] = rl["relearn_curve_values"]
        print(json.dumps({kk: vv for kk, vv in row.items() if kk != "relearn_curve_values"}), flush=True)
        rows.append(row)
    res = {"cell": cfg.get("_cell", {}), "pre_unlearn_forget_rate": pre, "rows": rows}
    with open(os.path.join(out_dir, "truncation.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


def build_like(model):
    import copy

    return copy.deepcopy(model)
