"""Amendment A3 (2026-11-19, post-hoc / exploratory): compare unlearning methods at a common operating point.

Run v4 showed that at the matched budget (2,000 steps, lr 3e-5) gradient ascent and variational EWC drive retain
generation to chance while gradient difference and label annealing barely forget, so no method sits where the oracle
does (cat rate <= 0.10, retain accuracy ~0.95).  Rung-by-rung claims about *methods* need a common operating point.

  trajectories  one unlearning run per grid entry (seed 0); cat rate and retain accuracy every ``eval_every`` steps
                with fixed sampling noise across steps (common random numbers), no checkpoint written
  select        the pre-specified rule in ``configs/diffusion_oppoint.yaml`` picks one (hyperparameters, steps) per
                method; written to ``results/D6_oppoint/selected.json``
  audit         full audit (TRAK off) of each selected point and of the zero-step ``extra_methods``, seeds 0 and 1

Grid entries are split across parallel processes by index, so two GPUs trace half the grid each.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from ..common.config import load_yaml, resolve
from ..common.deadline import check
from ..common.kaggle_env import results_root

OP_CFG = "configs/diffusion_oppoint.yaml"
SWEEP_CFG = "configs/diffusion_unlearn_sweep.yaml"


def _op() -> Dict:
    return load_yaml(OP_CFG)


def _dir() -> str:
    d = os.path.join(results_root(), "D6_oppoint")
    os.makedirs(d, exist_ok=True)
    return d


def _fmt(v) -> str:
    return f"{v:g}".replace(".", "p").replace("-", "m")


def traj_name(entry: Dict) -> str:
    s = f"{entry['method']}_lr{_fmt(float(entry['lr']))}"
    if "ewc_weight" in entry:
        s += f"_ewc{_fmt(float(entry['ewc_weight']))}"
    return s


def _cell(entry: Dict, seed: int, frac: float) -> Dict:
    c = {"method": entry["method"], "forget_frac": frac, "seed": seed}
    for k in ("lr", "steps", "ewc_weight"):
        if k in entry:
            c[k] = entry[k]
    return c


def _cfg(cell: Dict, experiment_id: str) -> Dict:
    base = load_yaml(SWEEP_CFG)
    cfg = resolve(base, cell)
    cfg["experiment_id"] = experiment_id
    cfg["logging"]["output_dir"] = results_root()
    cfg["audit"].setdefault("attribution", {})["enabled"] = False
    return cfg


def _atomic_json(obj, path: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


# ------------------------------------------------------------------ stage 1: trajectories

def run_trajectories(proc_index: Optional[int] = None, n_procs: int = 2) -> None:
    from ..audit.run_audit import _retain_sample_accuracy
    from ..diffusion.classifier import load_classifier
    from ..diffusion.ddpm import Diffusion
    from ..diffusion.sample import class_rates, sample_class
    from ..diffusion.unlearn import unlearn_cell
    from ..common.kaggle_env import torch_device

    op = _op()
    tr = op["trajectory"]
    frac = float(op["forget_frac"])
    device = torch_device()
    for i, entry in enumerate(op["grid"]):
        if proc_index is not None and i % n_procs != proc_index:
            continue
        path = os.path.join(_dir(), f"traj_{traj_name(entry)}.json")
        if os.path.exists(path):
            print("trajectory exists:", path, flush=True)
            continue
        check(float(os.environ.get("UA_TRAJ_MARGIN_S", 1500)), f"trajectory {traj_name(entry)}")
        cfg = _cfg(_cell(entry, int(tr.get("seed", 0)), frac), "D6_oppoint_traj")
        diff = Diffusion(int(cfg["model"].get("timesteps", 1000)), device=device)
        clf = load_classifier(device)
        fc = int(cfg["data"]["forget_class"])
        s = cfg["sampling"]
        points: List[Dict] = []

        def cb(model, step, cfg=cfg, diff=diff, clf=clf, fc=fc, s=s, points=points):
            imgs = sample_class(model, diff, fc, int(tr["n_forget_samples"]), int(s["ddim_steps"]), float(s["guidance"]), seed=4242)
            fr = class_rates(imgs, clf, fc)
            rs = _retain_sample_accuracy(model, diff, cfg, clf, 4243, per_class=int(tr["retain_per_class"]), detail=True)
            rate, racc = fr["valid"], rs["valid"]  # A5: selection uses the gated rates; raw ones kept for comparison
            points.append({"step": step, "forget_rate": rate, "retain_acc": racc, "forget_rate_raw": fr["raw"], "retain_acc_raw": rs["raw"], "degenerate_rate": fr["degenerate"]})
            print(f"[oppoint] {traj_name(entry)} step {step}: cat rate {rate:.3f}, retain acc {racc:.3f}", flush=True)

        out = unlearn_cell(cfg, callback=cb, callback_every=int(tr["eval_every"]), save=False)
        _atomic_json({"entry": entry, "points": points, "final_delta_norm": out.get("delta_norm_total")}, path)


# ------------------------------------------------------------------ stage 2: selection (pure, unit-tested)

def select_point(points: List[Dict], max_forget: float, min_retain: float) -> Optional[Dict]:
    """The pre-specified rule on one method's pooled points (each carries its grid ``entry``)."""
    cand = [p for p in points if p["step"] >= 1]
    hit = [p for p in cand if p["forget_rate"] <= max_forget and p["retain_acc"] >= min_retain]
    if hit:
        best = min(hit, key=lambda p: (p["step"], -p["retain_acc"], float(p["entry"]["lr"])))
        return {**best, "reached_target": True}
    keep = [p for p in cand if p["retain_acc"] >= min_retain]
    if keep:
        best = min(keep, key=lambda p: (p["forget_rate"], p["step"], float(p["entry"]["lr"])))
        return {**best, "reached_target": False}
    return None


def wait_and_select(poll_s: int = 120) -> Dict:
    import time

    op = _op()
    names = [traj_name(e) for e in op["grid"]]
    while True:
        missing = [n for n in names if not os.path.exists(os.path.join(_dir(), f"traj_{n}.json"))]
        if not missing:
            break
        # a crashed trajectory stage in the other process would otherwise hold this barrier until the deadline
        prog = os.path.join(results_root(), "pipeline_progress.jsonl")
        if os.path.exists(prog) and any(r.get("stage") == "oppoint" and "failed" in r for r in (json.loads(l) for l in open(prog) if l.strip())):
            raise RuntimeError(f"an oppoint trajectory stage failed; missing {missing}")
        print(f"waiting for trajectories {missing}", flush=True)
        check(poll_s, f"trajectories {missing}")
        time.sleep(poll_s)
    by_method: Dict[str, List[Dict]] = {}
    for n in names:
        t = json.load(open(os.path.join(_dir(), f"traj_{n}.json")))
        for p in t["points"]:
            by_method.setdefault(t["entry"]["method"], []).append({**p, "entry": t["entry"]})
    sel = op["selection"]
    out = {m: select_point(pts, float(sel["max_forget_rate"]), float(sel["min_retain_acc"])) for m, pts in by_method.items()}
    _atomic_json(out, os.path.join(_dir(), "selected.json"))
    print("[oppoint] selected:", json.dumps(out, default=float), flush=True)
    return out


# ------------------------------------------------------------------ stage 3: audit the selected points

def _oracle_relearn_steps(seed: int, frac: float) -> Optional[float]:
    from ..common.config import cell_name

    p = os.path.join(results_root(), "D3_ddpm_unlearn", cell_name({"method": "oracle", "forget_frac": frac, "seed": seed}), "audit_record.json")
    return json.load(open(p)).get("relearn_steps_to_90") if os.path.exists(p) else None


def audit_selected(seeds) -> None:
    from ..audit.run_audit import audit_diffusion_cell
    from ..common.config import run_dir
    from ..common.explog import finish_entry, start_entry
    from ..diffusion.unlearn import unlearn_cell

    op = _op()
    frac = float(op["forget_frac"])
    selected = json.load(open(os.path.join(_dir(), "selected.json")))
    todo = []
    for m, p in selected.items():
        if p is None:
            print(f"[oppoint] {m}: no utility-preserving operating point; not audited", flush=True)
            continue
        e = dict(p["entry"])
        e["steps"] = int(p["step"])
        todo.append((e, bool(p["reached_target"])))
    todo += [({"method": m}, True) for m in op.get("extra_methods", [])]
    for e, reached in todo:
        for s in seeds:
            cell = {"method": e["method"], "forget_frac": frac, "seed": int(s)}
            for k in ("lr", "steps", "ewc_weight"):
                if k in e:
                    cell[k] = e[k]
            cfg = _cfg(cell, "D6_oppoint_audit")
            d = run_dir(cfg)
            if os.path.exists(os.path.join(d, "audit_record.json")):
                print("skip", cell, flush=True)
                continue
            check(float(os.environ.get("UA_OPCELL_MARGIN_S", 1800)), f"oppoint cell {cell}")
            meta = start_entry(cfg, d, "operating-point unlearn + audit (A3)")
            m = unlearn_cell(cfg)
            rec = audit_diffusion_cell(cfg, oracle_relearn_steps=_oracle_relearn_steps(int(s), frac))
            rec["oppoint_reached_target"] = reached
            with open(os.path.join(d, "audit_record.json"), "w") as f:
                json.dump(rec, f, indent=2, default=float)
            for k in ("forget_class_rate_conditional", "retain_sample_accuracy", "mia_lira_auc", "probe_accuracy", "relearn_steps_to_90", "top5_energy_wmean", "relearn_alignment"):
                m[k] = rec.get(k)
            finish_entry(meta, m)
