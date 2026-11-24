"""Sequential, resumable driver for the diffusion track: ``python -m src.pipeline <stage> [<stage> ...]``.

Every stage is idempotent: it checks for its outputs and skips (or resumes) when they exist, so after a reclaimed
Colab session the same command line continues where it stopped.

  classifier     CIFAR-10 classifier for the generation-rate metric
  base           D1 base DDPM (all classes)
  oracle_c       D2 class-regime oracles  (forget_frac 0.10, seeds 0 and 1)
  sweep_c        D3-D5 class regime: every method x seeds 0,1, unlearn + full audit
  truncate_c     H3 intervention on the gradient-ascent class cells
  oracle_s       D2 subset-regime oracles (forget_frac 0.05, seeds 0 and 1)
  sweep_s        D3-D5 subset regime
  analyze        hypothesis report + figures
  wait_oracle_c  barrier: block until every oracle of the class regime is fully trained (parallel seed processes)
  wait_oracle_s  same for the subset regime
  oppoint        A3 operating-point trajectories (grid split across the parallel processes)
  wait_oppoint   barrier: all trajectories done, then apply the pre-specified selection rule
  sweep_op       A3 audit of the selected operating points (+ embedding_reset), this process's seeds, TRAK off

Parallel use (Kaggle 2x T4): run one process per GPU with ``UA_SEEDS=0`` / ``UA_SEEDS=1``; the audit of each cell
needs *both* oracles of its forget fraction (noise floor), hence the wait stages.  ``UA_DEADLINE`` (unix seconds)
makes long stages save and stop; the process then exits 0 so a hosted session keeps its output.
"""
from __future__ import annotations

import json
import os
import sys
import time

from .common.config import load_yaml, resolve
from .common.kaggle_env import results_root

BASE_CFG = "configs/diffusion_base.yaml"
SWEEP_CFG = "configs/diffusion_unlearn_sweep.yaml"
SEEDS = tuple(int(s) for s in os.environ.get("UA_SEEDS", "0,1").split(","))
ALL_SEEDS = (0, 1)  # oracles every audit needs, whichever process trains them


def _base_cfg(cell=None):
    cfg = resolve(load_yaml(BASE_CFG), cell or {})
    cfg["logging"]["output_dir"] = results_root()
    return cfg


def stage_classifier():
    from .diffusion.classifier import classifier_path, train_classifier

    if os.path.exists(classifier_path()):
        print("classifier exists:", classifier_path())
        return
    print(train_classifier(_base_cfg()))


def stage_base():
    from .diffusion.train import train_ddpm

    print(train_ddpm(_base_cfg(), retain_only=False, seed=0))


def _oracles(frac):
    from .diffusion.train import train_ddpm

    for s in SEEDS:
        print(train_ddpm(_base_cfg({"seed": s, "forget_frac": frac}), retain_only=True, seed=s))


def stage_oracle_confirm():
    """C1 (2026-11-21): train the oracle jobs in UA_ORACLE_JOBS ("class:seed,class:seed,..."), split across the
    parallel processes by index.  Each job is an exact retrain on CIFAR-10 minus that whole class (frac 0.10);
    training resumes from its own checkpoint, so a run cut by the deadline continues next cycle."""
    from .diffusion.train import train_ddpm

    jobs = [tuple(int(v) for v in j.split(":")) for j in os.environ.get("UA_ORACLE_JOBS", "").split(",") if j.strip()]
    pi = _proc_index()
    for i, (fc, s) in enumerate(jobs):
        if pi is not None and i % len(ALL_SEEDS) != pi:
            continue
        print(f"[C1] oracle class {fc} seed {s}", flush=True)
        print(train_ddpm(_base_cfg({"seed": s, "forget_frac": 0.10, "forget_class": fc}), retain_only=True, seed=s), flush=True)


def _sweep(frac, methods=None):
    from argparse import Namespace

    from .run import cmd_sweep

    base = load_yaml(SWEEP_CFG)
    base["sweep"] = {"method": methods or base["sweep"]["method"], "forget_frac": [frac], "seed": list(SEEDS)}
    tmp = os.path.join(results_root(), f"_sweep_{frac:g}_s{'-'.join(map(str, SEEDS))}.yaml")
    import yaml

    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w") as f:
        yaml.safe_dump(base, f)
    cmd_sweep(Namespace(config=tmp, minimum=False, skip_existing=True))


def _wait_oracles(frac, poll_s=180):
    """Block until all ALL_SEEDS oracles for ``frac`` are trained to the configured step count."""
    import torch

    from .common.deadline import check
    from .diffusion.train import ckpt_path

    steps = int(_base_cfg()["train"]["steps"])
    while True:
        missing = []
        for s in ALL_SEEDS:
            p = ckpt_path(_base_cfg({"seed": s, "forget_frac": frac}), "retain", s, frac)
            st = int(torch.load(p, map_location="cpu", weights_only=False).get("step", 0)) if os.path.exists(p) else 0
            if st < steps:
                missing.append((s, st))
        if not missing:
            print(f"oracles for frac {frac:g} complete", flush=True)
            return
        print(f"waiting for oracles {missing} (frac {frac:g})", flush=True)
        check(poll_s, f"oracles {missing}")
        time.sleep(poll_s)


def stage_truncate_c():
    from .audit.run_audit import truncation_intervention
    from .common.config import run_dir

    base = load_yaml(SWEEP_CFG)
    from .common.deadline import check

    for s in SEEDS:
        cfg = resolve(base, {"method": "gradient_ascent", "forget_frac": 0.10, "seed": s})
        cfg["logging"]["output_dir"] = results_root()
        if os.path.exists(os.path.join(run_dir(cfg), "truncation.json")):
            print("truncation exists for seed", s)
            continue
        check(float(os.environ.get("UA_TRUNC_MARGIN_S", 5400)), f"truncation seed {s}")
        truncation_intervention(cfg, [1, 4, 16, 64])


def stage_analyze():
    from .analysis.figures import analyze

    analyze(results_root())


def _proc_index():
    return SEEDS[0] if len(SEEDS) == 1 else None


def stage_oppoint():
    from .diffusion.oppoint import run_trajectories

    run_trajectories(_proc_index(), len(ALL_SEEDS))


def stage_wait_oppoint():
    from .diffusion.oppoint import wait_and_select

    wait_and_select()


def stage_sweep_op():
    from .diffusion.oppoint import audit_selected

    audit_selected(SEEDS)


STAGES = {
    "classifier": stage_classifier,
    "base": stage_base,
    "oracle_c": lambda: _oracles(0.10),
    "oracle_confirm": stage_oracle_confirm,
    "wait_oracle_c": lambda: _wait_oracles(0.10),
    "wait_oracle_s": lambda: _wait_oracles(0.05),
    "sweep_c": lambda: _sweep(0.10),
    "truncate_c": stage_truncate_c,
    "oracle_s": lambda: _oracles(0.05),
    "sweep_s": lambda: _sweep(0.05),
    "analyze": stage_analyze,
    "oppoint": stage_oppoint,
    "wait_oppoint": stage_wait_oppoint,
    "sweep_op": stage_sweep_op,
}


def main(argv=None):
    from .common.deadline import DeadlineReached

    names = (argv if argv is not None else sys.argv[1:]) or list(STAGES)
    failed = []
    for n in names:
        t0 = time.time()
        print(f"\n########## stage {n} (seeds {SEEDS}) ##########", flush=True)
        try:
            STAGES[n]()
        except DeadlineReached as e:
            print(f"########## DEADLINE in stage {n}: {e} -- stopping cleanly ##########", flush=True)
            with open(os.path.join(results_root(), "pipeline_progress.jsonl"), "a") as f:
                f.write(json.dumps({"stage": n, "seeds": list(SEEDS), "deadline": True, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            return 1 if failed else 0
        except Exception as e:  # noqa: BLE001
            # 2026-11-19: one crashing stage used to end the process, wasting the rest of a quota-metered session
            # (run v4: truncate_c crashed and the subset stages never ran).  Record it, run the remaining stages,
            # and exit non-zero at the end so the driver still stops for a fix.
            import traceback

            traceback.print_exc()
            failed.append(n)
            print(f"########## stage {n} FAILED ({type(e).__name__}: {e}); continuing ##########", flush=True)
            with open(os.path.join(results_root(), "pipeline_progress.jsonl"), "a") as f:
                f.write(json.dumps({"stage": n, "seeds": list(SEEDS), "failed": f"{type(e).__name__}: {str(e)[:300]}", "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            continue
        print(f"########## stage {n} done in {(time.time() - t0) / 3600:.2f} h ##########", flush=True)
        with open(os.path.join(results_root(), "pipeline_progress.jsonl"), "a") as f:
            f.write(json.dumps({"stage": n, "seeds": list(SEEDS), "hours": (time.time() - t0) / 3600, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    if failed:
        print(f"########## failed stages: {failed} ##########", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
