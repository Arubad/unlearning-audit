"""Command-line entry point.  Notebooks are thin wrappers around these commands.

Diffusion track:
  python -m src.run train_base       --config configs/diffusion_base.yaml
  python -m src.run train_classifier --config configs/diffusion_base.yaml
  python -m src.run train_oracle     --config configs/diffusion_base.yaml --cell seed=0,forget_frac=0.05
  python -m src.run sweep            --config configs/diffusion_unlearn_sweep.yaml [--minimum] [--skip-existing]
  (sweep = for each cell: unlearn -> audit)

LLM track:
  python -m src.run llm_finetune     --config configs/llm_base.yaml
  python -m src.run llm_oracle       --config configs/llm_base.yaml --cell forget_frac=0.05,seed=0
  python -m src.run llm_sweep        --config configs/llm_unlearn_sweep.yaml [--minimum]

  python -m src.run analyze [--results results]
  python -m src.run smoke
"""
from __future__ import annotations

import argparse
import os

from .common.config import cell_name, config_hash, iter_cells, load_yaml, parse_cell, resolve, run_dir
from .common.explog import finish_entry, start_entry
from .common.kaggle_env import hf_cache_dir, import_previous_results, results_root


def _prep(args):
    cfg = resolve(load_yaml(args.config), parse_cell(getattr(args, "cell", "")))
    cfg["logging"]["output_dir"] = results_root(cfg.get("logging", {}).get("output_dir", "results"))
    return cfg


def cmd_train_base(args):
    from .diffusion.train import train_ddpm

    print(train_ddpm(load_yaml(args.config), retain_only=False, seed=0))


def cmd_train_classifier(args):
    from .diffusion.classifier import train_classifier

    print(train_classifier(load_yaml(args.config)))


def cmd_train_oracle(args):
    from .diffusion.train import train_ddpm

    cfg = _prep(args)
    print(train_ddpm(cfg, retain_only=True, seed=int(cfg["unlearn"]["seed"])))


def cmd_sweep(args):
    from .audit.run_audit import audit_diffusion_cell
    from .diffusion.unlearn import unlearn_cell

    base = load_yaml(args.config)
    for cell in iter_cells(base, minimum=args.minimum):
        cfg = resolve(base, cell)
        cfg["logging"]["output_dir"] = results_root(cfg.get("logging", {}).get("output_dir", "results"))
        d = run_dir(cfg)
        if args.skip_existing and os.path.exists(os.path.join(d, "audit_record.json")):
            print("skip", cell)
            continue
        print(f"\n===== {cell_name(cell)}  {config_hash(cfg)} =====")
        meta = start_entry(cfg, d, "unlearn + audit")
        m = unlearn_cell(cfg)
        if cfg["unlearn"]["method"] != "oracle":
            rec = audit_diffusion_cell(cfg)
            m["attr_drop"] = rec.get("attr_drop")
            m["relearn_steps_to_90"] = rec.get("relearn_steps_to_90")
        finish_entry(meta, m)


def cmd_llm_finetune(args):
    from .llm.finetune import finetune

    cfg = _prep(args)
    print(finetune(cfg, retain_only=False, full_ft=bool(cfg["unlearn"].get("full_ft", False)), seed=int(cfg["unlearn"].get("seed", 0))))


def cmd_llm_oracle(args):
    from .llm.finetune import finetune

    cfg = _prep(args)
    print(finetune(cfg, retain_only=True, full_ft=bool(cfg["unlearn"].get("full_ft", False)), seed=int(cfg["unlearn"].get("seed", 0))))


def cmd_llm_sweep(args):
    from .llm.unlearn import unlearn_llm

    base = load_yaml(args.config)
    for cell in iter_cells(base, minimum=args.minimum):
        cfg = resolve(base, cell)
        cfg["logging"]["output_dir"] = results_root(cfg.get("logging", {}).get("output_dir", "results"))
        d = run_dir(cfg)
        meta = start_entry(cfg, d, "llm unlearn + audit")
        m = unlearn_llm(cfg)
        finish_entry(meta, m)
    print("LLM audit (MIA/attribution/probe) reuses src/audit on the merged weights; see notebook 06.")


def cmd_analyze(args):
    from .analysis.figures import analyze

    analyze(args.results or results_root())


def cmd_smoke(args):
    from .smoke import run_smoke

    run_smoke()


def main(argv=None):
    hf_cache_dir()
    import_previous_results()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, needs_cell in (
        ("train_base", cmd_train_base, False), ("train_classifier", cmd_train_classifier, False), ("train_oracle", cmd_train_oracle, True),
        ("llm_finetune", cmd_llm_finetune, True), ("llm_oracle", cmd_llm_oracle, True),
    ):
        s = sub.add_parser(name); s.add_argument("--config", required=True)
        if needs_cell:
            s.add_argument("--cell", default="")
        s.set_defaults(fn=fn)
    for name, fn in (("sweep", cmd_sweep), ("llm_sweep", cmd_llm_sweep)):
        s = sub.add_parser(name); s.add_argument("--config", required=True); s.add_argument("--minimum", action="store_true"); s.add_argument("--skip-existing", action="store_true"); s.set_defaults(fn=fn)
    s = sub.add_parser("analyze"); s.add_argument("--results", default=None); s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("smoke"); s.set_defaults(fn=cmd_smoke)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
