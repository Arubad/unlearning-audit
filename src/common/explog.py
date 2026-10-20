"""Experiment-log entries: one per run, expectation written before launch.

``start_entry`` appends the pre-launch part (config hash, commit, expectation)
to ``EXPERIMENT_LOG.md`` and writes ``results/<run>/run.json``.  ``finish_entry``
appends observed metrics and writes ``metrics.json``.  Entries are append-only.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
from typing import Any, Dict

from .config import config_hash, git_commit_short


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or "cpu"


def start_entry(resolved: Dict[str, Any], run_dir: str, expected: str, log_path: str = "EXPERIMENT_LOG.md") -> Dict[str, Any]:
    meta = {
        "experiment_id": resolved.get("experiment_id"),
        "cell": resolved.get("_cell", {}),
        "config_hash": config_hash(resolved),
        "git_commit": git_commit_short(),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": _gpu_name(),
        "expected": expected,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"meta": meta, "config": {k: v for k, v in resolved.items() if not k.startswith("_")}}, f, indent=2, default=str)
    if os.path.exists(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n## {meta['started'][:10]} — {meta['experiment_id']} — {meta['cell']}\n"
                f"- Config hash: `{meta['config_hash']}`\n- Git commit: `{meta['git_commit']}`\n"
                f"- Hardware: {meta['hardware']}\n- Expected: {expected}\n- Observed: _(running)_\n"
            )
    return meta


def finish_entry(meta: Dict[str, Any], metrics: Dict[str, Any], log_path: str = "EXPERIMENT_LOG.md", next_change: str = "") -> None:
    metrics = dict(metrics)
    metrics["_meta"] = meta
    metrics["_finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(meta["run_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    if os.path.exists(log_path):
        short = {k: v for k, v in metrics.items() if not k.startswith("_") and isinstance(v, (int, float, str))}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"- Observed ({meta['config_hash']}): `{json.dumps(short, default=float)}`\n")
            if next_change:
                f.write(f"- Next change: {next_change}\n")
