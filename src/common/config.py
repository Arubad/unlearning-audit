"""YAML config loading, cell overrides and config hashing.

A *config* is the YAML file for a whole experiment; a *cell* is one point of its
`sweep` grid (for example ``gamma=0.3,beta=0.1,seed=0``).  ``resolve`` applies
the cell to the config and returns a flat dict that fully determines one run.
The sha256 of that resolved dict (first 12 hex chars) is the run's ``config_hash``
and is written into ``EXPERIMENT_LOG.md`` and ``results/<run>/metrics.json``.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from typing import Any, Dict, Iterator, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _parse_scalar(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except Exception:  # pragma: no cover
        return s


def parse_cell(cell: str | None) -> Dict[str, Any]:
    """``"gamma=0.3,beta=0.1,seed=0"`` -> ``{"gamma": 0.3, "beta": 0.1, "seed": 0}``."""
    if not cell:
        return {}
    out: Dict[str, Any] = {}
    for part in cell.split(","):
        if not part.strip():
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _parse_scalar(v.strip())
    return out


# Where each sweep key lives in the config tree.  Keys not listed here are set
# at the top level of the resolved config.
SWEEP_KEY_PATHS: Dict[str, str] = {
    "gamma": "annotator.gamma",
    "minority_mass": "annotator.mixture.minority_mass",
    "K": "annotator.mixture.K",
    "label_flip_eta": "annotator.label_flip_eta",
    "beta": "train.beta",
    "seed": "train.seed",
    "algorithm": "train.algorithm",
    "lora_r": "model.lora.r",
    "capacity": "model.capacity",
    # repo 2
    "weight_bits": "intervention.weight_bits",
    "kv_bits": "intervention.kv_bits",
    "kv_scheme": "intervention.kv_scheme",
    "sparsifier": "intervention.sparsifier",
    "keep_frac": "intervention.keep_frac",
    "layer": "intervention.layer",
    "head": "intervention.head",
    # repo 3
    "method": "unlearn.method",
    "forget_frac": "unlearn.forget_frac",
    "retain_available": "unlearn.retain_available",
    "track": "track",
}


def resolve(config: Dict[str, Any], cell: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Apply a cell to a copy of the config.  The ``sweep`` block is removed."""
    cfg = copy.deepcopy(config)
    cfg.pop("sweep", None)
    for k, v in (cell or {}).items():
        _set_dotted(cfg, SWEEP_KEY_PATHS.get(k, k), v)
    cfg["_cell"] = dict(cell or {})
    return cfg


def config_hash(resolved: Dict[str, Any]) -> str:
    payload = {k: v for k, v in resolved.items() if not k.startswith("_")}
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def iter_cells(config: Dict[str, Any], minimum: bool = False) -> Iterator[Dict[str, Any]]:
    """Expand ``sweep`` (or ``sweep_minimum`` when ``minimum``) into cells."""
    key = "sweep_minimum" if minimum and "sweep_minimum" in config else "sweep"
    grid = config.get(key, {}) or {}
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def cell_name(cell: Dict[str, Any]) -> str:
    if not cell:
        return "single"
    return "_".join(f"{k}{str(v).replace('.', 'p')}" for k, v in cell.items())


def run_dir(resolved: Dict[str, Any], root: str | None = None) -> str:
    root = root or resolved.get("logging", {}).get("output_dir", "results")
    exp = resolved.get("experiment_id", "exp")
    d = os.path.join(root, exp, cell_name(resolved.get("_cell", {})))
    os.makedirs(d, exist_ok=True)
    return d


def git_commit_short() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
