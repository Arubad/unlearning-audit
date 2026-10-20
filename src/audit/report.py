"""Assemble one audit record for a checkpoint pair and run the do-nothing control checks.

The record is the row the hypothesis tests consume.  The control assertions
encode the pre-registered expected-to-fail check: the unmodified checkpoint must
look un-forgotten on every axis.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from .spectrum import spectrum_of_checkpoints


def assemble_record(cell: Dict, forgetting: Dict, mia: Dict, attribution: Dict, relearn: Dict, probe: Dict, spectrum: Dict, oracle_relearn_steps: Optional[float] = None) -> Dict:
    rec = {"cell": cell}
    rec.update(forgetting)
    rec.update({k: v for k, v in mia.items() if not isinstance(v, (list, dict))})
    rec.update({k: v for k, v in attribution.items() if not k.endswith("per_example")})
    rec.update({k: v for k, v in relearn.items() if not k.startswith("relearn_curve")})
    rec.update(probe)
    rec.update({"effective_rank_wmean": spectrum.get("effective_rank_wmean"), "top5_energy_wmean": spectrum.get("top5_energy_wmean"), "delta_norm_total": spectrum.get("delta_norm_total")})
    if oracle_relearn_steps is not None and relearn.get("relearn_steps_to_90") not in (None, float("inf")):
        rec["relearn_ratio_to_oracle"] = relearn["relearn_steps_to_90"] / max(oracle_relearn_steps, 1e-9)
    return rec


def control_assertions(record: Dict) -> Dict:
    """For the do_nothing cell: everything should say 'not forgotten'."""
    checks = {
        "mia_above_chance": np.isnan(record.get("mia_loss_auc", np.nan)) or record.get("mia_loss_auc", 0) > 0.55,
        "probe_full": record.get("probe_accuracy", 1.0) >= 0.8,
        "no_relearn": record.get("relearn_steps_to_90", 0) == 0,
        "no_attr_drop": abs(record.get("attr_drop", 0.0)) < 0.05,
        "zero_delta": record.get("delta_norm_total", 0.0) < 1e-6,
    }
    checks["verdict"] = "passed" if all(v for k, v in checks.items() if k != "verdict") else "FAILED: harness reports forgetting for an unmodified checkpoint"
    return checks


def save_record(record: Dict, out_dir: str) -> None:
    with open(os.path.join(out_dir, "audit_record.json"), "w") as f:
        json.dump(record, f, indent=2, default=float)
