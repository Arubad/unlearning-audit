"""Membership-inference module: INTERFACE ONLY (to be implemented by the author).

The audit's MIA axis (loss-based, SecMI-style, offline LiRA with the oracle as
the OUT model, class-geometry MIA) is described in PREREGISTRATION.md and
README.md section 3.  The implementation was not generated with the rest of the
code drop; write it against the interface below so that ``report.py`` and the
hypothesis tests keep working.  Each function returns an AUC in [0, 1] computed
as forget (members) vs held-out non-members of the same class, with larger
scores meaning "member".  References to read first: Shokri et al. 2017,
Carlini et al. 2022 (LiRA), Duan et al. 2023 (SecMI), the ICML 2026
class-geometry MIA paper in READING.md.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..common.stats import bootstrap_ci  # noqa: F401  (available for CIs over examples)


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


def mia_suite(model, original_model, oracle_models, data, cfg: Dict, device) -> Dict[str, float]:
    """Return ``{"mia_loss_auc", "mia_secmi_auc", "mia_lira_auc", "mia_cmia_auc"}`` for one checkpoint.

    NOT IMPLEMENTED in this code drop.  Until implemented, the audit reports NaN for the
    MIA axis and the H1 test is marked inconclusive.
    """
    return {"mia_loss_auc": float("nan"), "mia_secmi_auc": float("nan"), "mia_lira_auc": float("nan"), "mia_cmia_auc": float("nan"), "mia_note": "not implemented: see src/audit/mia.py docstring"}
