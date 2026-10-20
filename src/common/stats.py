"""Statistics used by every figure: bootstrap CIs, paired comparisons, rank correlations.

All functions are numpy-only so they can be unit-tested without torch.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np


def bootstrap_ci(x: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI of ``stat`` over the rows of ``x``."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(x[i]) for i in idx])
    return {
        "mean": float(stat(x)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(n),
    }


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """CI of mean(a - b) with resampling over paired items (same prompt/problem)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired comparison needs equal-length arrays"
    return bootstrap_ci(a - b, np.mean, n_boot=n_boot, alpha=alpha, seed=seed)


def win_rate(scores_policy: Sequence[float], scores_ref: Sequence[float], tie: float = 0.5) -> float:
    a = np.asarray(scores_policy, dtype=float)
    b = np.asarray(scores_ref, dtype=float)
    return float(np.mean(np.where(a > b, 1.0, np.where(a < b, 0.0, tie))))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """Spearman correlation of x and y controlling for z (rank-based partial correlation)."""
    rx, ry, rz = (_rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))

    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    # if either variable is (numerically) a function of z there is nothing left to correlate
    if (ex**2).sum() < 1e-8 * (rx**2).sum() or (ey**2).sum() < 1e-8 * (ry**2).sum():
        return float("nan")
    denom = np.sqrt((ex**2).sum() * (ey**2).sum())
    return float((ex * ey).sum() / denom) if denom > 0 else float("nan")


def linear_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Least-squares y = a + b x with R^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"intercept": float(coef[0]), "slope": float(coef[1]), "r2": r2}


def iqr(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


def ci_overlaps(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
