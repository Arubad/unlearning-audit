"""numpy/scipy/sklearn-only tests for the audit cores (no torch).  Run: python3 -I tests/test_audit_cores.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.diffusion.ddpm import linear_schedule_np, ddim_step_np
from src.audit.mia import auc_from_scores
from src.audit.spectrum import matrix_spectrum_np, aggregate_spectra
from src.audit.attribution import project_np, trak_scores_np, attribution_summary
from src.audit.relearn import steps_to_fraction
from src.audit.probe import probe_accuracy_np
from src.audit.report import control_assertions
from src.diffusion.data import cifar_split_indices
from src.common.stats import partial_spearman, spearman


def test_schedule():
    s = linear_schedule_np(1000)
    assert s["alphas_cumprod"][0] < 1 and s["alphas_cumprod"][-1] > 0
    assert np.all(np.diff(s["alphas_cumprod"]) < 0)  # strictly decreasing
    assert abs(s["sqrt_abar"][0] ** 2 + s["sqrt_1m_abar"][0] ** 2 - 1) < 1e-9


def test_ddim_recovers_x0():
    # if eps is exact, one DDIM step toward abar_prev=1 returns x0
    x0 = np.array([0.3, -0.7, 1.0])
    abar_t = 0.4
    noise = np.array([1.0, -1.0, 0.5])
    x_t = np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * noise
    x_prev = ddim_step_np(x_t, noise, abar_t, 1.0)
    assert np.allclose(x_prev, x0, atol=1e-9)


def test_auc():
    assert abs(auc_from_scores([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9
    assert abs(auc_from_scores([0, 1, 2], [0, 1, 2]) - 0.5) < 1e-9
    assert abs(auc_from_scores([0, 1], [2, 3]) - 0.0) < 1e-9


def test_spectrum_rank1_vs_full():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(64, 1)), rng.normal(size=(1, 64))
    rank1 = u @ v
    s1 = matrix_spectrum_np(rank1)
    assert s1["top5_energy"] > 0.999 and s1["effective_rank"] < 1.5
    full = rng.normal(size=(64, 64))
    sf = matrix_spectrum_np(full)
    assert sf["top5_energy"] < 0.5 and sf["effective_rank"] > 20
    agg = aggregate_spectra({"a": s1, "b": sf})
    assert 0 < agg["top5_energy_wmean"] < 1 and agg["delta_norm_total"] > 0


def test_trak_projection_and_scores():
    rng = np.random.default_rng(1)
    g = rng.normal(size=(20, 5000)).astype(np.float32)
    phi = project_np(g, 1024, seed=7)
    assert phi.shape == (20, 1024)
    # Johnson-Lindenstrauss: squared norms (the diagonal of the Gram matrix) are preserved within ~20%
    true_sq = (g ** 2).sum(1)
    approx_sq = (phi ** 2).sum(1)
    rel = np.abs(approx_sq - true_sq) / true_sq
    assert rel.max() < 0.25, rel.max()
    sc = trak_scores_np(phi, phi[:5], phi[:8], lam=1e-2)
    assert sc.shape == (5, 8)


def test_attribution_summary_drop():
    rng = np.random.default_rng(2)
    orig = rng.normal(size=(10, 12))
    after = 0.1 * orig  # 90% drop in magnitude
    s = attribution_summary(orig, after, seed=0)
    assert 0.85 < s["attr_drop"] < 0.95


def test_steps_to_fraction():
    steps = [0, 50, 100, 150]
    vals = [0.1, 0.3, 0.6, 0.95]
    assert steps_to_fraction(steps, vals, target_value=1.0, frac=0.9) == 150
    assert steps_to_fraction(steps, vals, target_value=1.0, frac=0.99) == float("inf")
    assert steps_to_fraction([0], [0.95], target_value=1.0, frac=0.9) == 0


def test_probe_recovers_separable():
    rng = np.random.default_rng(3)
    # forget class linearly separable in features -> probe accuracy high
    n = 300
    f = rng.normal(2.0, 1.0, size=(n, 8))
    o = rng.normal(-2.0, 1.0, size=(n, 8))
    X = np.vstack([f, o])
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    perm = rng.permutation(2 * n)
    cut = int(0.7 * 2 * n)
    tr, te = perm[:cut], perm[cut:]
    res = probe_accuracy_np(X[tr], y[tr], X[te], y[te])
    assert res["probe_accuracy"] > 0.95


def test_h3_partial_spearman_removes_norm_confound():
    rng = np.random.default_rng(4)
    norm = rng.uniform(1, 5, size=40)
    # relearn depends ONLY on norm; effective rank is independent -> partial corr ~ 0
    relearn = 100 / norm + rng.normal(0, 1, size=40)
    rank = rng.uniform(1, 30, size=40)
    assert abs(partial_spearman(rank, relearn, norm)) < 0.4
    # now make relearn depend on rank too (low rank -> fast relearn) at fixed norm
    relearn2 = 100 / norm + 2 * rank + rng.normal(0, 1, size=40)
    assert partial_spearman(rank, relearn2, norm) > 0.4


def test_cifar_split():
    labels = np.array([0, 1, 2, 3, 3, 3, 3, 4] * 100)
    s = cifar_split_indices(labels, forget_class=3, forget_frac=0.01, seed=0)
    assert len(s["forget"]) == int(round(0.01 * len(labels)))
    assert set(s["forget"]).issubset(set(np.where(labels == 3)[0].tolist()))
    assert len(set(s["forget"]) & set(s["retain"])) == 0


def test_control_assertions():
    good = {"mia_loss_auc": 0.8, "probe_accuracy": 0.95, "relearn_steps_to_90": 0, "attr_drop": 0.0, "delta_norm_total": 0.0}
    assert control_assertions(good)["verdict"] == "passed"
    bad = dict(good, delta_norm_total=5.0, relearn_steps_to_90=50)
    assert control_assertions(bad)["verdict"].startswith("FAILED")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
