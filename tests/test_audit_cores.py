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




def test_mia_cores():
    from src.audit.mia import auc_from_scores, lira_scores_np, cmia_scores_np, class_centroids_np, mia_from_losses
    rng = np.random.default_rng(0)
    # loss-based: members have lower loss
    lm, ln = rng.normal(1.0, 0.3, 500), rng.normal(1.5, 0.3, 500)
    assert mia_from_losses(lm, ln) > 0.8
    assert abs(mia_from_losses(ln, ln) - 0.5) < 1e-9
    # offline LiRA: OUT models give high loss on everyone; a memorising target gives low loss on members only
    out = np.stack([rng.normal(1.5, 0.2, 500) for _ in range(3)])
    s_m = lira_scores_np(lm, out)
    s_n = lira_scores_np(ln, out)
    assert auc_from_scores(s_m, s_n) > 0.8
    # one OUT model falls back to the global std without error
    assert np.isfinite(lira_scores_np(lm, out[:1])).all()
    # class-geometry: members sit at the forget-class centroid, non-members between classes
    cents = np.eye(3) * 5
    feats = np.vstack([cents[1] + 0.1 * rng.normal(size=(50, 3)), 0.5 * (cents[1] + cents[2]) + 0.1 * rng.normal(size=(50, 3))])
    sc = cmia_scores_np(feats, None, 1, cents)
    assert auc_from_scores(sc[:50], sc[50:]) > 0.95
    c = class_centroids_np(np.array([[0, 0], [2, 2], [1, 1]]), np.array([0, 0, 1]), 3)
    assert np.allclose(c[0], [1, 1]) and np.allclose(c[1], [1, 1]) and np.allclose(c[2], 0)




def test_llm_attribution_projection_is_deterministic():
    from src.audit.attribution import project_np, trak_scores_np, attribution_summary
    rng = np.random.default_rng(1)
    g = rng.normal(size=(20, 5000)).astype(np.float32)
    a = project_np(g, 128, seed=7)
    b = project_np(g, 128, seed=7)
    assert a.shape == (20, 128) and np.allclose(a, b)           # same seed -> same projection
    assert not np.allclose(a, project_np(g, 128, seed=8))        # different seed -> different
    phi_tr, phi_f, phi_q = a[:12], a[12:16], a[:4]
    sc = trak_scores_np(phi_tr, phi_f, phi_q, lam=1e-2)
    assert sc.shape == (4, 4)
    # a model whose gradients shrink toward zero loses influence -> positive drop
    summ = attribution_summary(sc, sc * 0.1)
    assert summ["attr_drop"] > 0.5


# ---------------------------------------------------------------- amendment 2026-11-18

def test_config_aliases_reach_unlearn_and_data():
    from src.common.config import resolve
    cfg = {"train": {}, "unlearn": {"seed": 0, "forget_frac": 0.05}, "data": {"forget_frac": 0.05}}
    r = resolve(cfg, {"seed": 1, "forget_frac": 0.1})
    assert r["unlearn"]["seed"] == 1 and r["train"]["seed"] == 1
    assert r["unlearn"]["forget_frac"] == 0.1 and r["data"]["forget_frac"] == 0.1


def test_closeness_reference_points():
    from src.audit.fingerprint import closeness_np
    rng = np.random.default_rng(0)
    l0 = rng.normal(size=(200, 3))
    o1 = l0 + 1.0 + 0.1 * rng.normal(size=(200, 3))
    o2 = l0 + 1.0 + 0.1 * rng.normal(size=(200, 3))
    assert abs(closeness_np(l0, l0, [o1, o2])["E"]) < 1e-9                    # do nothing -> E = 0
    e_rep = closeness_np(l0 + 1.0 + 0.1 * rng.normal(size=(200, 3)), l0, [o1, o2])["E"]
    assert 0.9 < e_rep < 1.1                                                  # independent replicate -> E ~ 1
    assert closeness_np(l0 + 3.0, l0, [o1, o2])["E"] < 0                      # overshoot -> negative
    c = closeness_np(l0 + 0.5, l0, [o1, o2])
    assert c["E_lo"] <= c["E"] <= c["E_hi"] and c["floor_estimated"]
    # no trace of the removal (original as close to the oracles as they are to each other) -> NaN, not noise
    o3, o4 = l0 + 0.1 * rng.normal(size=(200, 3)), l0 + 0.1 * rng.normal(size=(200, 3))
    assert np.isnan(closeness_np(l0 + 0.5, l0, [o3, o4])["E"])


def test_signed_fingerprint_and_query_sets():
    from src.audit.fingerprint import query_sets, signed_fingerprint_np, NEAR_CLASSES
    rng = np.random.default_rng(0)
    ref = rng.normal(size=50)
    l0 = rng.normal(size=(50, 2))
    assert signed_fingerprint_np(l0 + ref[:, None], l0, ref) > 0.99
    labels = np.repeat(np.arange(10), 100)
    split = {"forget": np.where(labels == 3)[0].tolist(), "retain": np.where(labels != 3)[0].tolist()}
    q = query_sets(labels, np.repeat(np.arange(10), 20), split, 3, 50, seed=1)
    assert set(labels[q["near"]["indices"]]) <= set(NEAR_CLASSES) and 3 not in set(labels[q["far"]["indices"]])
    assert q == query_sets(labels, np.repeat(np.arange(10), 20), split, 3, 50, seed=1)   # fixed across calls


def test_rank_truncation_matches_norm():
    from src.audit.mechanism import truncated_delta_np, flat_cosine
    rng = np.random.default_rng(0)
    d = {"w": rng.normal(size=(32, 16, 3, 3)), "b": rng.normal(size=(32,))}
    t1 = truncated_delta_np(d, 1)
    assert np.linalg.matrix_rank(t1["w"].reshape(32, -1)) == 1
    assert np.isclose(np.linalg.norm(t1["w"]), np.linalg.norm(d["w"]))       # matched norm
    assert np.allclose(t1["b"], d["b"])                                        # 1-D tensors untouched
    assert np.isclose(flat_cosine(d, d), 1.0) and np.isclose(flat_cosine(d, {k: -v for k, v in d.items()}), -1.0)


def test_countsketch_preserves_inner_products():
    try:
        import torch  # noqa
    except Exception:
        return
    from src.audit.attribution import countsketch_np
    rng = np.random.default_rng(0)
    g = rng.normal(size=(6, 20000)).astype(np.float32)
    s = countsketch_np(g, 4096, seed=3)
    assert np.allclose(s, countsketch_np(g, 4096, seed=3))
    G, S = g @ g.T, s @ s.T
    assert np.allclose(np.diag(S) / np.diag(G), 1.0, atol=0.15)


def test_hypotheses_run_on_synthetic_records():
    from src.analysis.hypotheses import test_h1, test_h2, test_h3
    recs = []
    for seed in (0, 1):
        for m, e_f, e_n, steps, t5, er in (("oracle", 1, 1, 800.0, 0.1, 40.0), ("do_nothing", 0, 0, 0.0, 0.0, 0.0), ("gradient_ascent", 0.8, 0.1, 50.0, 0.9, 2.0 + seed),
                                           ("label_annealing", 0.7, 0.2, 100.0, 0.8, 3.0 + seed), ("gradient_difference", 0.75, 0.15, 150.0, 0.85, 4.0 + seed), ("retain_finetune", 0.0, 0.05, float("inf"), 0.3, 20.0)):
            recs.append({"cell": {"method": m, "seed": seed}, "regime": "class", "E_forget": e_f, "E_forget_lo": e_f - .05, "E_forget_hi": e_f + .05, "E_near": e_n,
                         "E_near_lo": e_n - .05, "E_near_hi": e_n + .05, "mia_loss_auc": 0.5, "probe_balanced_accuracy": 0.9 if m != "oracle" else 0.7,
                         "relearn_steps_to_90": steps, "top5_energy_wmean": t5, "effective_rank_wmean": er, "delta_norm_total": 1.0 + seed, "relearn_alignment": 0.5})
    h1, h2, h3 = test_h1(recs), test_h2(recs), test_h3(recs)
    assert h1["verdict"]["gradient_ascent"]["verdict"] == "confirmed"
    assert h2["verdict"]["class"]["gradient_ascent"]["verdict"] == "confirmed"
    assert h3["n_cells"] == 6 and h3["top5_energy_retain_finetune"] == 0.3




def test_llm_closeness_suite():
    from src.audit.fingerprint import fingerprint_suite_llm, llm_query_indices
    q = llm_query_indices({"forget": 50, "near": 300, "far": 0}, 40, seed=1)
    assert q == llm_query_indices({"forget": 50, "near": 300, "far": 0}, 40, seed=1) and "far" not in q
    assert len(q["forget"]) == 40 and len(set(q["near"])) == 40
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    # stub "models" are per-example NLL offsets; oracles differ from each other by small noise
    m0, mu = ("orig", base), ("after", base + 1.0)
    o1, o2 = ("o1", base + 1.0 + 0.05 * rng.normal(size=100)), ("o2", base + 1.0 + 0.05 * rng.normal(size=100))
    ex = {"forget": list(range(100))}
    nll = lambda m, e: m[1][e]
    out = fingerprint_suite_llm({"original": m0, "after": m0}, [o1, o2], ex, nll)
    assert abs(out["E_forget"]) < 1e-9                                  # do nothing -> 0
    out = fingerprint_suite_llm({"original": m0, "after": mu}, [o1, o2], ex, nll)
    assert out["E_forget"] > 0.9 and out["E_forget_lo"] <= out["E_forget"] <= out["E_forget_hi"]
    assert "d_floor_forget" in out and out["snr_forget"] > 1.2


def test_oppoint_selection_rule():
    from src.diffusion.oppoint import select_point, traj_name
    e1, e2 = {"method": "gradient_ascent", "lr": 1e-5}, {"method": "gradient_ascent", "lr": 3e-5}
    pts = [{"step": 0, "forget_rate": 0.02, "retain_acc": 0.95, "entry": e1},     # step 0 never selected
           {"step": 400, "forget_rate": 0.30, "retain_acc": 0.94, "entry": e1},
           {"step": 800, "forget_rate": 0.08, "retain_acc": 0.92, "entry": e1},
           {"step": 200, "forget_rate": 0.05, "retain_acc": 0.50, "entry": e2},   # forgets but breaks the model
           {"step": 800, "forget_rate": 0.09, "retain_acc": 0.93, "entry": e2}]
    s = select_point(pts, 0.10, 0.90)
    assert s["reached_target"] and s["step"] == 800 and s["entry"] is e2      # tie on step -> higher retain acc
    s = select_point([p for p in pts if p["forget_rate"] > 0.10 or p["retain_acc"] < 0.9], 0.10, 0.90)
    assert not s["reached_target"] and s["forget_rate"] == 0.30               # fallback: best forgetting at utility
    assert select_point([pts[3]], 0.10, 0.90) is None                         # no utility-preserving point
    assert traj_name({"method": "variational_ewc", "lr": 3e-5, "ewc_weight": 100.0}) == "variational_ewc_lr3em05_ewc100"


def test_standardized_closeness():
    """A difference confined to one low-variance column is invisible to plain E but recovered after standardising."""
    from src.audit.fingerprint import closeness_np, standardized_closeness_np

    rng = np.random.default_rng(0)
    n = 400
    scale = np.array([1.0, 1.0, 0.01])  # column 3 = "high noise": tiny absolute differences
    base = rng.normal(size=(n, 3)) * scale
    o0 = base + rng.normal(size=(n, 3)) * 0.1 * scale
    o1 = base + rng.normal(size=(n, 3)) * 0.1 * scale
    shift = np.array([0.0, 0.0, 0.5]) * scale  # the original differs from both oracles only in column 3
    l0 = base + rng.normal(size=(n, 3)) * 0.1 * scale + shift
    plain = closeness_np(l0, l0, [o0, o1], n_boot=50)
    std = standardized_closeness_np(l0, l0, [o0, o1], n_boot=50)
    assert plain["snr"] < 1.2 and np.isnan(plain["E"])
    assert std["snr"] > 2.0
    # an exact copy of an oracle is E = 1 - floor-term; doing nothing is E = 0
    assert abs(std["E"]) < 1e-9
    e_or = standardized_closeness_np(o0, l0, [o0, o1], n_boot=50)["E"]
    assert e_or > 0.8


def test_degeneracy_gate():
    """A5: smooth images pass, noise is flagged, and noise labelled as the target class does not count as valid."""
    import torch

    from src.diffusion.sample import HF_DEGENERATE, class_rates, degenerate_mask, hf_energy

    g = torch.Generator().manual_seed(0)
    smooth = torch.nn.functional.interpolate(torch.rand(8, 3, 4, 4, generator=g) * 2 - 1, size=32, mode="bilinear", align_corners=False)
    noise = torch.rand(8, 3, 32, 32, generator=g) * 2 - 1
    assert (hf_energy(smooth) < HF_DEGENERATE).all() and (hf_energy(noise) > HF_DEGENERATE).all()
    imgs = torch.cat([smooth, noise])
    assert degenerate_mask(imgs).tolist() == [False] * 8 + [True] * 8

    class AlwaysTarget(torch.nn.Module):  # labels everything as class 3
        def forward(self, x):
            out = torch.zeros(x.shape[0], 10)
            out[:, 3] = 1.0
            return out

    r = class_rates(imgs, AlwaysTarget(), 3)
    assert r["raw"] == 1.0 and r["valid"] == 0.5 and r["degenerate"] == 0.5


def test_score_c2_rules():
    """C2 scoring (PREREGISTRATION 'Confirmatory study C2') on hand-built inputs."""
    from src.analysis.label_diag import score_c2

    T = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 990]

    def qs(snr_plain, snr_t=None, pr=1.1, loo=None):
        d = {"snr_plain": snr_plain, "snr_std": snr_plain, "snr_t": snr_t or [1.0] * 12, "pair_ratio_t": [pr] * 12}
        if loo is not None:
            d["loo"] = [{"plain": {"E": e}, "std": {"E": e}} for e in loo]
        return d

    good_t = [1.5, 3, 4, 5, 4, 3, 1.5, 1, 0.9, 0.9, 0.8, 0.7]
    ctrl = {"timesteps": T, "labels": {
        "null_label": {"forget_train": qs(3.7, good_t, 1.2, [1.0, 1.01, 0.99]), "forget_test": qs(3.5), "near": qs(1.1), "far": qs(1.05)},
        "forget_label": {"forget_train": qs(1.2, None, 16.0, [float("nan"), 2.2, -0.97]), "forget_test": qs(1.2), "near": qs(1.0), "far": qs(1.0)}}}
    emb = {"s0": {"forget_row_moved_rel": 1e-7}}
    v = score_c2(ctrl, emb)
    assert all(v[k] == "held" for k in ("C2-0", "C2-A", "C2-B", "C2-C", "C2-D", "C2-E", "C2-F", "C2-G")) and v["sanity_forget_row_untrained"]
    # weak trace -> C2-A / C2-C untestable; miscalibrated held-out retrain -> C2-A failed
    ctrl["labels"]["null_label"]["forget_train"]["snr_plain"] = 1.2
    v = score_c2(ctrl, emb)
    assert v["C2-0"] == "failed" and v["C2-A"] == v["C2-C"] == "untestable"
    ctrl["labels"]["null_label"]["forget_train"].update(snr_plain=3.7, loo=[{"plain": {"E": 0.5}, "std": {"E": 0.5}}] * 3)
    assert score_c2(ctrl, emb)["C2-A"] == "failed"
    # homogeneous forget-label floor and calibrated forget-label LOO -> C2-F, C2-G failed
    ctrl["labels"]["forget_label"]["forget_train"].update(pair_ratio_t=[1.2] * 12, loo=[{"plain": {"E": 1.0}, "std": {"E": 1.0}}] * 3)
    v = score_c2(ctrl, emb)
    assert v["C2-F"] == "failed" and v["C2-G"] == "failed"


def test_score_c3b_rule():
    """C3-B-B (revised homogeneity rule) on ship-like and failing inputs."""
    from src.analysis.label_diag import score_c3b

    T = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 990]
    qs = lambda snr, pr, loo=None: {"snr_plain": snr, "snr_std": snr, "snr_t": [1.5, 1.8, 2, 2, 2, 1.7, 1.2, .8, .75, .7, .6, .6],
                                    "pair_ratio_t": pr, **({"loo": [{"plain": {"E": e}, "std": {"E": e}} for e in loo]} if loo else {})}
    ship_null = [1.24, 1.15, 1.06, 1.04, 1.11, 1.42, 1.8, 2, 1.5, 2, 2.1, 1.7]
    ship_forget = [1.98, 2.06, 2.85, 3.23, 2.35, 2.13, 2, 3, 5, 14, 16, 10]
    ctrl = {"timesteps": T, "labels": {
        "null_label": {"forget_train": qs(1.85, ship_null, [0.94, 1.13, 0.94]), "forget_test": qs(1.54, ship_null), "near": qs(1.08, ship_null), "far": qs(1.03, ship_null)},
        "forget_label": {"forget_train": qs(1.29, ship_forget, [1.81, 1.14, -1.92]), "forget_test": qs(1.2, ship_forget), "near": qs(1.0, ship_forget), "far": qs(1.0, ship_forget)}}}
    v = score_c3b(ctrl, {"s0": {"forget_row_moved_rel": 1e-7}})
    assert v["C2-B"] == "failed" and v["C3-B-B"] == "held" and v["claim_b_confirmed"]
    ctrl["labels"]["null_label"]["forget_train"]["pair_ratio_t"] = [1.6] + ship_null[1:]
    assert score_c3b(ctrl, {"s0": {"forget_row_moved_rel": 1e-7}})["C3-B-B"] == "failed"
    ctrl["labels"]["null_label"]["forget_train"]["pair_ratio_t"] = ship_null
    ctrl["labels"]["forget_label"]["forget_train"]["pair_ratio_t"] = [1.3] * 12  # null not below forget at t=400 (1.42)
    assert score_c3b(ctrl, {"s0": {"forget_row_moved_rel": 1e-7}})["C3-B-B"] == "failed"


def test_score_c3a_rules():
    """C3-A scoring on hand-built model tables."""
    from src.analysis.c3_consequence import score_c3a

    e = lambda E, lo=None: {"E": E, "E_lo": E - 0.1 if lo is None else lo, "E_hi": E + 0.1, "snr": 2.0}
    ob = {"o0": {"rate_valid": 0.02}, "o1": {"rate_valid": 0.08}, "o2": {"rate_valid": 0.05}}
    M = {"do_nothing": {"removed_label": e(0.0), "null_label": e(0.0), "rate_valid": 0.8}}
    for i, E in enumerate([0.1, 0.9, -1.0, 0.3] + [0.0] * 16):
        M[f"reset_{100 + i}"] = {"removed_label": e(E), "null_label": e(0.0), "rate_valid": 0.1 if i != 1 else 0.12}
    for s in (0, 1):
        M[f"forget_class8_methodlabel_annealing_seed{s}"] = {"removed_label": e(-0.5), "null_label": e(-0.3), "rate_valid": 0.0}
        M[f"forget_class8_methodretain_finetune_seed{s}"] = {"removed_label": e(0.2), "null_label": e(0.4, 0.2), "rate_valid": 0.6}
    v = score_c3a({"oracle_behaviour": ob, "models": M})
    assert v["C3-A1"] == "held" and v["n_reset_pass"] == 1 and v["C3-A2"] == "held" and v["C3-A3"] == "held" and v["C3-A4"] == "held"
    M["reset_101"]["rate_valid"] = 0.2  # behaviour above max oracle + 0.05 -> no pass
    M["forget_class8_methodretain_finetune_seed0"]["null_label"] = e(1.0, 0.85)
    M["forget_class8_methodlabel_annealing_seed1"]["null_label"] = e(0.1)
    v = score_c3a({"oracle_behaviour": ob, "models": M})
    assert v["C3-A1"] == "failed" and v["C3-A3"] == "failed" and v["C3-A4"] == "failed"
    assert score_c3a({"oracle_behaviour": ob, "models": {k: x for k, x in M.items() if "_method" not in k}})["C3-A3"] == "untestable"



def test_c4_row_swap_and_rules():
    from src.analysis.c4_controls import row_swap_stats, score_c4r, score_c4x
    rng = np.random.default_rng(0)
    T = [10, 50, 100, 200, 300, 400, 500]
    net = {n: rng.normal(0, 0.01, (64, len(T))) for n in range(3)}
    row = {r: rng.normal(0, 1.0, (64, len(T))) for r in range(3)}  # the row dominates the response
    L = {(n, r): net[n] + row[r] for n in range(3) for r in range(3)}
    st = row_swap_stats(L, T)
    assert st["row_wins_low_t"] == 6
    assert st["H_common"] < st["H_indep"] or st["H_indep"] >= 1.0
    st["loo_common"] = [1.0, 0.9, 1.1, float("nan"), 1.3, 0.7, 2.0, 1.0, 1.0]
    v = score_c4r({"class1": st, "class3": st, "class8": st, "class9": st})
    assert v["R1_overall"] == "held" and v["R3"]["class1"] == "held" and v["R3_overall"] == "held"
    flip = {(n, r): row[n] + net[r] for n in range(3) for r in range(3)}  # the network dominates
    st2 = row_swap_stats(flip, T)
    assert st2["row_wins_low_t"] == 0
    st2["loo_common"] = [float("nan")] * 9
    v2 = score_c4r({"a": st2, "b": st2, "c": st2, "d": st2})
    assert v2["R1_overall"] == "failed" and v2["R3_overall"] == "untestable"
    assert score_c4x({"c": [0.1, -0.2, 0.49]})["X1"] == "held"
    assert score_c4x({"c": [0.1, 0.6]})["X1"] == "failed"
    assert score_c4x({"c": [0.1, float("nan")]})["X1"] == "failed"


def test_c4_frechet_and_rules():
    from src.analysis.c4_fd import frechet, score_c4f
    rng = np.random.default_rng(1)
    f = rng.normal(size=(500, 8)); g = rng.normal(loc=1.0, size=(500, 8))
    mu, S = f.mean(0), np.cov(f, rowvar=False)
    assert abs(frechet(mu, S, mu, S)) < 1e-6
    assert abs(frechet(f.mean(0), np.cov(f, rowvar=False), g.mean(0), np.cov(g, rowvar=False)) - 8.0) < 1.5
    d = {"retrain_pairs_k": [10, 20, 30], "retrain_pairs_0": [5, 5, 6], "orig_vs_retrain": [50, 60, 70],
         "redraw_vs_retrain": [25] * 20 + [40] * 40}
    v = score_c4f({"a": d, "b": d, "c": dict(d, redraw_vs_retrain=[40] * 60)})
    assert v["F1"]["a"] == "held" and abs(v["F1_fraction"]["a"] - 1 / 3) < 1e-9 and v["F1"]["c"] == "failed"
    assert v["F1_overall"] == "held" and v["F2_overall"] == "held" and v["F3_overall"] == "held"
    assert score_c4f({"a": dict(d, orig_vs_retrain=[25, 60, 70])})["F3_overall"] == "failed"


def test_c5_fido_pooling_seeds_and_rules():
    from src.analysis.c4_fd import frechet
    from src.analysis.c5_fido import fid_o, pooled_stats, score_c5, seed_for
    rng = np.random.default_rng(2)
    # pooling = Gaussian fit to the concatenation
    a, b = rng.normal(size=(300, 4)), rng.normal(loc=2.0, size=(200, 4))
    mu, S = pooled_stats([a, b])
    cat = np.concatenate([a, b])
    assert np.allclose(mu, cat.mean(0)) and np.allclose(S, np.cov(cat, rowvar=False))
    # identical per-label features -> FID_O 0; a shift in one label of ten moves the pooled FD less than the label alone
    A = {l: rng.normal(loc=l, size=(400, 3)) for l in range(10)}
    assert abs(fid_o(A, A)) < 1e-6
    B = dict(A); B[3] = A[3] + 3.0
    full = fid_o(A, B); single = frechet(*pooled_stats([A[3]]), *pooled_stats([B[3]]))
    assert 0 < full < single
    assert abs(fid_o(A, B, [l for l in range(10) if l != 3])) < 1e-6  # retained-only ignores label 3
    # registered seeds are distinct across all (kind, class, seed, label) used
    seeds = [seed_for("origA", l) for l in range(10)] + [seed_for("origB", l) for l in range(10)]
    seeds += [seed_for("retrain", l, k=k, s=s) for k in (8, 1, 9) for s in (0, 1, 2) for l in range(10)]
    seeds += [seed_for("redraw", k, k=k, r=r) for k in (8, 1, 9) for r in range(100, 120)]
    assert len(seeds) == len(set(seeds))
    # rules
    d = {"retrain_pairs": [2.0, 3.0, 4.0], "same_model_floor": 1.0, "orig_vs_retrain": [5.0, 6.0, 7.0],
         "redraw_vs_retrain": [3.5] * 20 + [9.0] * 40, "retrain_pairs_retained": [1.0, 1.5, 2.0]}
    v = score_c5({"a": d, "b": d, "c": dict(d, redraw_vs_retrain=[9.0] * 60, same_model_floor=2.5)})
    assert v["O1"]["a"] == "held" and abs(v["O1_fraction"]["a"] - 1 / 3) < 1e-9 and v["O1"]["c"] == "failed"
    assert v["O1_overall"] == "held" and v["O2"]["c"] == "failed" and v["O2_overall"] == "held"
    assert v["O3_overall"] == "held" and v["O4_overall"] == "held"
    assert score_c5({"a": dict(d, orig_vs_retrain=[3.9, 6.0, 7.0])})["O3_overall"] == "failed"
    assert score_c5({"a": dict(d, retrain_pairs_retained=[2.5, 2.5, 2.5])})["O4_overall"] == "failed"


if __name__ == "__main__":  # noqa
    # Must stay the last statement in the file: tests defined below it would never run.
    # Guard: every top-level `def test_` in the source must have been executed.
    import re
    ran = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and getattr(fn, "__module__", None) == "__main__":
            fn(); print("ok", name); ran.append(name)
    defined = re.findall(r"^def (test_\w+)", open(__file__).read(), flags=re.M)
    missing = sorted(set(defined) - set(ran))
    assert not missing and len(ran) == len(defined), f"tests defined but not run: {missing}"
    print(f"all {len(ran)} tests passed")
