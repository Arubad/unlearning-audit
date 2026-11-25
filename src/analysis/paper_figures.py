"""Paper figures, regenerated from the cached analysis outputs in results/figures/*.json (no GPU, no checkpoints).

Usage:  python -m src.analysis.paper_figures [results_root] [out_dir]
Writes fig1..fig5 as PDF (for the paper) and PNG (for previews) into out_dir (default paper/figures).

Fig 1  calibration: held-out-retrain E under the removed-label vs the null-label reference (C2, C3-B; per class)
Fig 2  the removed-label embedding row of every exact retrain never moves from its initialisation
Fig 3  C3-A: zero-step re-draws of that row pass the removed-label audit; the null-label audit scores them 0
Fig 4  C3-K: re-fitting only that row restores the class in re-draws, not in retrains
Fig 5  per-timestep retrain-pair heterogeneity, removed label vs null label
"""
from __future__ import annotations

import json
import math
import os
import sys

CLASSES = {1: "automobile", 3: "cat", 8: "ship", 9: "truck"}
# reference categorical palette, slots 1-3 (validated all-pairs, light mode); see the dataviz skill's palette.md
C_NULL, C_REM, C_REM_STD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BAND = "#eeeeea"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    print("wrote", os.path.join(out, name + ".{pdf,png}"))


def fig1_calibration(L, out):
    """Held-out retrain closeness E (leave-one-oracle-out) per class; E = 1 means the audit recognises a retrain."""
    import matplotlib.pyplot as plt
    series = [("forget_label", "plain", C_REM, "o", "removed label, plain"),
              ("forget_label", "std", C_REM_STD, "s", "removed label, standardised"),
              ("null_label", "plain", C_NULL, "D", "null label, plain (proposed)")]
    lo_clip, hi_clip, undef_y = -2.6, 2.6, 3.05
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    ax.axhspan(0.6, 1.4, color=BAND, zorder=0, lw=0)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    order = [1, 3, 8, 9]
    for gi, c in enumerate(order):
        labs = L[f"class{c}"]["controls"]["labels"]
        for si, (lab, stat, col, mk, name) in enumerate(series):
            for li, e in enumerate(labs[lab]["forget_train"]["loo"]):
                x = gi + (si - 1) * 0.26 + (li - 1) * 0.07
                r = e[stat]
                if not _ok(r["E"]):
                    ax.scatter([x], [undef_y], marker=mk, s=22, facecolors="white", edgecolors=col, lw=1.1, zorder=3)
                    continue
                y = min(max(r["E"], lo_clip), hi_clip)
                if _ok(r.get("E_lo")) and _ok(r.get("E_hi")):
                    ax.plot([x, x], [max(r["E_lo"], lo_clip), min(r["E_hi"], hi_clip)], color=col, lw=1.2, zorder=2)
                ax.scatter([x], [y], marker=mk, s=22, color=col, edgecolors="white", lw=0.6, zorder=3)
    ax.axhline(2.85, color=GRID, lw=0.8)
    ax.text(3.62, undef_y, "undefined\n(SNR < 1.2)", va="center", ha="left", fontsize=7, color=INK2)
    ax.text(3.62, 1.0, "calibrated\n(E = 1)", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CLASSES[c] + ("\n(pre-registered)" if c in (8, 9) else "") for c in order])
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(lo_clip - 0.15, undef_y + 0.3)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylabel("held-out retrain closeness E")
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=mk, color=col, mec="white", ms=6, label=name) for _, _, col, mk, name in series]
    h.append(Line2D([], [], ls="", marker="o", color="white", mec=INK2, ms=5, label="hollow = undefined"))
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.3, columnspacing=1.0)
    _save(fig, out, "fig1_calibration")
    plt.close(fig)


def fig2_rows(L, out):
    """Distance moved from initialisation by the removed-label row vs the trained rows, every exact retrain."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    order = [1, 3, 8, 9]
    for gi, c in enumerate(order):
        emb = L[f"class{c}"]["embedding"]
        for si, s in enumerate(sorted(emb)):
            x = gi + (si - 1) * 0.12
            ax.scatter([x], [emb[s]["other_rows_moved_mean"]], color=C_NULL, marker="D", s=20, edgecolors="white", lw=0.6,
                       zorder=3, label="trained rows (mean)" if gi == 0 and si == 0 else None)
            ax.scatter([x], [emb[s]["forget_row_moved"]], color=C_REM, marker="o", s=22, edgecolors="white", lw=0.6,
                       zorder=3, label="removed-label row" if gi == 0 and si == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CLASSES[c] for c in order])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("‖row − init‖₂ after 40k steps")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("3 retrains per class", loc="left", color=INK2, fontsize=7.5)
    _save(fig, out, "fig2_untrained_row")
    plt.close(fig)


def fig3_consequence(C, out):
    """C3-A: removed-label vs null-label E of 20 zero-step re-draws per class; pass = E in [0.5, 1.5] and behaviour gate."""
    import matplotlib.pyplot as plt
    import numpy as np
    order = [("class8", 8), ("class1", 1), ("class9", 9)]
    lo, hi = -4.0, 4.0
    fig, ax = plt.subplots(figsize=(6.4, 1.8))
    rng = np.random.default_rng(0)  # vertical jitter only (display)
    yt, yl = [], []
    for gi, (key, c) in enumerate(order):
        v = C[key]
        th = v["c3a_verdicts"]["behaviour_threshold"]
        resets = [m for k, m in v["models"].items() if k.startswith("reset")]
        stat = v["removed_label_statistic"]
        rows = [("removed_label", C_REM, "o", f"removed label ({'standardised' if stat == 'std' else 'plain'})"),
                ("null_label", C_NULL, "D", "null label, plain (proposed)")]
        for ri, (ref, col, mk, name) in enumerate(rows):
            y0 = gi * 2.6 + (1 - ri) * 1.0
            yt.append(y0); yl.append(f"{CLASSES[c]} · {'removed' if ri == 0 else 'null'}")
            n_pass = 0
            for m in resets:
                e = m[ref]["E"]
                if not _ok(e):
                    continue
                passed = 0.5 <= e <= 1.5 and m["rate_valid"] <= th
                n_pass += passed
                y = y0 + rng.uniform(-0.22, 0.22)
                xc = min(max(e, lo), hi)
                clipped = xc != e
                ax.scatter([xc], [y], marker=(">" if e > hi else "<") if clipped else mk, s=18 if passed else 14,
                           color=col if passed else "white", edgecolors=col, lw=1.0, zorder=3)
            ax.text(hi + 0.25, y0, f"{n_pass}/{len(resets)} pass", va="center", ha="left", fontsize=7.5, color=INK)
    ax.axvspan(0.5, 1.5, color=BAND, zorder=0, lw=0)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(-0.6, (len(order) - 1) * 2.6 + 1.6)
    ax.invert_yaxis()
    ax.set_xlabel("closeness E of a zero-step re-draw of the removed-label row (pass band 0.5–1.5; ◂ clipped)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=C_REM, mec=C_REM, label="removed-label audit"),
         Line2D([], [], ls="", marker="D", color=C_NULL, mec=C_NULL, label="null-label audit"),
         Line2D([], [], ls="", marker="o", color="white", mec=INK2, label="fails E band or behaviour gate (hollow)"),
         Line2D([], [], ls="", marker="o", color=INK2, mec=INK2, label="passes (filled)")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handletextpad=0.3)
    _save(fig, out, "fig3_redraws_pass")
    plt.close(fig)


def fig4_knowledge(K, C, out):
    """C3-K: forget-class generation rate before -> after re-fitting only the removed-label row (network frozen)."""
    import matplotlib.pyplot as plt
    order = [8, 1, 9]
    fig, ax = plt.subplots(figsize=(6.4, 1.55))
    xt, xl = [], []
    for gi, c in enumerate(order):
        ms = K[c]["models"]
        orig = C[f"class{c}"]["models"]["do_nothing"]["rate_valid"]
        base = gi * 4.0
        ax.plot([base - 0.4, base + 3.0], [orig, orig], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        if gi == len(order) - 1:
            ax.text(base + 3.05, orig, "original", va="center", ha="left", fontsize=7, color=INK2)
        oracles = sorted(k for k in ms if k.startswith("oracle"))
        resets = sorted(k for k in ms if k.startswith("reset"))
        for j, k in enumerate(oracles + resets):
            x = base + j * 0.5 + (0.3 if k in resets else 0.0)
            col = INK2 if k in oracles else C_REM
            b, a = ms[k]["before"]["valid"], ms[k]["after"]["valid"]
            ax.plot([x, x], [b, a], color=col, lw=1.3, zorder=2, solid_capstyle="butt")
            ax.scatter([x], [a], s=14, color=col, edgecolors=col, lw=1.0, zorder=3)  # after: filled, exactly at the value
            ax.scatter([x], [b], s=12, color="white", edgecolors=col, lw=1.0, zorder=3)
        xt += [base + 0.5, base + 2.3]
        xl += [f"{CLASSES[c]}\nretrains", f"{CLASSES[c]}\nre-draws"]
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("samples classified as the class")
    ax.grid(axis="x", visible=False)
    ax.set_title("before (○) → after (●) 400 Adam steps on the 256-number row only, network frozen", loc="left",
                 color=INK2, fontsize=7.5)
    _save(fig, out, "fig4_row_refit")
    plt.close(fig)


def fig5_timesteps(L, out):
    """Max/min distance over retrain pairs, per timestep: removed label vs null label (training images of the class)."""
    import matplotlib.pyplot as plt
    order = [1, 3, 8, 9]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.0), sharey=True)
    for ax, c in zip(axes, order):
        ctl = L[f"class{c}"]["controls"]
        t = ctl["timesteps"]
        for lab, col, name in (("forget_label", C_REM, "removed label"), ("null_label", C_NULL, "null label")):
            r = ctl["labels"][lab]["forget_train"]["pair_ratio_t"]
            ax.plot(t, r, color=col, lw=1.5, marker="o", ms=2.8, label=name)
        ax.axvspan(1, 400, color=BAND, zorder=0, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(CLASSES[c], loc="left")
        ax.set_xlabel("timestep t")
        ax.grid(which="minor", visible=False)
        ax.set_xlim(7, 1400)
        ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"]); ax.minorticks_off()
    axes[0].set_ylabel("max/min retrain-pair distance")
    fig.legend(handles=axes[0].get_lines(), loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=2)
    fig.text(0.99, 1.02, "shaded: registered homogeneity window t ≤ 400", ha="right", va="bottom", fontsize=7, color=INK2)
    _save(fig, out, "fig5_pair_heterogeneity")
    plt.close(fig)


def main(root: str = "results", out: str = "paper/figures") -> None:
    import matplotlib
    matplotlib.use("Agg")
    _style()
    os.makedirs(out, exist_ok=True)
    fd = os.path.join(root, "figures")
    L = json.load(open(os.path.join(fd, "label_diag.json")))
    C = json.load(open(os.path.join(fd, "c3a_consequence.json")))
    K = {c: json.load(open(os.path.join(fd, f"c3_knowledge_c{c}.json"))) for c in (8, 1, 9)}
    fig1_calibration(L, out)
    fig2_rows(L, out)
    fig3_consequence(C, out)
    fig4_knowledge(K, C, out)
    fig5_timesteps(L, out)


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
