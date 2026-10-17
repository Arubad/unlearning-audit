# Experiment log — unlearning-audit

Every run gets an entry **before** launch (fill `Expected`) and is completed **after** it finishes (fill `Observed`, `Next change`). Append-only. The W&B project `unlearning-audit` mirrors entries, keyed by `config_hash`. Progress at the milestone level is tracked in [`ROADMAP.md`](ROADMAP.md).

Config hash = first 12 hex characters of `sha256` over the resolved YAML (`python -m src.utils.config_hash configs/<file>.yaml`).

## Entry format

```
## YYYY-MM-DD — <experiment id> — <one-line title>
- Config: configs/<file>.yaml  (hash: <12 hex>)
- Git commit: <short sha>
- Seeds: [..]
- Hardware: <GPU, platform>
- Wall-clock / GPU-h:
- Expected: <pre-registered prediction for this cell, with numbers>
- Observed: <numbers with 95 % CI; link to results/<run>/metrics.json and figure>
- Control status: <did the "do nothing" baseline fail every forget test on this harness version?>
- Next change: <exactly one thing changed before the next run, and why>
```

---

## 2026-10-13 — TEMPLATE — D3 gradient-ascent unlearning, CIFAR-10 class 3, forget 5 % (example entry, not a real run)

> **This entry is a template.** Placeholder values illustrate the required fields; nothing below was run.

- Config: configs/ddpm_unlearn_sweep.yaml  (hash: `0000deadbeef`)
- Git commit: `abc1234`
- Seeds: [0, 1, 2]
- Hardware: 1× T4 16 GB, Kaggle
- Wall-clock / GPU-h: 0.25 h per seed, 0.75 h total
- Expected: forget-class generation rate < 5 % (method "works" by its own metric); MIA AUC ≤ 0.55; probe accuracy ≥ 80 % of base; relearning steps ≤ 1/5 of oracle's (H1); top-5 singular mass > 70 % (H3).
- Observed: _(to be filled) gen-rate X %; MIA AUC a ± CI; probe acc b %; relearn steps c vs oracle d; m5 = e; results/D3_ga_f5_s0/metrics.json_
- Control status: "do nothing" checkpoint on commit `abc1234`: MIA AUC _, probe _, relearn 0 steps, Δθ = 0 — PASS/FAIL.
- Next change: _(to be filled) e.g. "none — proceed to label annealing" or "lower GA learning rate 1e-4 → 3e-5 because retain FID exploded at seed 2"._

---

_(real entries start below)_
