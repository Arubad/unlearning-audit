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

## 2026-11-18 — SMOKE — end-to-end harness check after the 2026-11-18 amendment
- Config: `src/smoke.py` (0.70M-param UNet, 200 train steps, 2 oracles, forget_frac 0.10)
- Git commit: uncommitted working tree on top of `fb1a285` (amendment + fixes)
- Seeds: [0] (oracles 0, 1)
- Hardware: 1× T4 16 GB, Google Colab free tier (Colab CLI)
- Wall-clock / GPU-h: 432 s / 0.12 GPU-h
- Expected: every stage runs; do-nothing structural checks pass (Δθ = 0, E = 0, no attribution drop, relearn 0); MIA/probe checks may fail on a 200-step model.
- Observed: all stages ran (classifier, base, 2 oracles, oracle/do_nothing/GA/label_annealing/retain_finetune cells, truncation k ∈ {1,4,full}, analyze). Do-nothing: zero_delta, zero_closeness, no_attr_drop, no_relearn PASS; mia_above_chance FAIL (AUC 0.50), probe_full FAIL (0.58) — expected for an untrained model. E values were unbounded (±600) because d(orig, O) ≈ d_floor in a 200-step model → added the snr guard (E = NaN when d(orig,O)/d_floor < 1.2). Legacy statistic at the oracle: attr_drop = −1.95 (|TRAK| grows at the oracle), consistent with the reason H2 was re-operationalised. Relearn ratio 0/0 counted as a pass → now NaN.
- Control status: structural checks PASS; memorisation checks are deferred to the real base model.
- Next change: data loading switched to the HF parquet mirror (Toronto server served ~50 kB/s); then D1 base.

## 2026-11-18 — C0 — CIFAR-10 classifier for the generation-rate metric
- Config: `src/diffusion/classifier.py` defaults (via `src.pipeline classifier`)
- Git commit: uncommitted working tree on top of `fb1a285`
- Seeds: [0]
- Hardware: 1× T4, Google Colab free tier
- Wall-clock / GPU-h: 0.08 h
- Expected: ≥ 85 % test accuracy (enough to score "is this sample a cat" for the forget-class generation rate).
- Observed: 88.6 % test accuracy. `results/checkpoints/cifar10_classifier.pt`.
- Control status: n/a.
- Next change: none — proceed to D1.

## 2026-11-18 — D1 — base class-conditional DDPM, all 10 classes
- Config: `configs/diffusion_base.yaml` (hash `3fb21da8c84a` as resolved now; the run used `ckpt_every: 5000`, changed to 2000 afterwards — checkpoint cadence only, no effect on training)
- Git commit: uncommitted working tree on top of `fb1a285`
- Seeds: [0]
- Hardware: 1× T4, Google Colab free tier, uninterrupted
- Wall-clock / GPU-h: 3.32 h (≈ 0.30 s/step, GPU-bound at 96 % utilisation)
- Expected: 6.48M-param UNet, 40k steps × batch 128 ≈ 102 epochs; loss plateau ≈ 0.03 (eps-MSE).
- Observed: mean eps-MSE over the last 2k steps 0.0294. `results/checkpoints/ddpm_cifar10_base.pt` (+ `_history.json`). FID not computed (optional per amendment).
- Control status: deferred to the do_nothing audit cell (D3).
- Next change: none — proceed to D2.

## 2026-11-19 — D2 (class regime) — two oracle retrains on the retain set, forget_frac 0.10 (no cats)
- Config: `configs/diffusion_base.yaml` + `{seed: s, forget_frac: 0.10}` (hash `8ab3131bf0d9` seed 0, `9f96fda27bbf` seed 1)
- Git commit: uncommitted working tree on top of `fb1a285` + Kaggle runner changes (deadline, per-seed processes)
- Seeds: oracle seeds [0, 1]; split_seed 0 (same 5,000 cats removed for both)
- Hardware: seed 0 — Colab T4 to step 16,000 (session reclaimed ≈ 04:30 IST), then Kaggle T4 (16,001 → 17,000 in a 15-min harness test, 17,001 → 40,000 in run v2). Seed 1 — Kaggle T4 (0 → 900 in the test, then 901 → 40,000 in run v2), in parallel with seed 0 on the second GPU.
- Wall-clock / GPU-h: seed 0 ≈ 1.6 h (Colab) + 1.91 h (Kaggle); seed 1 3.21 h.
- Expected: same plateau as D1 (retain data are 90 % of D1's and the schedule is identical).
- Observed: last-2k-step mean eps-MSE 0.0287 (s0), 0.0311 (s1). Both at 40,000 steps.
- Note (reproducibility): resumes restore model, EMA, optimiser, AMP scaler and RNG state, but the data loader is re-seeded on resume, so seed 0's data order after step 16,000 differs from an uninterrupted run. This changes which realisation of training randomness seed 0 is, not its distribution; the two oracles remain exchangeable draws for the noise floor. Base (Colab) and oracles (Kaggle, torch 2.11) ran on different software stacks; claims are distributional, so this is recorded, not corrected.
- Control status: n/a.
- Next change: none — proceed to D3 class sweep.

## 2026-11-19 — D3/D4/D5 (class regime) — sweep attempt 1, CRASHED (no data)
- Config: `configs/diffusion_unlearn_sweep.yaml`, forget_frac 0.10, seeds {0, 1} split across two GPUs
- Git commit: uncommitted working tree
- Hardware: Kaggle 2× T4, run `arushb/ua-runner` v2
- Wall-clock / GPU-h: crashed ≈ 10 min into the first cell (oracle cells, both seeds)
- Expected: oracle cells audit cleanly; E on the oracle cells = 1 − 1/SNR by construction (one reference oracle, floor = 0).
- Observed: `torch.OutOfMemoryError` in `projected_grads_diffusion` (TRAK per-example gradients): `vmap(grad)` over a batch of 16 on the 6.48M-param UNet needed ≈ 14 GB. The smoke test used a 0.70M-param UNet and did not exercise this. Forgetting, MIA and fingerprint stages had completed for both cells but the record is only written at the end, so nothing was saved. The local driver read Kaggle's "COMPLETE" status as success and started to relaunch; stopped by hand before a second run.
- Control status: not reached.
- Next change: per-example gradients in vmap chunks of 4 (config `audit.attribution.grad_chunk`), halving on OOM down to 1; attribution made non-fatal (secondary readout; failure is recorded as `attr_error`); per-axis timings printed; the runner writes `_runner_status.json` and the driver stops on any non-zero pipeline exit instead of relaunching. Relaunched as run v3, 2026-11-19 12:45 IST.

## 2026-11-19 — D3/D4/D5 (class regime) — sweep attempt 2, CRASHED (stale code, no data)
- Hardware: Kaggle 2× T4, run `arushb/ua-runner` v3, 12:45–12:57 IST
- Observed: identical OOM traceback with the *old* line numbers and none of the new log lines: the kernel mounted the dataset version uploaded at 12:42 (old code) instead of the 12:44 version with the fix, although `kaggle datasets status` reported "ready" before the push.
- Next change: the driver now publishes code + state at the start of every cycle, downloads `ua_code.bin` back until its sha256 matches the local build, bakes that sha into the runner, and the runner exits with `stale_code` if the mounted code differs. Relaunched as run v4 (code `bab2d7ed64cc`), 13:07 IST.

## 2026-11-19 — compute budget — Kaggle GPU quota nearly exhausted during run v4
- Observed 15:18 IST: `kaggle quota` reported 27.42 / 30 GPU-h used (resets Sat 2026-11-21 00:00 UTC). Charge rate measured at about 1 GPU-h per wall-hour on T4×2. Run v4 had an 11 h budget, so it will hit the quota at about 17:50 IST, and a session stopped by the quota may save no output.
- Next change: driver v2 is quota-aware. It waits for the refresh when less than 3 h remain and bakes a budget of min(11 h, remaining − 0.5 h) into the runner, so every run ends at its own clean deadline.

## 2026-11-19 — L4 (LLM audit) — E ported; two protocol bugs fixed before any LLM data (no GPU)
- Change: oracle-normalised closeness E on answer-token NLL (`fingerprint_suite_llm`), with query sets forget / heldout / near (retain-train) / far (world_facts). Two oracle seeds are listed in `configs/llm_unlearn_sweep.yaml` (`oracle_checkpoints`). The LLM smoke test now trains 2 oracles and asserts E is present.
- Bug 1: the held-out MIA non-member slice depended on the training/cell seed, so for seed ≥ 1 cells the "non-members" were base-model training data, and oracle seeds trained on different retain sets. Fixed with one split seed (`HELDOUT_SPLIT_SEED`) for base, oracles, unlearning and audit. The held-out index file is renamed `heldout_indices_f{frac}.json`.
- Bug 2: GradDiff's retain term sampled the held-out slice, i.e. it trained on MIA non-members. It now uses the retain-train slice only.
- Analysis: per-track reports (LLM under `report["llm"]`), and the do-nothing control is kept for every (track, regime, seed) cell.
- Verification: 20/20 numpy and torch core tests pass locally, including a new `test_llm_closeness_suite`. Report logic was checked on synthetic two-track records. The LLM end-to-end smoke has NOT been run (no transformers/GPU locally); it runs first in the LLM GPU job.

## 2026-11-19 — D3/D4/D5 (class regime) — sweep run v4, COMPLETED (14/14 cells); truncation CRASHED
- Config: `configs/diffusion_unlearn_sweep.yaml`, forget_frac 0.10, 7 methods × seeds {0, 1} (one seed per T4)
- Git commit: uncommitted working tree, code `bab2d7ed64cc`
- Hardware: Kaggle 2× T4, run `arushb/ua-runner` v4, 13:07–19:38 IST (6.49 h). The session continued about 1.5 h past the exhausted weekly quota.
- Wall-clock per cell: about 52 min. Forgetting + MIA + fingerprint 330–375 s; attribution (TRAK) **2,410 s**; probe/spectrum/alignment < 10 s; relearning 45–275 s.
- Expected (P1): do-nothing MIA AUC 0.52–0.62, so the control may fail. Expected (H2): E_forget high, E_near low for approximate methods.
- Observed (full table and interpretation: `paper/RESEARCH_JOURNAL.md` §7):
  - **MIA.** do_nothing loss-MIA 0.516, SecMI 0.519, LiRA 0.539, so P1 held and the control fails on MIA.
  - **Noise floor.** $d(\theta_0,O)/d_\text{floor}$ is 0.96–1.00 on every query set, so E is undefined everywhere by the SNR guard. Removing all cats shifts the mean cat loss by +0.0056, below the per-example seed floor of 0.0076, and shifts near/far losses by 0.0000.
  - **gradient_ascent / variational_ewc.** Cat rate 0.00–0.03, but retain generation accuracy collapses to 0.10–0.12 (chance). They are 165–175× the floor from the oracle on cats and about 1,600× on retain classes. Relearning takes 50–100 steps vs 500–550 from the oracle (ratio 0.10–0.20). Top-5 energy 0.93–0.94; alignment 0.035–0.043.
  - **gradient_difference.** Cat rate 0.65–0.73, retain 0.93–0.94.
  - **label_annealing.** No forgetting (0.78–0.83).
  - **retain_finetune.** Control unchanged (0.87–0.92).
  - **Probe.** Does not discriminate: original 0.73/0.76, oracle 0.73/0.77.
  - **TRAK validity check.** corr(prediction, oracle change) ≈ −0.04, so TRAK is uninformative about retraining. It correlates 0.12–0.19 with the changes made by ascent methods.
- Control status: FAILED on `mia_above_chance` (predicted, P1) and on `probe_full`. The latter is a mis-calibrated absolute threshold of 0.8 set before data, replaced by `probe_above_chance` ≥ 0.6 (harness fix, logged). Zero-delta, zero-closeness and no-relearn checks passed.
- Crash: `truncate_c` raised FileNotFoundError in both processes. `cell_name` depended on dict key order, so the truncation stage looked in `methodgradient_ascent_forget_frac0p1_seed{s}/` instead of `forget_frac0p1_methodgradient_ascent_seed{s}/`. No sweep data was lost.
- Next change: `cell_name` sorts keys (existing directories unchanged). The truncation intervention runs first in the next session. Driver restarted at 19:46; it waits for the quota refresh (Sat 21 Nov 05:30 IST).

## 2026-11-21 — run v5 — truncation (P5), operating-point sweep (A3, exploratory), subset regime — LAUNCHED
- Code: sha256 `07a0d9e15cd8…`, verified as served by Kaggle before push. Budget 11.0 h of a fresh 30 GPU-h weekly quota.
- Stages: `truncate_c` → `oppoint` (8 trajectories over 2 GPUs) → `wait_oppoint` (selection rule, written to `results/D6_oppoint/selected.json`) → `sweep_op` (selected points + `embedding_reset`, seeds 0 and 1, TRAK off) → subset-regime oracles and sweep, as time allows.
- Expected: written down before any data. See P5 and A3-a … A3-e in `paper/RESEARCH_JOURNAL.md` §3.
- Pre-flight: a local CPU smoke of every new stage passed (tiny model, 60 steps, numbers meaningless). The smoke wrote three throwaway entries into this log through the relative log path. They were removed on 2026-11-21 and were never experiments.
- Process note: the driver missed the 05:45 IST launch and pushed at 12:46. Its quota-wait `sleep` does not advance while macOS sleeps. No quota or data was lost. Fix after this run: wait in short slices against wall-clock time.
- Observed (2026-11-21 21:25): COMPLETED. Exit codes [0, 0]; 7.38 GPU-h. Stages: truncate_c, oppoint, wait_oppoint, sweep_op, oracle_s, wait_oracle_s, sweep_s, analyze. `ALL_DONE` written. 32 audit records: 14 class + 14 subset + 4 A3.
- Truncation (P5): every rank, k = 1 included, at matched norm leaves retain accuracy at 0.10–0.12. Rank is irrelevant at matched norm; the norm destroys the model.
- A3: only gradient difference has an operating point (lr 3e-4, step 600). It does not transfer to seed 1: cat 0.794. Ascent and variational EWC collapse within 200 steps at every setting. A3-a…e all failed.
- Subset regime: do-nothing plain SNR 1.30, so E is defined; LiRA 0.615. Gradient ascent and gradient difference LiRA 0.95–0.98.
- Details and interpretation: `paper/RESEARCH_JOURNAL.md` §7.10.
- Next change: confirmatory study C1 (cycle 6).

## 2026-11-21 — C1 (confirmatory) — oracles for class 1 (seeds 0, 1, 2) + cat oracle seed 2 — LAUNCHED
- Registered in `PREREGISTRATION.md` before any C1 data. Code `827e2688a52f`, budget 8.0 h, `UA_STAGES=oracle_confirm`, jobs `1:0,1:1,1:2,3:2` (two per GPU).
- Pushed 21:30 IST; running immediately.
- Expected: 4 × 40k-step retrains, ≈ 6.4 wall-h. Scoring with `python -m src.analysis.c1_report` on MPS.
- Observed (2026-11-22): COMPLETED, kernel v6. All four retrains reached 40k steps (3.1–3.3 h each). Collected into `results/checkpoints/` at about 04:18 IST. Weekly quota afterwards: 14.03 h used, 15.97 h left.
- Process note: the Mac rebooted at about 10:41 IST, after collection, and the reboot wiped the session scratchpad (driver scripts, venv, `kout_*` logs). Nothing in the repo was lost. The runner was re-pulled from Kaggle with `kaggle kernels pull`.
- Registered verdicts (`src/analysis/c1_report.py`, unchanged):

  | | C1-1 | C1-2 | C1-3 | C1-4 | C1-5 | C1-6 |
  |---|---|---|---|---|---|---|
  | automobile | failed | failed | failed | failed | held | failed |
  | cat | held | held | held | held | held | failed |

  **The headline did not replicate**; it is class-specific.
  - Automobile train standardised SNR 1.18, plain 1.22. There is no high-noise trace.
  - Cat, 3 oracles: standardised SNR 1.83, plain 0.95. Its calibration fails: held-out s1 scores standardised E −0.18.
- Post-hoc diagnosis (Amendment A6; `src/analysis/label_diag.py`):
  - The retrain's embedding row for the removed label is its random initial value: relative movement ~1e-7, against ~0.97 for trained rows.
  - Removed-label queries therefore have a heterogeneous floor: max/min oracle-pair distance 16× for automobile, 4.4× for cat.
  - Null-label queries have a homogeneous floor (≤ 1.2×). For automobile they give a calibrated audit: plain SNR 3.74; held-out retrain E 1.00/1.01/0.99; test set 3.47; near/far 1.14/1.07. For cat the null-label trace is weak (1.21).
- Interpretation: `paper/RESEARCH_JOURNAL.md` §7.12–7.13.
- Next change: confirmatory study C2 on a fresh class (ship) for the null-label claim.

## 2026-11-22 — C2 (confirmatory) — oracles for class 8 (ship), seeds 0, 1, 2 — LAUNCHED
- Registered in `PREREGISTRATION.md` ("Confirmatory study C2") before any C2 data. Scoring code: `src/analysis/label_diag.py`, unit-tested (`test_score_c2_rules`).
- Kernel v7; same dataset version and code `827e2688a52f` (the oracle stage is class-generic). `UA_ORACLE_JOBS=8:0,8:1,8:2`, budget 7.5 h. Process 0 trains 8:0 then 8:2; process 1 trains 8:1.
- Pushed about 11:03 IST. A collect-only watcher (`scratchpad/kaggle_collect_c2.sh`) copies only the class-8 files into `results/`.
- Expected: about 6.5 wall-h and about 6.7 GPU-h of quota, leaving about 9 h. The predictions are C2-0 … C2-G.
- Observed (scored 2026-11-23 ≈ 13:30 IST with the registered code, unchanged; `results/figures/label_diag.json`, key `class8`):
  - Runner exit codes [0, 0], 6.51 h; quota ≈ 9.5 GPU-h left until 2026-11-28. The first download attempt failed; the second (18:40, 22 Nov) succeeded.
  - Sanity: the ship row moved 2.1/3.3/3.4e-6 (relative ≈ 1e-7) vs ≈ 0.97 for trained rows. Held.
  - **C2-0 held** (null forget-train plain SNR 1.85). **C2-A held** (held-out plain E 0.94 [0.86, 1.03], 1.13 [1.04, 1.22], 0.94 [0.87, 1.02]).
  - **C2-B FAILED**: null-label max/min pair ratio 1.24, 1.15, 1.06, 1.04, 1.11 at t = 10…300, then **1.42 at t = 400** (threshold 1.3).
  - **C2-C held** (test SNR 1.54, 17 % below train). **C2-D held** (near 1.08, far 1.03). **C2-E held** (peak 2.06 at t = 200 vs 0.75 at t ≥ 700).
  - **C2-F held** (forget-label pair ratio 3.23 at t = 200). **C2-G held** (forget-label held-out std E 1.81, 1.14, −1.92; plain NaN, 0.61, −0.78).
  - Per the registered rules: claim (a) confirmed; claim (b) **not confirmed as registered**, because C2-B is one of its four conjuncts. Calibration (C2-A), the operational part, held.
  - Unregistered observation: null-label *standardised* E is not calibrated on ship (0.58, 1.33, 0.57). Plain E is the statistic that calibrates on both classes.

## 2026-11-23 — C3 (confirmatory) — consequence for audited models (C3-A) and a 4th class, truck (C3-B) — LAUNCHED
- Registered in `PREREGISTRATION.md` ("Confirmatory study C3") at ≈ 14:35 IST, before any C3 data. The C3-A amendment (plain-E fallback when the forget-label std SNR < 1.2, i.e. automobile) was added at ≈ 14:50, also before any C3-A data.
- Scoring: `src/analysis/c3_consequence.py` (C3-A) and `label_diag.score_c3b` (C3-B), both unit-tested (`test_score_c3a_rules`, `test_score_c3b_rule`). The suite is 26/26.
- **C3-B:** kernel v8 pushed 14:36 IST; `UA_ORACLE_JOBS=9:0,9:1,9:2`, budget 7.0 h, same code `827e2688a52f`. Truck was drawn with `default_rng(20261123)` from {0, 2, 4, 5, 6, 7, 9}. Watcher: `scratchpad/kaggle_collect_c3b.sh`.
- **C3-A unlearning cells** (5 methods × seeds 0, 1 × classes 8, 1; plus class 9):
  - They run locally on the M4 GPU under `caffeinate` (`scratchpad/c3/queue_local.sh`). Local MPS is ≈ 1 s/step, ≈ 35 min per 2000-step cell.
  - The first timing attempt ran on the CPU, because `kaggle_env.torch_device` knows only cuda/cpu. The local wrapper now patches in MPS. The timing cells were deleted and are not scored.
  - When v8 is collected, `scratchpad/c3/chain_c3a_kaggle.sh` stops the local queue after its current cell and pushes kernel v9 (budget 2.4 h, deadline-aware) with every remaining cell. Its outputs are copied into `results/C3_unlearn` and marked `DEVICE=kaggle_cuda`.
  - The splits Kaggle regenerates were checked identical to the local ones for classes 1 and 8.
- Quota before C3: ≈ 9.5 GPU-h (refresh 2026-11-28). v8 ≈ 6.6 h, v9 ≤ 2.5 h.
- Observed (C3-B, 2026-11-24 ≈ 00:37 IST):
  - v8 COMPLETE in 6.84 GPU-h (exit codes 0, 0). Truck retrains are in `results/checkpoints/ddpm_cifar10_retain_c9_f0.1_s{0,1,2}.pt`.
  - Forget row untrained (1.8–3.2e-6). Null-label LOO plain E 1.08, 0.97, 0.96, every CI containing 1. Removed-label std E 0.56, −0.95, 0.86.
  - C3-B-B held, with a max null pair ratio of 1.49 at t = 10. **Claim (b) confirmed on truck.** The full table is in `PREREGISTRATION.md` → "C3-B outcome". Output: `results/figures/label_diag.json` → `class9`.
- Observed (C3-A cells): all 30 cells done (3 classes × 5 methods × 2 seeds).
  - 11 ran locally on M4/MPS: 27–55 min each, ≈ 0.8–1.6 s/step.
  - 19 ran on Kaggle v9: 1.64 GPU-h, 19/19 cells, exit codes 0, 0.
  - Quota left ≈ 1 GPU-h until 2026-11-28.
- Observed (C3-A scoring, 00:37–02:40 IST locally, ≈ 40 min per class):
  - **All 12 registered predictions held** (C3-A1…A4 × ship, automobile, truck).
  - Resets passing the removed-label audit: 6/20 on ship, 9/20 on automobile (plain E, per the amendment), 12/20 on truck.
  - Reset removed-E SD: 4.77, 0.72, 3.03.
  - Max method null-E lower CI: −0.85, −0.29, 0.16.
  - Label annealing null E: −1.87/−1.87, −0.79/−0.81, −1.33/−1.34.
  - Exploratory: at the sweep hyper-parameters, GA and VEWC collapse; GD collapses on the removed label; LA and RF barely forget (gated rate 0.89–1.00).
  - Full table: `PREREGISTRATION.md` → "C3-A outcome". Output: `results/figures/c3a_consequence.json`.

## 2026-11-24 — C3-K (EXPLORATORY, not registered) — is ship knowledge still in a reset model? Row-only re-attachment
- Why: C3-A1 shows zero-step embedding resets passing the removed-label audit for ship, while the null-label audit scores them 0. That 0 holds *by construction*, since a reset does not touch the null pathway. This test is not tautological.
- Design (`src/analysis/c3_knowledge.py`):
  - Freeze the network and optimise only the forget-label embedding row (256 numbers), using Adam with lr 0.05, 400 steps and batch 64.
  - Data: 1,000 ship train images, disjoint from the 256 audit images.
  - Models: three passing resets (seeds 102, 109, 110) and the three ship retrains.
  - Then measure the gated ship rate of 200 samples (DDIM 50, guidance 2.0, seed 4242).
  - Also measure the unconditional (null-label) ship rate of 1,000 samples, for the original against each retrain.
- Predictions, written before running (exploratory; the thresholds are guesses):
  - (K1) after re-attachment, every reset reaches a ship rate ≥ 0.8;
  - (K2) every retrain stays ≤ 0.5 after the same re-attachment;
  - (K3) the original's unconditional ship rate exceeds every retrain's.
- If K2 fails (a retrain's network can be steered to ships through one row), the "knowledge survives" reading of C3-A1 is weaker, and that will be reported.
- Observed (2026-11-24, 02:40–03:14 IST, local M4/MPS, 34 min; `results/figures/c3_knowledge_c8.json`). Gated ship rate of 200 ship-label samples, before → after re-attaching only row 8:
  - re-drawn models: reset 102 0.000 → **0.985**; reset 109 0.005 → **0.980**; reset 110 0.000 → **0.980**. The original's rate is 0.995.
  - retrains: s0 0.005 → 0.180; s1 0.005 → 0.125; s2 0.000 → 0.190.
  - Unconditional ship rate (1,000 samples): original 0.083; retrains 0.019, 0.013, 0.015. A re-drawn model's null pathway is identical to the original's, so its rate is 0.083.
  - No degenerate samples in any after-measurement.
- **K1, K2 and K3 all held** (exploratory).
  - A re-drawn model that passed the removed-label audit regains full ship generation when 256 numbers are re-fit. The ship knowledge was never removed.
  - A retrain given the same 256-number re-fit reaches only 12.5–19 %. One row can steer a ship-naive network part of the way, which resembles textual inversion, but the retrains never get close.
- Limits: 1 class (ship); 3 re-draws; one re-fit budget (400 steps, lr 0.05). A larger budget might raise the retrains' rate. Not registered.
- **Extension (written 03:20 IST, before running):** the same test on automobile (1) and truck (9). Re-draws are the first three C3-A1 passers of each class: (100, 104, 106) and (100, 101, 104). The predictions K1–K3 are unchanged.
- Observed, extension (03:20–04:22 IST, local; `results/figures/c3_knowledge_c{1,9}.json`). Gated forget-class rate, before → after re-fitting one row:
  - **automobile:** re-draws 100 / 104 / 106: 0.000 / 0.010 / 0.035 → **0.985 / 0.985 / 0.990**. The original's rate is 0.985. Retrains: 0.010 / 0.000 / 0.000 → 0.085 / 0.150 / 0.190. Unconditional rate: original 0.030; retrains 0.003 / 0.007 / 0.006.
  - **truck:** re-draws 100 / 101 / 104: 0.000 / 0.005 / 0.000 → **0.960 / 0.975 / 0.955**. The original's rate is 0.975. Retrains: 0.015 / 0.035 / 0.000 → 0.155 / 0.245 / 0.155. Unconditional rate: original 0.029; retrains 0.005 / 0.002 / 0.006.
  - **K1, K2 and K3 held on all three classes:** 9 / 9 re-draws reach ≥ 0.955; 9 / 9 retrains stay ≤ 0.245; the original's unconditional rate is 3.4–14× each retrain's.
