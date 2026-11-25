# Pre-registration — Did it really forget? Auditing diffusion and LLM unlearning

**Project:** `unlearning-audit`
**Author:** Arush Badhe
**Date frozen:** 2026-10-13
**Status:** Frozen before the first unlearning run. Not edited after the freezing commit; changes go in the dated "Amendments" appendix.

## 1. Problem statement

Determine whether published approximate-unlearning methods erase the influence of forgotten data or merely suppress it, and identify the property of the parameter update that explains the difference.

## 2. Research question

For class/concept unlearning in diffusion models and example-level unlearning in small LLMs, how far does each method move the model toward the retrain-from-scratch oracle, as measured jointly by membership inference, influence/attribution estimates and relearning speed, and which mechanistic property of the unlearning update explains the gap?

## 3. Hypotheses and pre-specified numeric tests

### H1 — Suppression, not erasure
- **Confirmed if:** for gradient-ascent and label-annealing methods, MIA AUC on the forget set is ≤ 0.55 while linear-probe accuracy on the forgotten class remains ≥ 80 % of its pre-unlearning value and relearning steps-to-90 % are ≤ 1/5 of the oracle's, in at least 2 of 3 seeds.
- **Refuted if:** probe accuracy falls below 50 % of original, or relearning speed is within 2× of the oracle.

### H2 — Influence survives
- **Confirmed if:** the mean attribution score of forget examples on retain generations drops by less than 50 % under approximate unlearning versus more than 90 % under retraining, with non-overlapping 95 % CIs.
- **Refuted if:** the approximate drop is within 10 percentage points of the retrain drop.

### H3 — Low-rank mechanism
- **Confirmed if:** the top-5 singular values capture more than 70 % of ‖Δθ‖_F for approximate methods versus less than 30 % for the oracle delta, and Spearman(effective rank, relearning steps) ≥ 0.7 across method × seed cells.
- **Refuted if:** |ρ| < 0.3, or the spectra are indistinguishable.

### Pre-specified secondary analyses (not pass/fail)
- Position of the variational/Bayesian method relative to approximate and oracle on each of the three tests (prediction: in between).
- f-INE-style test: fraction of forget examples whose attribution score after unlearning exceeds the seed-to-seed null at FDR 0.1 (full budget only; needs ≥ 3 oracle retrains).
- Sensitivity of the attribution ranking to Hessian damping λ ∈ {1e-3, 1e-2, 1e-1} and to LoRA-subspace vs full-parameter gradients.
- Unconditional-sample drift of forget-class samples (diffusion track).
- Per-layer localisation of ‖Δθ_ℓ‖ (DisCEdit/ModHiFi-style component view).

## 4. Expected-to-fail control

The **"do nothing" baseline** (original checkpoint, zero unlearning steps) is run through the full audit on every harness version. It must report: MIA AUC clearly above 0.5 on the forget set, probe accuracy equal to its own pre-unlearning value, relearning steps = 0, attribution drop = 0 %, Δθ = 0. Any deviation halts experiments until fixed.

## 5. Models, data and fixed settings

| Item | Value |
|---|---|
| Diffusion base | Small UNet DDPM, CIFAR-10 32×32, 1000-step linear schedule; ≈ 6 T4-h (3 h in minimum version) |
| Diffusion forget target | one class (class index fixed to 3, "cat") |
| LLM base | Qwen2.5-0.5B + LoRA rank 16 (alpha 32, dropout 0.05) fine-tuned on a TOFU-style fictitious-author set |
| LLM forget target | fictitious-author subset at 1 % / 5 % / 10 % |
| Substitutes if too slow | CelebA 32×32 (attribute as forget target); SmolLM-135M |
| Methods | gradient ascent; gradient difference; label annealing (UAI 2025 style); variational/Bayesian (TMLR 2026 style); NPO-style (LLM only); retrain oracle |
| Matched compute | identical number of gradient steps per approximate method, fixed per track and reported |
| Hessian damping | λ = 1e-2 (default), sensitivity sweep as above |
| Gradient projection | random projection to 4096 dims, fixed projection seed |
| Diffusion MIA timesteps | t ∈ {50, 200, 400, 600, 800} |
| Relearning | fine-tune on 5 % of the forget set, report steps to 90 % of pre-unlearning forget performance |
| Linear probes | logistic regression on frozen mid-UNet / mid-transformer features |
| Seeds | {0, 1, 2} across methods; oracle retrains at seeds {0, 1, 2} (+ {3, 4} for the f-INE null if budget allows) |

## 6. Statistics

- Mean plus 95 % bootstrap CI (10,000 resamples) over seeds for every figure number.
- MIA AUC CIs by bootstrap over examples; relearning curves paired by seed.
- Spearman ρ for H3 computed over all method × seed cells (≥ 12 cells in the full version; reported as preliminary if < 9).
- Non-overlapping CIs is the pre-specified criterion for H2; no further testing is added post hoc.

## 7. What will be reported regardless of outcome

A negative-results section with each hypothesis marked confirmed / refuted / under-powered and the mechanism now believed; the audit library release is independent of the outcome.

## 8. Amendments

Format: `YYYY-MM-DD — what changed — why — git commit`. All amendments below were made **before any experiment was run** (Week 0), following an independent review recorded in [`REVIEW.md`](docs/history/REVIEW_2026-10-18.md).

- 2026-10-18 — **H3 restricted to full-parameter tracks and made a partial correlation.** Old: Spearman(effective rank, relearning steps) ≥ 0.7 across all method × seed cells. New: the test runs only on the DDPM (full UNet weights) and on a new full-fine-tune LLM arm (Qwen2.5-0.5B, 8-bit Adam); the correlation is the partial Spearman correlation of effective rank with relearning steps controlling for ‖Δθ‖_F, same ≥ 0.7 / < 0.3 thresholds. — Why: under LoRA rank 16, every Δθ (including the oracle's) has rank ≤ 16 per layer by construction, so the spectrum comparison is meaningless there; and a smaller update relearns faster regardless of rank (From Dormant to Deleted, arXiv 2505.22310; FRAG, arXiv 2608.25429), so norm is a confound. — commit: this one.
- 2026-10-18 — **Added independent variable: LLM parameterisation** {LoRA r16; full fine-tune with 8-bit Adam}, and **dependent variable: update norm ‖Δθ‖_F** (per layer and total). — Why: needed for the new H3. — commit: this one.
- 2026-10-18 — **H2 promoted to headline; new pre-specified secondary analysis:** Spearman correlation between surviving retain-set influence of forget examples and relearning steps-to-90 % across method × seed cells. — Why: the influence axis is the part of the audit not found in prior work (nearest: GUDA, arXiv 2601.22651, which uses unlearning to approximate attribution, the reverse direction); tying it to relearning connects H2 to H3. — commit: this one.
- 2026-10-18 — **H1 reframed as a cross-modality replication with stronger MIA baselines.** Thresholds unchanged. MIA is now reported as loss-based, LiRA-style and class-geometry MIA (CMIA; Unlearning Isn't Forgetting, ICML 2026), not loss-based alone. — Why: H1 is established for LLMs (Hu et al. ICLR 2025; Łucki et al. TMLR 2025; Unlearning Isn't Deletion, arXiv 2505.16831) and for classifiers (An Illusion of Unlearning?, AISTATS 2026; From Dormant to Deleted), so it is a replication on a class-unlearned DDPM, and a loss-MIA-only "AUC ≈ 0.5" baseline would be called a strawman. LiRA and CMIA are full-version only. — commit: this one.
- 2026-10-18 — **Diffusion base model rescaled** from "small UNet DDPM ≈ 6 T4-h" to a class-conditional UNet of 5–10M parameters trained for tens of thousands of steps, FID expected in the tens and reported as such. — Why: the original DDPM (35.7M params, 800k steps) took ≈ 10.6 h on a TPU v3-8 (Ho et al. 2020), far beyond 6 T4-hours; a relative audit against an oracle trained identically does not need a competitive FID. — commit: this one.
- 2026-10-18 — **f-INE-style seed-null hypothesis test moved to stretch** (secondary analysis 2; oracle seeds {3, 4}). — Why: it needs 3–5 oracle retrains and is the most expensive arm; the headline tests do not depend on it. At minimum scale the attribution drop is reported with bootstrap CIs over retain queries and the absence of a seed null is stated. — commit: this one.
- 2026-10-18 — **LLM track built on OpenUnlearning** (NeurIPS 2025 D&B, arXiv 2506.12618) rather than re-implemented. — Why: it already packages TOFU, GA, GradDiff, NPO and MIA metrics; the repository's added value is the diffusion arm, the influence axis and the checkpoint-pair interface. — commit: this one.
- 2026-10-18 — **Attribution caching protocol made explicit:** per-example gradients computed on a training-set subset at a few timestep draws, random-projected to 4,096 dims before storage. — Why: a 35M-parameter UNet gradient is ≈ 143 MB in fp32; whole gradients cannot be stored. — commit: this one.
- 2026-10-18 — **dtype note:** Qwen2.5-0.5B LoRA cells run in fp32 on T4. — Why: T4 has no bf16 and Qwen2.5 in fp16 can produce NaNs; 0.5B fits in fp32. — commit: this one.
- 2026-10-18 — **Citations removed from the proposal:** *Adapt then Unlearn* (TMLR 2025; a GAN paper), the text-guided subset-selection paper (TMLR 2026; about simplicity bias), STAMP (ICML 2025; dataset-contamination detection, not unlearning). Title corrected: *Partially Blinded Unlearning: Class Unlearning for Deep Networks from Bayesian Perspective* (AAAI 2025). — Why: see REVIEW.md §1. — commit: this one.
- 2026-10-20 — **Diffusion backbone fixed: ~5–10M-param class-conditional UNet (base 64, mults (1,2,2), one 16×16 attention block), classifier-free guidance with a null class (label dropout 0.1), DDIM-50 sampling at guidance 2.0.** — Why: the pre-registration left the architecture open; this is the smallest model that trains to FID in the tens in a free-tier budget. ‖params‖ is printed at train time; trim `base_channels` to 48 if a cell exceeds 10M. — commit: code drop.
- 2026-10-20 — **`label_annealing` and `variational_ewc` are our readings of Panda et al. (UAI 2025) and the TMLR 2026 variational method, flagged as interpretations in `src/diffusion/unlearn.py`.** Label annealing: train on forget-class images with the class label annealed from the true class toward the null class over the matched steps, plus retain descent. Variational: forget ascent + diagonal-Fisher-weighted L2 to the original parameters (Fisher on retain). — Why: neither paper's code was available at authoring time; replace with the authors' implementations when obtained, as a dated amendment. — commit: code drop.
- 2026-10-20 — **Influence (H2) is TRAK-style (Park et al. 2023) without the model ensemble**: per-example gradients at 3 fixed diffusion timesteps / per-example LLM gradients, random-projected to 4096 dims (fixed-seed Rademacher), score φ(zᵣ)ᵀ(ΦᵀΦ+λI)⁻¹φ(z_f), λ = 1e-2. The statistic is the mean |score| of forget examples on retain queries and its drop vs the oracle. — Why: concretises "gradient-based influence"; the single-model TRAK is what a free-tier budget allows. — commit: code drop.
- 2026-10-20 — **MIA (loss-based, SecMI, offline-LiRA with the oracle as OUT, class-geometry MIA) is specified as an interface but NOT implemented in this drop** (`src/audit/mia.py`). — Why: time-boxed; H1 is marked inconclusive until it is written. All other axes are complete. — commit: code drop.
- 2026-10-20 — **Offline LiRA uses the oracle retrains as the OUT distribution**; **probe = L2 logistic regression on mid-block features at t=200**; **relearn fine-tunes on 5% of the forget set at lr 1e-4, steps-to-90% of the pre-unlearning forget-class generation rate**; **spectrum skips tensors with either dimension < 8 and reshapes conv kernels to out×(in·k·k)**. — Why: pin the remaining free choices so the pre-registration is executable. — commit: code drop.
- 2026-10-20 — **NPO implemented as Zhang et al. (2024): minimise −(2/β)·E_forget[log σ(−β(logπ_θ − logπ_ref))], reference = adapters-disabled model (LoRA) or a frozen copy (full-FT), plus retain NLL.** — commit: code drop.
- 2026-10-20 — **MIA non-members are a seeded 5% held-out slice of the retain split, excluded from both base and oracle fine-tuning** (`data.heldout_frac`, indices recorded in `results/llm_checkpoints/heldout_indices_*.json`). — Why: a membership-inference attack needs non-members the model never saw; carving them from retain (not forget) keeps members and non-members on the same distribution so the attack measures memorisation, not class. — commit: llm-audit.
- 2026-10-20 — **LLM audit runner implemented** (`src/llm/audit.py`): evaluate_llm, mia_suite_llm (loss NLL, Min-K% prob, offline LiRA, two-centroid CMIA), attribution_llm (projected per-example answer-NLL gradients; LoRA params for adapters, a fixed 2M-coordinate random subset for full-FT), probe (logistic regression on answer-token hidden states), relearn (fine-tune on 5% of forget, ROUGE-L recovery), spectrum (LoRA B@A delta, flagged `lora:true` and excluded from H3; full state-dict delta for full-FT). — commit: llm-audit.

### Amendment 2026-11-18 — design review before the first GPU run

Made after a design review on 2026-11-18, **before any base model, oracle or unlearning run existed** (`results/` empty; `EXPERIMENT_LOG.md` had only the template). Hypothesis *claims* are unchanged; their operationalisations are corrected where the old ones could not discriminate. The old statistics are still computed and reported as secondary readouts.

- 2026-11-18 — **Two explicit forget regimes.** `forget_frac` is a fraction of the 50k training images, so 0.01/0.05 remove 500/2,500 *random* cats while the other cats stay in the retain set; only 0.10 removes the whole class. *Class regime* = `forget_frac 0.10` (oracle never sees a cat); *subset regime* = `forget_frac 0.05` (oracle still generates cats). H1 and the relearning part of H3 are evaluated in the class regime only: in the subset regime the oracle's forget-class generation rate is already at the pre-unlearning level, its steps-to-90% are 0, and the relearn ratio is undefined. The forget-size ablation is now a regime comparison, not a dose–response. — Why: the old design mixed class-level and example-level unlearning and divided by an oracle relearn time of 0. — commit: amendment-2026-11-18.
- 2026-11-18 — **H2 re-operationalised: oracle-normalised closeness E** (`src/audit/fingerprint.py`). For per-example eps-MSE losses at fixed (timestep, noise) on a query set, d(a,b) = RMS difference and E(u) = (d(orig,O) − d(u,O)) / (d(orig,O) − d_floor), d_floor = distance between two oracle seeds. E = 0: no closer to the retrain than doing nothing; 1: as close as an independent retrain; < 0: moved away (over-forgetting / collateral damage). Computed on four fixed query sets: forget members, held-out forget-class images, retain images of near classes (bird, deer, dog, frog, horse) and of far classes (vehicles). *Confirmed* (per regime, per approximate method) if E_near < 0.5 with its 95% CI below 0.5 while E_forget − E_near ≥ 0.25 with non-overlapping CIs. *Refuted* if E_near ≥ 0.8. — Why: the old statistic, mean |TRAK score| of forget examples on retain queries evaluated at each checkpoint, estimates the effect of *adding* the example at that checkpoint; it is defined and non-zero for a model that never saw the example, and is expected to *grow* at the oracle (forget examples unfitted → larger gradients). "> 90% drop under retraining" was not a prediction the oracle could meet. E is the indistinguishability-from-retrain definition of unlearning applied to a functional, in the spirit of KLOM (Georgiev et al. 2024). Secondary: correlation of the retain-query loss change with the influence-function removal prediction Σ_f τ₀(f,q) at the original model; corr(oracle change, prediction) is reported as a validity check of the attribution tool. The old `attr_drop` is still reported.
- 2026-11-18 — **Second oracle seed moved from stretch into the core design** (class regime: seeds 0 and 1). — Why: without a seed-to-seed floor no distance to the oracle can be called a gap (it is the f-INE premise); E needs it. Oracle seeds use an initialisation stream distinct from the base model's so both are exchangeable replicates.
- 2026-11-18 — **Forget set fixed per fraction (`data.split_seed = 0`), independent of the run seed.** Run seeds vary training/unlearning randomness only. — Why: with per-seed forget sets each seed needed its own oracle and oracle seeds were not replicates of one counterfactual.
- 2026-11-18 — **H1 reference point: the oracle, not only the pre-unlearning model.** H1's probe criterion now additionally requires the unlearned model's probe (balanced accuracy, forget class vs rest) to exceed the oracle's: a DDPM that never saw cats may still separate cat images linearly, so "probe ≥ 80% of the original" alone is not evidence of suppression. Thresholds otherwise unchanged; relearning ratio uses the same-seed oracle. Censoring: an approximate method that relearns while the oracle never reaches 90% within the budget has ratio 0.
- 2026-11-18 — **H3: matched-steps `retain_finetune` control added and alignment measured.** `retain_finetune` = plain descent on retain data with the same steps and lr, never touching the forget set. H3's spectrum criterion additionally requires approximate methods' top-5 energy to exceed the control's by ≥ 0.2. — Why: θ_oracle − θ_base is the difference of two independent 40k-step runs and is full-rank by construction, so "approximate > 70% vs oracle < 30%" would pass for any short fine-tune. Secondary readout: relearn alignment a = cos(g_f(θ_u), Δ), with the same norm-controlled partial Spearman; in a linearised model relearning speed depends on how much of Δ the forget gradient points back through, of which low rank is one cause. Secondary intervention: per-layer rank-k truncation of Δ at matched ‖Δ‖_F (k ∈ {1, 4, 16, 64}) for the gradient-ascent cells, measuring forgetting and relearning vs k (`src.run truncate`). Relearning times that never reach the target are right-censored and tied at the maximum in rank statistics, not dropped.
- 2026-11-18 — **Bug fixes found in review (no data affected; none had been run):** (1) the sweep key `seed` set only `train.seed` while unlearning/audit read `unlearn.seed` (every seed would have run seed 0), and `forget_frac` never reached `data.forget_frac` (the oracle retrain's split); (2) oracle checkpoints for different forget fractions shared one path; (3) the diffusion configs had no `audit.mia` block (KeyError at the first audit); (4) oracle cells skipped the audit, so `relearn_ratio_to_oracle` was never set and H1 could not be confirmed; (5) the probe passed the true class label into `features()`, so the class embedding leaked the label into the probed features — now null-class conditioned, as CMIA already was; (6) NPO's LoRA reference was `disable_adapter()` (raw pretrained Qwen, never fine-tuned on TOFU) — now a frozen copy of the fine-tuned base; (7) TRAK projection for the diffusion track switched from a dense Rademacher matrix regenerated per batch to a cached CountSketch (sparse JL; inner products preserved in expectation), for runtime.
- 2026-11-18 — **Utility readout added:** retain-class conditional sample accuracy (classifier on 50 samples per retain class); FID stays optional.
- 2026-11-18 — **Platform:** runs on Google Colab T4 (free tier) via the Colab CLI instead of Kaggle; checkpoints are downloaded to the local machine after every stage because free sessions can be reclaimed.

### Amendment A3 2026-11-19 — after the class-regime sweep (POST-HOC; everything below is exploratory)

Made **after** the 14 class-regime cells of run v4 existed, and before any of the new data. These changes are post-hoc. Results obtained under them are labelled *exploratory* and are kept apart from the pre-registered tests, which are reported as they came out (`paper/RESEARCH_JOURNAL.md` §7).

- 2026-11-19 — **Operating-point comparison** (`configs/diffusion_oppoint.yaml`, `src/diffusion/oppoint.py`).
  - What:
    - *Trajectories.* Seed 0 traces (cat rate, retain accuracy) every 200 steps for a fixed grid: gradient ascent at lr {3e-6, 1e-5, 3e-5}; gradient difference at lr {1e-4, 3e-4}; variational EWC at lr 3e-5 with ewc_weight {1e2, 1e3, 1e4}. Each run is 2,000 steps.
    - *Selection rule (fixed here, before any trajectory).* Target: cat rate ≤ 0.10 and retain accuracy ≥ 0.90. Take the smallest step meeting both, then the higher retain accuracy, then the lower lr. Otherwise take the lowest cat rate with retain accuracy ≥ 0.90, flagged `reached_target: false`. Otherwise the method is reported as having no utility-preserving operating point.
    - *Audit.* The selected points are audited with seeds 0 and 1.
  - Why: at the matched budget, ascent-type methods drove retain generation to chance and the others barely forgot. Method-level ladder claims need a common operating point.
- 2026-11-19 — **New exploratory method `embedding_reset`.** It re-draws the cat row of the class-embedding table from N(0, 1), with zero gradient steps. Why: run v4 found that the oracle differs from the original in conditional generation but not in per-example losses, which suggests "forgetting the class" lives in the never-trained class embedding. This method tests that directly.
- 2026-11-19 — **TRAK attribution disabled for new cells** (`audit.attribution.enabled: false`). Why: the pre-registered validity check gave corr(TRAK prediction, oracle change) ≈ −0.04 in both seeds, and TRAK took 40 of 52 minutes per cell. The 14 class-regime records keep their TRAK readout. The signed oracle-change correlation is still computed.
- 2026-11-19 — **Harness fix: the do-nothing probe check** is changed from accuracy ≥ 0.8 (absolute, set before data) to ≥ 0.6 (above the balanced-class chance level of 0.5). Why: the original model's probe is 0.73 / 0.76 and the oracle's is 0.73 / 0.77, so the old check failed for reasons unrelated to forgetting. The MIA check is unchanged; its failure was predicted (P1) and is reported.
- 2026-11-19 — **Post-hoc readout:** $d(u,O)/d_\text{floor}$, the distance to the oracle in units of the seed-to-seed floor, is reported where $E$ is undefined by the SNR guard. It is not a pass/fail test.
- 2026-11-19 — **LLM protocol fixes before any LLM data:** the held-out MIA non-member split now uses a fixed seed (it was seed-dependent), and GradDiff's retain term now excludes the held-out slice. E is ported to answer-token NLL, with 2 oracle seeds.

### Amendment A4 2026-11-21 — timestep decomposition and standardised closeness (POST-HOC; exploratory)

- **What.**
  - The original-vs-oracle loss difference is decomposed per timestep (12 timesteps, 10–990) and per conditioning (true label vs null), from the v4 checkpoints (`src/diffusion/timestep_decomp.py`).
  - A timestep-standardised closeness `standardized_closeness_np` is added. It is `closeness_np` after dividing each timestep column by its oracle-pair floor.
  - Twenty independent `embedding_reset` draws are evaluated.
- **Why.** In v4 the plain $E$ was undefined on every query set (SNR ≈ 1), while conditional generation differed completely.
- **Status of the readouts.**
  - Predictions A4-a…e were written in `paper/RESEARCH_JOURNAL.md` §3 before each computation.
  - $E_\text{std}$ is reported **alongside** the pre-registered $E$, never instead of it. The pre-registered H2 verdict stays "untestable".

### Amendment A5 2026-11-21 — validity gate on classifier-based behaviour metrics (POST-HOC; measurement fix)

- **Finding that prompted it.**
  - Sampled images from the v4 checkpoints show that gradient difference **seed 0**'s cat-conditioned samples are saturated noise (100 % flagged by the gate), yet the CIFAR classifier labels 65 % of them "cat". Seed 1 still generates real cats (0 % degenerate, 76 % cat). Gradient ascent and variational EWC produce 100 % noise for every class, labelled mostly "frog".
  - Gradient ascent's noise is labelled "frog", which made it look like it forgot.
  - Classifier-based rates therefore do not measure forgetting on degenerate outputs.
- **Fix.**
  - A sample is *degenerate* if its high-frequency energy (mean squared neighbour difference) exceeds 0.1267. That is the 99.9th percentile over 5,000 real CIFAR-10 training images (max 0.158). Observed degenerate samples sit at 0.36–0.46.
  - Every behaviour metric now records `raw`, `valid` (classified as the class AND not degenerate) and `degenerate`.
  - Decisions use `valid`: the relearning recovery criterion, the truncation forgetting rate, and the A3 operating-point selection.
- **Scope.**
  - Run v5 was launched before this fix. Its A3 selection and truncation rates use the raw rate, and are re-scored locally from saved checkpoints where those exist. The trajectories save no checkpoints, so the selection rule's inputs are reported as raw, with this caveat.
  - Pre-registered H1's rung-0 criterion is reported both ways.

---

## Confirmatory study C1 — registered 2026-11-21 ≈ 21:30 IST, before any C1 data exists

**Claim under test.**
- In a class-conditional ε-prediction DDPM in the generalisation regime, an exact retrain without class $k$ differs from the original on images of class $k$ **only at high noise**.
- The unweighted timestep-averaged ε-MSE distance used by loss-based audits is therefore at the seed-to-seed floor. Such an audit cannot tell the original from an exact retrain.
- Dividing each timestep by its own seed-to-seed floor recovers the difference, and it makes the closeness statistic $E$ calibrated: an independent retrain scores ≈ 1.

**Status of the evidence so far.** The claim was found post-hoc on class 3 (cat; Amendment A4). The thresholds below were chosen *after* seeing the cat data, which passes them. **The confirmatory evidence is class 1 (automobile)**, chosen because it is maximally unlike the cat (vehicle vs animal; its near set is the other vehicles), plus a calibration on cats that uses a new third cat oracle never seen before.

**New data.**
- Oracle retrains on CIFAR-10 minus class 1, seeds 0, 1, 2.
- Oracle retrain on CIFAR-10 minus class 3, seed 2.
- Training recipe identical to D2: `configs/diffusion_base.yaml`, 40k steps, Kaggle T4.
- The original model is the existing D1 base; it is not retrained.

**Analysis.**
- Code: `src/analysis/c1_report.py` and `src/diffusion/timestep_decomp.controls`, committed to the working tree before the data.
- Measurements:
  - $n = 256$ images per query set and common random numbers.
  - Per-timestep ratios use 12 timesteps (10–990); the audit statistics use the audit's 5 timesteps (50, 200, 400, 600, 800).
  - With 3 oracles, the floor is the mean pairwise oracle distance and the signal is the mean original-to-oracle distance.

**Predictions (each scored held/failed, separately for class 1 and class 3).**

| ID | Prediction | Threshold |
|---|---|---|
| C1-1 | The trace is localised at high noise | train-set ratio ≤ 1.25 at every $t \le 200$, AND ≥ 1.5 at ≥ 3 of $t \in \{500, \dots, 900\}$ |
| C1-2 | The plain average is blind | plain audit-5 SNR < 1.2 |
| C1-3 | Standardising recovers the trace | standardised audit-5 SNR ≥ 1.4 |
| C1-4 | The trace is class-level, not membership | test-set (never trained on) standardised SNR ≥ 1.4, and within 0.3 of the train-set value |
| C1-5 | No trace on other classes | near and far standardised SNR ≤ 1.2 |
| C1-6 | Standardised $E$ is calibrated; plain $E$ cannot recognise an exact retrain | leave-one-oracle-out: in every rotation the held-out oracle's standardised $E \in [0.6, 1.4]$, while plain $E$ is undefined (SNR < 1.2) in a majority of rotations |

**Interpretation rules.**
- *The headline holds* if C1-1 to C1-3 and C1-6 hold for class 1.
- *Class-specific* if they hold for cats but not for class 1. The paper then reports the cat result as a case study, not a general claim.
- C1-4 and C1-5 are mechanism checks.
- Any failure is reported with its numbers.

**Not part of C1.** These are kept as secondary or exploratory, reported with their labels:
- the method comparison (A3), rank truncation (P5), the subset regime, and the LLM track;
- the behaviour-metric validity gate (A5) and embedding-reset draws (A4-e).

**C1 outcome (scored 2026-11-22 with `src/analysis/c1_report.py`, unchanged since registration; full numbers in `paper/RESEARCH_JOURNAL.md` §7.12, `results/figures/c1_report.json`).**

| | C1-1 | C1-2 | C1-3 | C1-4 | C1-5 | C1-6 |
|---|---|---|---|---|---|---|
| class 1 (automobile, confirmatory) | failed | failed | failed | failed | held | failed |
| class 3 (cat, 3 oracles) | held | held | held | held | held | failed |

By the interpretation rules above, the headline did **not** replicate. The timestep-localised trace is *class-specific*. It also failed calibration even on cats: in one leave-one-out rotation the held-out cat oracle scored standardised E = −0.18. The post-hoc diagnosis follows in Amendment A6.

### Amendment A6 2026-11-22 — why C1 failed: the retrain's embedding for the removed label is untrained (POST-HOC; exploratory)

- **By construction**, an exact retrain never sees class $k$. Its class-embedding row $k$ gets no gradient, and with weight decay 0 it keeps its random initial value: it moved 0.0000 from the seed's initial value in all 6 retrains, while every other row moved about 1.0.
- Every C1 and A4 query conditioned on label $k$. It therefore compared the original's *trained* row $k$ against each retrain's *own random* row $k$. The "oracle" response to label $k$ is a draw-dependent arbitrary vector, not a counterfactual. The same mechanism explains why A4-e embedding-reset closeness was draw-dependent (SD 1.15).
- Evidence, from the same images and checkpoints:
  - Under label $k$ the three retrains disagree heterogeneously. The max/min oracle-pair distance at $t \le 400$ reaches 16.1 for automobile and 4.4 for cat.
  - Under the **null (unconditional) label**, which every model trains through CFG dropout, the ratio is 1.1–1.2 for both classes at every $t \le 400$.
- Null-label audit, exploratory:

  | class | train plain SNR | test plain SNR | near / far | held-out retrain plain E (3 rotations) |
  |---|---|---|---|---|
  | automobile | 3.74 | 3.47 | 1.14 / 1.07 | 1.00, 1.01, 0.99 (CIs within ±0.03) |
  | cat | 1.21 | 1.15 | 1.05 / 1.05 | undefined in 2 of 3 (SNR < 1.2) |

  The trace sits at low and mid noise ($t$ 50–400) and is absent at $t \ge 700$.
- Files: `results/figures/c1_embedding_diag.json`, `c1_null_label_controls.json`, `c1_pairwise_diag.json`. The code is `src/analysis/label_diag.py`, a repo copy of the scratch scripts that produced these numbers.

## Confirmatory study C2 — registered 2026-11-22 ≈ 11:00 IST, before the 11:03 push; no C2 data exists

**Claims under test (from Amendment A6, found post-hoc on classes 1 and 3).**
- (a) For class-removal audits of a class-conditional diffusion model, querying the removed label is ill-posed: the exact-retrain reference's response to that label is seed-specific.
- (b) Querying the null label gives a well-posed counterfactual. It has a homogeneous seed floor, a calibrated closeness statistic when there is a trace, a class-level (not membership) trace localised at low and mid noise, and no trace on other classes.

**New data.**
- Oracle retrains on CIFAR-10 minus **class 8 (ship)**, seeds 0, 1, 2.
- Recipe identical to C1: `configs/diffusion_base.yaml`, 40k steps, Kaggle T4. The original is the D1 base.
- Ship was chosen because it is neither class used to form the claim. Its near set is (0, 1, 9): airplane, automobile, truck.

**Analysis.**
- Code: `src/analysis/label_diag.py`, committed to the working tree before the data.
- Query sets are those of `timestep_decomp.controls` (n = 256, same RNG order, common random numbers).
- Null label = index 10. Audit statistics use the 5 audit timesteps; per-timestep ratios use 12.

**Predictions (class 8; each held / failed / untestable).**

| ID | Prediction | Threshold |
|---|---|---|
| C2-0 | Removing ships leaves a detectable null-label trace (risky: cat did not) | null-label forget-train plain SNR ≥ 1.5 |
| C2-A | Null-label closeness is calibrated | if C2-0 held: in every leave-one-oracle-out rotation the held-out retrain's plain $E \in [0.8, 1.2]$; otherwise untestable |
| C2-B | Null-label seed floor is homogeneous | max/min oracle-pair distance ≤ 1.3 at every $t \le 400$ |
| C2-C | The trace is class-level | null-label forget-test plain SNR ≥ 1.5 and within 20 % of forget-train (untestable if C2-0 failed) |
| C2-D | No trace on other classes | null-label near and far plain SNR ≤ 1.25 |
| C2-E | The trace is at low/mid noise | max snr_t over $t \in \{50, \dots, 400\}$ ≥ 1.5 × max snr_t over $t \ge 700$ |
| C2-F | The removed-label floor is heterogeneous | forget-label max/min oracle-pair distance ≥ 2.0 at some $t \le 400$ |
| C2-G | Removed-label closeness is not calibrated | forget-label leave-one-out: at least one rotation has standardised $E \notin [0.6, 1.4]$ or undefined |

Sanity check, not evidence: the class-8 embedding row moved < 1e-5 of its norm from init in all three retrains. *Amended ≈ 11:10 IST, while the C2 retrains were still training (no C2 data):* the first wording said an absolute 1e-6. Re-initialising locally gives 2–3e-6 on the classes 1 and 3 retrains, against row norm ≈ 16, i.e. about 1e-7 relative. That is float32 round-off between torch builds, against ≈ 0.97 for trained rows.

**Interpretation rules.**
- *Claim (b) is confirmed* if C2-0, C2-A, C2-B and C2-D hold.
  - If C2-0 fails, calibration is untestable on ships. Claim (b) then rests on automobile alone, and trace size is reported as class-dependent.
  - If C2-0 holds but C2-A fails, claim (b) is falsified.
- *Claim (a) is confirmed* if C2-F and C2-G hold.
- C2-C and C2-E are mechanism checks. Every number is reported, whatever the outcome.

### C2 outcome (scored 2026-11-23 ≈ 13:30 IST with the registered code, unchanged)

| ID | Observed | Verdict |
|---|---|---|
| sanity | ship row moved 2.1 / 3.3 / 3.4e-6 (≈ 1e-7 relative); trained rows ≈ 0.97 | held |
| C2-0 | null forget-train plain SNR 1.85 | **held** |
| C2-A | held-out plain E 0.94 [0.86, 1.03], 1.13 [1.04, 1.22], 0.94 [0.87, 1.02] | **held** |
| C2-B | null pair ratio 1.24 / 1.15 / 1.06 / 1.04 / 1.11 / **1.42** at t = 10 / 50 / 100 / 200 / 300 / 400 | **failed** (t = 400) |
| C2-C | null forget-test plain SNR 1.54 (−17 % vs train) | **held** |
| C2-D | null near 1.08, far 1.03 | **held** |
| C2-E | max snr_t 2.06 (t = 200) vs 0.75 at t ≥ 700 | **held** |
| C2-F | forget-label pair ratio 3.23 at t = 200 | **held** |
| C2-G | forget-label held-out std E 1.81, 1.14, −1.92 (plain NaN, 0.61, −0.78) | **held** |

**Applying the interpretation rules.**
- **Claim (a) is confirmed**: C2-F and C2-G held.
- **Claim (b) is not confirmed as registered**: C2-B failed, and it is one of the four required conjuncts. The registered rule is applied as written. We do not relax the 1.3 threshold after seeing data.
- What held inside claim (b): calibration (C2-A), class-level trace (C2-C), specificity (C2-D) and low/mid-noise localisation (C2-E).
- What failed: floor homogeneity at t = 400 only. At t ≤ 300 the ratio is ≤ 1.24. At t = 400 the null-label ratio (1.42) is still well below the forget-label ratio there (2.13).

**Unregistered observation (exploratory).** Null-label *standardised* closeness is not calibrated on ship: held-out E 0.58, 1.33, 0.57, against 0.88, 1.05, 1.02 on automobile. Plain closeness is the null-label statistic that calibrated on both classes.

## Confirmatory study C3 — registered 2026-11-23 ≈ 14:35 IST, before the 14:36 push of kernel v8; no C3 data exists

**Why.**
- C2 confirmed claim (a). Claim (b) was not confirmed as registered: its homogeneity conjunct C2-B failed, while calibration held.
- Two gaps remain.
  - (i) *Consequence.* Does the reference choice change the verdict an auditor reaches about unlearning methods? So far every claim is about retrains only.
  - (ii) *Generality of claim (b)* beyond automobile and ship.

### C3-A — consequence for audited models (local, Apple M4 / MPS, 0 GPU-h)

**Models** (forget class k ∈ {1 automobile, 8 ship}; and k = 9 once C3-B's retrains exist):
- `do_nothing`.
- 20 `embedding_reset` draws: original network, row k ← N(0, 1) with torch generator seeds 100–119, as in A4-e.
- The five approximate methods of `configs/diffusion_unlearn_sweep.yaml` at its registered hyper-parameters (2000 steps, lr 3e-5): gradient_ascent, gradient_difference, label_annealing, variational_ewc, retain_finetune. Unlearning seeds 0 and 1.

**Statistics** (forget-train query set of `label_diag`, n = 256, audit timesteps, the three retrains of class k as references):
- *Removed-label audit*: forget-label standardised E (the A4 statistic; plain E also reported).
- *Null-label audit*: null-label plain E (the statistic that calibrated in C2).
- *Behaviour*: gated forget-class rate of 200 class-k-conditioned samples (DDIM 50, guidance 2.0, sampling seed 4242, A5 gate). The same is computed for the three retrains.
- A model *passes the removed-label audit* if forget-label std E ∈ [0.5, 1.5] AND its gated rate ≤ (max retrain gated rate + 0.05).

**Predictions (per class; each held / failed).**

| ID | Prediction | Threshold |
|---|---|---|
| C3-A1 | The removed-label audit can be passed by an intervention that changes no data-dependent parameter | ≥ 1 of the 20 reset draws passes the removed-label audit |
| C3-A2 | Removed-label closeness of reset draws is draw-dependent | SD over the 20 draws of forget-label std E ≥ 0.3 |
| C3-A3 | No tested approximate method is retrain-equivalent under the null-label audit | no method × seed has null-label plain E lower 95 % CI bound ≥ 0.8 |
| C3-A4 | Label annealing moves the null pathway *away* from the retrain: it trains the null label on forget images | label_annealing null-label plain E < 0 for both seeds |

**Design property, not evidence.**
- Every reset draw leaves the null pathway identical to the original. Its null-label E is therefore exactly 0 ("nothing removed"), the same as `do_nothing`.
- This is consistent with A3-e: embedding_reset relearns fastest of all methods.

### C3-B — fresh-class test of claim (b) (Kaggle, 3 retrains, ≈ 6.7 GPU-h)

**Class choice.**
- Drawn at random *before* any data: `np.random.default_rng(20261123)` over the unused classes {0, 2, 4, 5, 6, 7, 9}. The draw gave **class 9 (truck)**.
- Near set (by the `label_diag` rule): (0, 1, 8).
- Retrain seeds 0, 1, 2 with the same recipe and code as C1/C2.

**Predictions.** These are C2-0 … C2-G with identical thresholds, scored by `label_diag.score_c2`, with one revision, disclosed:
- **C3-B-B (replaces C2-B, revised after C2 data).** At every t ≤ 400, the null-label max/min oracle-pair ratio is ≤ 1.5 AND below the forget-label ratio at the same t.
- The revision is motivated by C2 (1.42 at t = 400). It is therefore tested only on this fresh class. C2-B stays failed for ship.

**Interpretation rules.**
- Claim (b) is confirmed on truck if C2-0, C2-A, C3-B-B and C2-D hold. If C2-0 fails (weak trace, as for cat), calibration is untestable on truck.
- Whatever the outcome, the paper reports the count of classes on which the null-label audit had power (SNR ≥ 1.5) and was calibrated.

*C3-A amendment, ≈ 14:50 IST, before any C3-A data (no unlearning cell has been run; only a 50-step timing cell for ship, which is not scored).*
- For automobile, the forget-label *standardised* SNR on the retrains is 1.18, already known from A6. That is below `closeness_np`'s `min_snr` = 1.2, so forget-label std E is undefined for every automobile model and C3-A1/A2 would be vacuous.
- Rule: where the forget-label std SNR < 1.2, the removed-label audit uses forget-label *plain* E instead. Automobile's plain SNR is 1.22. The class and statistic used are reported for each verdict.
- Ship (std SNR 1.53) keeps std E.

### C3-B outcome (scored 2026-11-24 ≈ 00:37 IST with the registered code, unchanged)
Truck (class 9), retrains s0–s2, code `827e2688a52f`, `label_diag.score_c3b`.

| Prediction | Observed | Verdict |
|---|---|---|
| Sanity: forget row untrained | rows moved 3.19e-6, 3.09e-6, 1.82e-6 (absolute; *[label corrected 2026-11-25: was "(relative)"]*) | held |
| C2-0 null-label power | null plain SNR 2.44 on forget-train (std 1.88) | held |
| C2-A null-label LOO calibration | plain E 1.08 [1.03, 1.13], 0.97 [0.92, 1.02], 0.96 [0.91, 1.01] | held |
| C3-B-B homogeneity (revised rule) | null pair ratios at t = 10…400: 1.49, 1.19, 1.15, 1.16, 1.09, 1.18. Max 1.49 against the 1.5 bound; every value is below the forget-label ratio at the same t (6.56, 5.43, 3.71, 2.40, 2.98, 2.36) | held (thin margin at t = 10) |
| C2-B (original rule, superseded for truck) | as above | failed, reported for completeness |
| C2-C generalises to forget-test | null plain SNR 2.20 | held |
| C2-D near/far null SNR ≈ 1 | 1.01 / 0.91 | held |
| C2-E, C2-F (forget label is not homogeneous) | forget-label max pair ratio for t ≤ 400 is 6.6 | held |
| C2-G removed-label LOO miscalibrated | std E 0.56 [0.48, 0.63], −0.95 [−1.10, −0.82], 0.86 [0.84, 0.88]; plain E NaN, −2.08, NaN | held |

**Claim (b) confirmed on truck** (C2-0, C2-A, C3-B-B and C2-D all held).
- The null-label audit is now calibrated on 3 of 4 classes: automobile, which was exploratory; ship, which was C2 (the calibration part held, homogeneity rule C2-B failed); and truck, which is C3-B. Cat is the exception: it lacks power, with null SNR 1.21.
- Exploratory, not registered: on truck, as on ship, the null-label *standardised* E is less well calibrated than plain E (0.57, 1.15, 0.93). This supports the recommendation to use plain E.

### C3-A outcome (scored 2026-11-24, 00:37–02:40 IST, locally on M4/MPS, with the registered code `c3_consequence.score_c3a`, unchanged)
Output: `results/figures/c3a_consequence.json`. Unlearning cells are in `results/C3_unlearn/` (30 cells: 11 local MPS, 19 Kaggle v9).

**Classes and settings**

| Class | Removed-label statistic | Retrain gated rates | Pass threshold on rate |
|---|---|---|---|
| ship (8) | std E (oracle std SNR 1.53) | 0.005, 0.005, 0.000 | 0.055 |
| automobile (1) | plain E (std SNR 1.18 < 1.2, per the amendment) | 0.010, 0.000, 0.000 | 0.060 |
| truck (9) | std E (oracle std SNR 1.44) | 0.015, 0.035, 0.000 | 0.085 |

**Per-class results**

| Class | C3-A1: resets passing the removed-label audit | C3-A2: SD of reset removed-label E (range) | C3-A3: max method × seed null-E lower CI | C3-A4: label_annealing null E (s0, s1) |
|---|---|---|---|---|
| ship (8) | **6 / 20** (seeds 102, 108, 109, 110, 111, 114) | 4.77 (−15.23 … 1.18) | −0.85 | −1.87, −1.87 |
| automobile (1) | **9 / 20** (100, 104, 106, 109, 110, 113, 114, 115, 116) | 0.72 (0.31 … 2.64) | −0.29 | −0.79, −0.81 |
| truck (9) | **12 / 20** (100, 101, 104, 106, 108, 109, 110, 111, 113, 114, 118, 119) | 3.03 (−9.59 … 1.52) | 0.16 (gradient_difference s1: 0.24 [0.16, 0.32]) | −1.33, −1.34 |

**Verdicts: all 12 registered predictions held** (C3-A1 … C3-A4 × 3 classes). The required classes were ship and automobile; truck was added once its retrains existed.

**Reading**
- On every tested class, a zero-step edit passes the retrain-reference removed-label audit, on both closeness and behaviour. The edit re-draws one embedding row and leaves every weight that saw data untouched.
  - The pass rate is 30 %, 45 % and 60 % of draws on ship, automobile and truck.
  - Whether a draw passes is luck: removed-label E varies over the draws by an SD of 0.72–4.77.
- The null-label audit scores every reset exactly 0. This is a design property, not evidence; see "C3-K" in EXPERIMENT_LOG for the non-tautological follow-up.
- Under the null-label audit, no approximate method at the sweep hyper-parameters is retrain-equivalent.
- Label annealing moves away from the retrain on all three classes, as predicted from its mechanism: it trains the null label on forget images.

**Exploratory observations (not registered)**
- *Collapse.* At the sweep hyper-parameters (2000 steps, lr 3e-5), gradient_ascent and variational_ewc destroy the model on every class: removed-label E ≈ −300 to −3000 and null E ≈ −450 to −1900. Their gated rate of 0 is reached by collapse.
- *Gradient_difference.* It also collapses on the removed label. On the null label it scores between −45.7 and 0.24.
- *Too little forgetting.* Label_annealing and retain_finetune keep generating the forget class (gated rate 0.89–1.00).
- So no tested method reaches retrain-like behaviour without collapse. This limits what C3-A can say about the methods; it is not a limit on what it says about the audit.

### C3-A: corrections to the outcome wording (2026-11-24, after an independent review; verdicts unchanged)
- "leaves every weight that saw data untouched" is wrong. The re-draw *overwrites* the original's trained row k with a random one; every *other* weight is untouched.
- "Whether a draw passes is luck" overstates it. The same 20 torch seeds give the same vectors for every class, and outcomes correlate (ship–truck Spearman ρ = 0.52). Passing is partly a property of the vector.
- Post-hoc analysis (not registered), computed from the existing JSON: under the C3-A pass rule with leave-one-out references, held-out *exact retrains* pass in 3/9 rotations (automobile 0/3, ship 1/3, truck 2/3), against 27/60 for re-draws.
- The removed-label statistic differs by class (std for ship and truck, plain for automobile, as amended before data). The SDs in A2 therefore are not comparable across classes.


---

## Study C4: mechanism, specificity and a published-metric analogue (registered 2026-11-24 12:53 IST, before any C4 data)

**Why.** An independent review of the workshop paper (journal §7.17) ranked these open weaknesses:
- (W7) the untrained row is not shown *causally* to cause the removed-label heterogeneity;
- (W2) the null-label audit is not shown to track *removal of class k* rather than "being an independently trained network";
- (W1) no published audit is shown to be affected.

Since then we found one that is. FID_O (Zheng, Tai & Yeh, CVPR 2026, arXiv:2603.22870) is "the FID between the generated images from the unlearned model and those from the oracle model *for all the classes*", against a single retrain, on CIFAR-10 class-conditional DDPMs. It includes the forgotten label.

**Scope.**
- Laptop only (Apple M4/MPS); forward passes and sampling from existing checkpoints; no training; no Kaggle.
- Every model uses EMA weights, as in all earlier studies.
- Before registration, a timing probe sampled 256 images from the original under label 0 and extracted Inception features. Nothing from it was saved or scored.

### C4-R: row-swap design (mechanism)
**Design.**
- For each class k ∈ {1, 3, 8, 9}, build a 3×3 grid of models M[n, r]: network n ∈ {retrain s0, s1, s2}, with its EMA row k replaced by retrain r's EMA row k. The diagonal is the real retrains.
- Losses: the label_diag pipeline unchanged (`forget_train` queries, 256 images, same noise, label k, all 12 timesteps). d(·,·) is the per-timestep RMS over images.
- Let T≤ = {10, 50, 100, 200, 300, 400}.

**Rules.**
- **R1 (the row dominates).**
  - Row effect: D_row(t) = mean over n, and over pairs r ≠ r', of d(M[n,r], M[n,r']).
  - Network effect: D_net(t) = mean over r, and over pairs n ≠ n', of d(M[n,r], M[n',r]).
  - Held for a class if D_row(t) > D_net(t) at ≥ 4 of the 6 t ∈ T≤.
  - **Held overall if held on ≥ 3 of 4 classes.**
- **R2 (a common row restores homogeneity).**
  - For a fixed row r: H_r = max over t ∈ T≤ of [max/min over the 3 network pairs of d(M[n,r], M[n',r])].
  - H_common = median over r of H_r. H_indep = the same statistic on the diagonal (the real retrains).
  - Held for a class if H_common ≤ 0.5 · H_indep.
  - **Held overall if held on ≥ 3 of 4 classes.**
- **R3 (a common row restores calibration).**
  - For each r, compute the leave-one-out plain E (audit timesteps {50, 200, 400, 600, 800}, closeness_np unchanged, original as θ0) of each held-out network against the other two, all carrying row r. This gives 9 rotations per class.
  - Held for a class if ≥ 2/3 of its *defined* rotations lie in [0.6, 1.4]. Classes with < 3 defined rotations are untestable.
  - **Held overall if held on every testable class, with ≥ 2 testable.**

**Registered interpretation.**
- R1 and R2 held: the independent random draw of the untrained row is the main cause of the removed-label heterogeneity.
- R1 or R2 failed: the heterogeneity comes mainly from network-specific responses to an *untrained input*. The paper's mechanism text must then say "untrained" rather than "independently drawn", whatever happens to calibration.

### C4-X: cross-class retrains (specificity of the null-label audit)
**Design.**
- For k ∈ {1, 8, 9}, the classes with null-label power, take the 9 retrains that removed a *different* class j ∈ {1, 3, 8, 9} \ {k} (3 seeds each). Each is an independently trained network that kept class k.
- Compute each one's null-label plain E against the three class-k retrains, with the original as θ0, using the label_diag queries and noise for class k.

**Rule.**
- **X1: held if all 27 point estimates are < 0.5.**
- Also reported separately: same-seed pairs (retrain(−j, s) vs retrain(−k, s) share an initialisation) vs different-seed pairs.

**Interpretation.** If X1 fails, null-label E partly measures "independent network" rather than "class k removed", and the paper must say so.

### C4-F: removed-label Fréchet distance (analogue of FID_O restricted to the forgotten label)
**Design.**
- Classes k ∈ {8, 1, 9}. Models:
  - the original;
  - the three retrains;
  - the 20 C3-A re-draws (seeds 100–119, constructed exactly as in C3-A).
- Samples: 1,000 per model under label k (C3-A sampler: 50 DDIM steps, guidance 2.0). The sampler seed is independent per model: 50000 + 1000·k + index, with the index in the order above.
- Features: pytorch-fid InceptionV3, 768-d block (full-rank covariance at n = 1,000). Images are mapped from [−1, 1] to [0, 1].
- FD is the Fréchet distance between Gaussian fits.
- Retained-label contrast: the original and the 3 retrains also sample 1,000 images under label 0 (airplane, retained in every retrain).

**Rules.**
- **F1 (single-retrain FD cannot separate re-draws from retrains).** Per class: of the 60 (re-draw, retrain) FDs, the fraction ≤ the largest of the 3 retrain–retrain FDs is ≥ 0.25. **Held overall if held on ≥ 2 of 3 classes.**
- **F2 (retrains disagree more under the removed label).** Per class: the mean retrain-pair FD under label k is > 1.5 × that under label 0. **Held overall if held on ≥ 2 of 3 classes.**
- **F3 (sanity).** Per class: the minimum original–retrain FD under label k is > the maximum retrain–retrain FD. **Held overall if held on every class.**
- Exploratory, not registered: the relationship between a re-draw's FD and its C3-A pass/fail.

**Code.**
- `src/analysis/c4_controls.py` (C4-R, C4-X) and `src/analysis/c4_fd.py` (C4-F).
- Written after this registration, and smoke-tested on tiny settings, whose outputs are discarded.
- The scoring functions get unit tests on hand-built inputs before any C4 data are scored.

### C4-R and C4-X outcome (scored 2026-11-24 13:05 IST with the registered code, unchanged; `results/figures/c4_controls.json`)
| class | row wins (of 6 t ≤ 400) | R1 | H_indep → H_common | R2 | defined common-row LOO E (in [0.6, 1.4]) | R3 |
|---|---|---|---|---|---|---|
| automobile | 2 | failed | 16.1 → 9.3 | failed | 3 of 9 (0) | failed |
| cat | 6 | held | 4.45 → 3.98 | failed | 3 of 9 (0) | failed |
| ship | 4 | held | 3.23 → 4.42 | failed | 5 of 9 (2) | failed |
| truck | 3 | failed | 6.56 → 8.49 | failed | 5 of 9 (3) | failed |

- **R1 failed (2/4), R2 failed (0/4), R3 failed.** D_row/D_net at t ≤ 400 lies between 0.69 and 1.71: changing the row and changing the network move removed-label losses by comparable amounts. A common row does not restore homogeneity or calibration.
- **Registered interpretation applies.** The heterogeneity comes mainly from network-specific responses to an *untrained input*, not from the independent draw itself. The paper must say "untrained" rather than "independently drawn". Corollary: sharing the row's initialisation across retrain seeds would *not* repair the removed-label reference. The draft listed that as an untested alternative.
- **X1 held.** All 27 cross-class retrains score null-label E < 0.5 (range −0.41 to 0.19; max 0.10 / 0.12 / 0.19 for automobile / ship / truck). Independently trained networks that kept class k are scored as "not removed". The null-label audit tracks removal of class k, not network identity.

### C4-F outcome (scored 2026-11-25 12:48 IST with the registered code; `results/figures/c4_fd.json`)
The scoring function `score_c4f` and all FD computations are unchanged from registration. Run-time changes (memory release; Inception featurisation chunk 250 → 50, verified to give identical features) are in EXPERIMENT_LOG.md.

| class | retrain-pair FD, label k | retrain-pair FD, label 0 | ratio | F2 | original–retrain FD | F3 | re-draw–retrain FD, median [range] | fraction ≤ max pair | F1 |
|---|---|---|---|---|---|---|---|---|---|
| ship (8) | 0.50, 0.92, 1.63 | 0.13, 0.13, 0.11 | 8.3 | held | 0.96, 1.12, 1.87 | failed | 1.23 [0.33, 2.71] | 0.73 | held |
| automobile (1) | 2.28, 2.07, 1.00 | 0.11, 0.11, 0.11 | 16.3 | held | 1.31, 2.23, 1.67 | failed | 1.62 [0.51, 2.74] | 0.85 | held |
| truck (9) | 0.80, 0.63, 1.00 | 0.15, 0.17, 0.12 | 5.5 | held | 1.24, 0.91, 1.34 | failed | 1.14 [0.25, 2.18] | 0.28 | held |

- **F1 held (3/3).** Under the removed label, a single-retrain FD puts 28–85 % of zero-step re-draws within the spread of retrain pairs.
- **F2 held (3/3).** Retrains disagree 5.5–16× more under the removed label than under airplane, which every retrain trained.
- **F3 failed (0/3).** F3 was the registered sanity check. On every class, the unmodified original is closer to at least one retrain than the farthest pair of retrains is to each other.
  - The instrument is precise: under airplane, all model pairs (original included) are at 0.11–0.18.
  - So the failure means that the removed-label reference spread is as large as the effect of doing nothing. It does not mean that the FD is too noisy to measure.
  - It is a stronger negative for the removed-label FD than F1 alone. F3 was meant to show that the metric has *some* power; it does not reliably separate doing nothing from retraining when compared against one retrain.
- **Caveats (registered design choices):**
  - 768-d Inception features (Mixed_6e block, less semantic than the 2048-d pool used for FID), chosen for full-rank covariance at n = 1,000.
  - Forgotten label only. FID_O pools all classes, so this measures the forgotten-label component of FID_O, not FID_O itself.
- **Exploratory, not registered:** re-draws that passed C3-A have a slightly lower mean FD to the retrains on ship and truck (1.08 vs 1.41, Mann–Whitney p = 0.09; 1.12 vs 1.31, p = 0.13), and an equal one on automobile (1.60 vs 1.60). There is no clear relationship.

### Record corrections (2026-11-25, found by the second independent review; no data or verdict changes)
- **C3-B table.** The truck row movements 3.19e-6, 3.09e-6 and 1.82e-6 are *absolute* (`forget_row_moved` in label_diag.json), not relative. The label is corrected in place.
- **`sanity_forget_row_untrained`.** label_diag.json has `False` for classes 1 and 3. That flag came from the superseded absolute threshold (Amendment A6 replaced it with relative movement). Under the relative criterion every removed row is untrained (≈10⁻⁷ relative movement, against ≈0.06 relative for trained rows).
- **C4-R and the paper.** R1 held on cat (6/6) and ship (4/6) and failed overall (2/4 classes). In the paper, "the cause is the untrained input, not the draw" is replaced by "the draw and each network's reading of an untrained input contribute comparably". That the input being *untrained* is the cause still rests on the contrast with the trained null label, not on an intervention.

## Study C5 — real FID_O against one retrain (registered 2026-11-25 13:55 IST, before any C5 data or code)
**Why.**
- The second independent review (6/10) found the link to published practice thin. C4-F measured a removed-label FD with 768-d features, not FID_O itself.
- FID_O (Zheng, Tai & Yeh, CVPR 2026, arXiv:2603.22870, §5.2 and App. A) is "the FID between the generated images from the unlearned model and those from the oracle model for all the classes". It uses 5,000 samples per class, DDIM with 50 steps, and one retrain. Their Table 5 reports FID_O = 0.00 for the retrain, i.e. the retrain compared with itself, with no retrain-to-retrain floor. Published values are SalUn 3.59–6.93 and SPM 1.80–1.97.
- C5 computes FID_O as defined, with a seed floor.

**Design.**
- Forgotten classes k ∈ {8, 1, 9} (ship, automobile, truck), the same models as C4-F:
  - the original;
  - the 3 exact retrains per class;
  - the 20 C3-A re-draws per class (seeds 100–119, constructed exactly as in C3-A/C4-F).
- Per (model, label): 1,000 samples, C3-A sampler (DDIM 50 steps, guidance 2.0). This is a compute-limited deviation from the 5,000 per class in Zheng et al.
  - **Original:** all 10 labels, twice, with independent sampler seeds (copies A and B). Copy B gives the same-model sampling floor.
  - **Retrains:** all 10 labels.
  - **Re-draws:** only label k. For labels ≠ k a re-draw is the original network bit for bit, so its samples for those labels are the original copy A samples (same seeds, identical images).
- Sampler seeds:
  - original A: 70000 + label; original B: 71000 + label;
  - retrain (k, s): 72000 + 100·k + 10·s + label;
  - re-draw (k, r): 73000 + 100·k + (r − 100).
- Features: pytorch-fid InceptionV3, 2048-d pool3 (standard FID), on images mapped from [−1, 1] to [0, 1]. Per-(model, label) features are cached so that pooled statistics can be assembled.
- **FID_O(a, b):** the FD between Gaussian fits to the pooled 10,000 samples of model a (all 10 labels) and of model b.
- **FID_O^ret(a, b):** the same, pooled over the 9 retained labels only.
- Per class, the quantities are:
  - 3 retrain-pair FID_O values;
  - the same-model floor FID_O(original A, original B);
  - 3 original–retrain values;
  - 60 re-draw–retrain values;
  - 3 retrain-pair FID_O^ret values.

**Rules** (per class; overall verdicts as stated).
- **O1 (FID_O against one retrain cannot separate re-draws from retrains).** Of the 60 (re-draw, retrain) FID_O values, the fraction ≤ the largest retrain-pair FID_O is ≥ 0.25. **Held overall if held on ≥ 2 of 3 classes.**
- **O2 (the retrain floor is not sampling noise).** The mean retrain-pair FID_O is > 1.5 × the same-model floor. **Held overall if held on ≥ 2 of 3 classes.**
- **O3 (sanity / power).** The minimum original–retrain FID_O is > the maximum retrain-pair FID_O. **Held overall if held on every class.** We are genuinely unsure: the 768-d single-label analogue (C4-F F3) failed. Pooling dilutes the forgotten label to 10 %, but the original's forgotten-class mode is absent from every retrain.
- **O4 (the forgotten label drives the floor).** The mean retrain-pair FID_O over all 10 labels is > 1.25 × the mean retrain-pair FID_O^ret over the 9 retained labels. **Held overall if held on ≥ 2 of 3 classes.**

**Registered interpretation.**
- If O1 holds, FID_O against a single retrain scores a zero-step edit as retrain-like at a rate comparable to genuine retrains.
- If O3 fails, FID_O against one retrain cannot reliably separate doing nothing from retraining.
- If O2 fails, the retrain floor is within sampling noise at n = 1,000 per class, and O1/O3 say nothing about retrain variability.
- We will not compare our values with Zheng et al.'s numerically as evidence: the model, sample size and guidance differ. We report them only as context for scale.

**Code.**
- `src/analysis/c5_fido.py`, written after this registration.
- A smoke test on tiny settings; its outputs are discarded unread.
- Unit tests for the pooled-statistics assembly and the O1–O4 scoring, on hand-built inputs, before any data.

**Compute.** ≈ 170 sampling runs of 1,000 images (≈ 4.3 min each on the laptop under the C4-F memory settings), ≈ 12 h, local MPS only.

### C5 outcome (scored 2026-11-26 03:20 IST with the registered code, `src/analysis/c5_fido.py`; `results/figures/c5_fido.json`)
Same-model sampling floor FID_O(original A, original B) = **4.48** (n = 1,000 per label, 10,000 pooled, 2048-d).

| | ship (8) | automobile (1) | truck (9) |
|---|---|---|---|
| retrain-pair FID_O | 7.23, 11.82, 12.18 | 24.13, 17.18, 16.40 | 9.96, 9.93, 8.64 |
| mean / floor | 2.33× | 4.30× | 2.12× |
| retrain-pair FID_O^ret (9 retained labels) | 6.67, 7.62, 6.69 | 6.32, 6.94, 5.84 | 7.33, 8.83, 6.09 |
| mean all-label / mean retained | 1.49 | 3.02 | 1.28 |
| original–retrain FID_O | 8.60, 9.81, 13.78 | 11.61, 23.40, 16.64 | 12.71, 11.64, 12.68 |
| re-draw–retrain FID_O, median [range] (60) | 10.83 [6.28, 18.54] | 14.87 [8.55, 29.47] | 11.63 [8.39, 17.91] |
| O1 fraction ≤ max retrain pair | 0.57 held | 0.93 held | 0.12 **failed** |
| O2 | held | held | held |
| O3 (min orig–retrain > max pair) | 8.60 < 12.18 **failed** | 11.61 < 24.13 **failed** | 11.64 > 9.96 held |
| O4 (> 1.25×) | held | held | held (thin: 1.28) |

**Overall verdicts (registered aggregation): O1 held (2/3), O2 held (3/3), O3 failed (needs 3/3; held on truck only), O4 held (3/3).**

**Registered interpretation applied.**
- O2 held: retrain-to-retrain FID_O is 2.1–4.3× the sampling floor at this n, so the retrain reference is not a fixed point.
- O1 held: on ship and automobile, FID_O against one retrain scores most zero-step re-draws inside the retrain-pair range. On truck it does not (12 %): the truck retrains agree more closely with each other (max pair 9.96) and most re-draws fall outside.
- O3 failed: on ship and automobile, the unmodified original is no farther from at least one retrain than the farthest retrain pair. On truck it is separated. FID_O against one retrain therefore cannot *reliably* separate doing nothing from retraining. This repeats the C4-F F3 failure on two of three classes; truck is the exception here, where it was not in C4-F.
- O4 held: the forgotten label inflates the retrain floor on every class (1.28–3.02×). Retained-label-only pairs (5.8–8.8) still exceed the sampling floor, so some retrain variability is not forgotten-label driven, which is consistent with general training-seed FID variance (Dufour, Efros & Pérez 2026, arXiv:2606.20536).
- Comparison with Zheng et al.'s published values (retrain 0.00, SalUn 3.59–6.93, SPM 1.80–1.97) is context for scale only, as registered: model, sample size (1,000 vs 5,000 per class) and guidance differ, and FID is biased upward at smaller n.

**Exploratory (computed after scoring; not registered).** Against a fixed retrain, the fraction of re-draws with *lower* FID_O than the unmodified original: ship 0.45/0.60/0.25, automobile 0.65/0.75/0.75, truck 0.50/0.65/0.65 (per retrain seed). A re-draw does stop generating class k (C3-A), so lower output-level FID_O is expected. What FID_O cannot see is that a one-row re-fit restores the class (C3-K).
