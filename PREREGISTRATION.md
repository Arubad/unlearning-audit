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

Format: `YYYY-MM-DD — what changed — why — git commit`. All amendments below were made **before any experiment was run** (Week 0), following an independent review recorded in [`REVIEW.md`](REVIEW.md).

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
