# Suppressed or erased? Auditing diffusion and LLM unlearning with influence estimation and membership inference

**Status:** Pre-registration v0.2 · Week 0 of 10 · Pre-registered 2026-10-13, amended 2026-10-18 before any run · Models: small class-conditional DDPM on CIFAR-10 (32×32, primary track), Qwen2.5-0.5B on a TOFU-style set via OpenUnlearning (corroborating track) · Seeds: 0, 1, 2

| Document | What it is |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Frozen hypotheses, numeric pass/fail tests, models, seeds. Changes only via the dated Amendments appendix. |
| [`REVIEW.md`](REVIEW.md) | Independent review of 2026-10-18: citation corrections, novelty against the literature, design flaws, what changed and why. Read this before the pre-registration. |
| [`ROADMAP.md`](ROADMAP.md) | Milestones as GitHub task lists. Tick them off; this is the progress tracker. |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Append-only log, one entry per run, expectation written before launch. |
| [`READING.md`](READING.md) | Ordered reading list with the reason for each item. |

**Problem in one sentence.** We determine whether published approximate-unlearning methods erase the influence of forgotten data or merely suppress it, measured by whether the *influence* of forget examples on the model's other outputs survives, and we test whether a property of the parameter update (its effective rank, at matched norm) explains how fast the forgotten data come back.

**Why this matters.** Unlearning papers typically certify success with a single number: membership-inference AUC on the forget set drops to chance, or the forgotten class is no longer generated. The LLM literature has already shown that this certificate is weak (relearning, probing and tampering attacks all recover "forgotten" knowledge; see [`REVIEW.md`](REVIEW.md) §3), and the classifier literature has shown the same with linear probes. What nobody has measured is whether the *influence* of the forgotten examples on retained outputs, in the Koh–Liang / TRAK sense, actually vanishes, nor whether any mechanical property of the update predicts relearning speed, nor whether any of this holds for class-unlearned diffusion models rather than text-to-image concept erasure. An audit that measures the gap to the retrain-from-scratch oracle along these axes, and explains the gap mechanically, is a prerequisite for trusting any of these methods in deployment.

---

## 1. Research proposition

### The gap

IISc has dense coverage of both sides of this problem, methods and measurement tools, without an audit that joins them.

On the **methods side**, Prathosh A.P.'s group has published *"Partially Blinded Unlearning: Class Unlearning for Deep Networks from Bayesian Perspective"* ([AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32682)), *"Concept Forgetting via Label Annealing"* ([UAI 2025](https://proceedings.mlr.press/v286/panda25a.html)) and *"Unlearning in Diffusion Models under Data Constraints: A Variational Inference Approach"* ([TMLR 2026](https://mlanthology.org/tmlr/2026/panda2026tmlr-unlearning/), [arXiv 2510.04058](https://arxiv.org/abs/2510.04058)). The last of these is the closest paper to our diffusion track and its "data constraints" setting (no retain data) is one of our ablations. Chiranjib Bhattacharyya's DisCEdit ([NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93306)) and ModHiFi ([arXiv 2511.19566](https://arxiv.org/pdf/2511.19566), NeurIPS 2025) combine pruning with class forgetting on CIFAR and ask which components carry predictive fidelity, which is the component-level view our per-layer analysis mirrors. VAL's Concept Siever ([TMLR 2025](https://mlanthology.org/tmlr/2025/singh2025tmlr-concept)) and Soma Biswas's GenErase ([CVPR 2026](https://cvpr.thecvf.com/virtual/2026/poster/38944), training-free, text-to-image only) are the vision-side concept-erasure methods; they become relevant only if the stretch Stable Diffusion arm is run.

On the **measurement side**, Prathosh's *"f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness"* ([ICLR 2026](https://www.iclr.cc/virtual/2026/poster/10009271)) is exactly the attribution tool needed to check whether influence has vanished, and its premise (two seeds differ by about as much as removing one example) is the null we would use for the stretch hypothesis test.

What all of these leave open is the audit itself: when a published method reports that the forget set is "gone", does the influence of that data on the model's outputs actually vanish, or has it merely been suppressed in a way that an influence estimator or a relearning attack still detects? And if it is suppression, what about the parameter update makes it so?

### The research question

For class unlearning in a diffusion model (primary) and example-level unlearning in a small LLM (corroborating), how far does each method move the model toward the retrain-from-scratch oracle, measured jointly by membership inference, influence/attribution estimates and relearning speed; does the surviving influence of forget examples on retain outputs predict relearning speed; and does the effective rank of the parameter update, at matched update norm, explain the gap?

### Why this is an audit, not a new method

The contribution is the protocol, the influence axis and the mechanism, not a competitor to any lab's method. Every method is evaluated by the same matched-compute rules against the same oracle, and the explanation (H3) is a property any method can be checked against. The LLM track is built on [OpenUnlearning](https://arxiv.org/pdf/2506.12618) (NeurIPS 2025 D&B), which already packages TOFU, GA, GradDiff, NPO and MIA metrics; what this repository adds is the diffusion arm, the influence axis and a checkpoint-pair interface.

---

## 2. Pre-registered hypotheses (v0.2)

Full text with dates and seeds is in [`PREREGISTRATION.md`](PREREGISTRATION.md); the 2026-10-18 amendments are in its appendix. The numerical tests are copied verbatim here. The order below is the order of the paper: H2 and H3 are the contribution, H1 is the replication that anchors them.

**H2 — influence survives (headline).** Gradient-based influence of forget examples on retain-set generations drops far less under approximate unlearning than under retraining.
*Confirmed* if the mean attribution score of forget examples on retain generations drops by less than 50% under approximate unlearning versus more than 90% under retraining, with non-overlapping 95% CIs. *Refuted* if the approximate drop is within 10 percentage points of the retrain drop.
*Pre-specified secondary analysis (added 2026-10-18):* Spearman correlation between surviving retain-set influence of forget examples and relearning steps-to-90% across method × seed cells. This ties H2 to H3.

**H3 — low-rank mechanism, at matched norm (amended 2026-10-18).** The parameter delta from approximate unlearning is near-low-rank, dominated by a few directions that cancel forget-class scores, whereas retrain-minus-original is full-rank; low-rankness predicts relearning speed *after controlling for update norm*. Tested only on full-parameter tracks (the DDPM, and a full-fine-tune LLM arm), because under LoRA the rank is bounded by construction. The variational method is predicted to sit between the extremes.
*Confirmed* if the top-5 singular values capture more than 70% of $\|\Delta\theta\|_F$ for approximate methods versus less than 30% for the oracle delta, and the partial Spearman correlation of effective rank with relearning steps, controlling for $\|\Delta\theta\|_F$, is $\ge 0.7$ across method × seed cells. *Refuted* if $|\rho_{\text{partial}}| < 0.3$ or the spectra are indistinguishable.

**H1 — suppression, not erasure (cross-modality replication).** Gradient-ascent-style and label-annealing unlearning push forget-set MIA AUC to about 0.5 but leave intermediate representations of the forgotten class largely intact, so a linear probe on frozen features still recovers it and few-shot relearning is an order of magnitude faster than for the oracle. This is established for LLMs ([Hu et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/18fd48d9cbbf9a20e434c9d3db6973c5-Abstract-Conference.html); [Łucki et al., TMLR 2025](https://mlanthology.org/tmlr/2025/ucki2025tmlr-adversarial); [Unlearning Isn't Deletion](https://arxiv.org/html/2505.16831v1)) and for classifiers ([An Illusion of Unlearning?, AISTATS 2026](https://proceedings.mlr.press/v300/gao26e.html); [From Dormant to Deleted](https://arxiv.org/html/2505.22310)); we test it on a class-unlearned DDPM, with stronger MIA baselines (loss-based, LiRA-style and class-geometry MIA) so that "AUC about 0.5" is not a strawman.
*Confirmed* if, for gradient-ascent and label-annealing methods, MIA AUC on the forget set is ≤ 0.55 while linear-probe accuracy on the forgotten class remains ≥ 80% of its pre-unlearning value and relearning steps-to-90% are ≤ 1/5 of the oracle's, in at least 2 of 3 seeds. *Refuted* if probe accuracy falls below 50% of original or relearning speed is within 2× of the oracle.

**Most interesting possible outcome.** If the approximate methods pass the influence test as well as MIA, "cheap unlearning is enough at this scale" is a positive and surprising result; the forget-set-size ablation would then say whether it is a capacity effect. If H3's correlation is absent at matched norm, the spectra still say whether low-rankness is the discriminating property or whether it is *which* layers carry the delta.

---

## 3. Experimental design

| Role | Variable | Levels |
|---|---|---|
| Independent | Unlearning method | gradient ascent; gradient difference; label annealing (UAI 2025 style); variational/Bayesian (TMLR 2026 style); NPO (LLM track); retrain-from-scratch oracle |
| Independent | Forget-set size | 1%, 5%, 10% of training data |
| Independent | Retain-data availability | full, none |
| Independent | Parameterisation (LLM track, for H3) | LoRA r16; full fine-tune with 8-bit Adam |
| Dependent | Membership-inference AUC | loss-based, LiRA-style, and class-geometry MIA (CMIA); multi-timestep for diffusion; forget vs held-out |
| Dependent | Influence/attribution | mean score of forget examples on retain-set outputs |
| Dependent | Forgetting | forget-class generation rate (diffusion); forget-set ROUGE/accuracy (LLM) |
| Dependent | Retain utility | FID on retain classes; benchmark accuracy (LLM) |
| Dependent | Relearning | steps to 90% recovery when fine-tuned on 1–5% of the forget set |
| Dependent | Linear-probe accuracy | frozen intermediate features |
| Dependent | Update norm $\|\Delta\theta\|_F$ | per layer and total; the confound H3 controls for |
| Dependent | Effective rank of per-layer $\Delta\theta$ | singular spectrum |
| Controlled | Base checkpoints | one class-conditional DDPM on CIFAR-10 at 32×32, 5–10M parameters; Qwen2.5-0.5B on a TOFU-style set |
| Controlled | Seeds | same 3 across methods |
| Controlled | Compute | matched gradient steps per method |
| Controlled | Evaluation | identical prompts / noise seeds; Hessian damping $\lambda$ fixed and reported |
| Control (expected to fail) | "Do nothing" baseline | original checkpoint run through the full audit; must show no forgetting on any axis |

**Tracks.** *Diffusion (primary):* a small class-conditional UNet DDPM on CIFAR-10, one class unlearned with four methods plus the oracle. The original DDPM ([Ho et al.](https://arxiv.org/pdf/2006.11239), 35.7M parameters, 800k steps) took about 10.6 h on a TPU v3-8, far more than a free-tier budget; in 3–6 T4-hours we expect a 5–10M-parameter model trained for tens of thousands of steps with FID in the tens. That is acceptable for a *relative* audit against an oracle trained the same way, and every figure says so. *LLM (corroborating):* Qwen2.5-0.5B on a TOFU-style fictitious-author benchmark through OpenUnlearning, with gradient ascent, gradient difference, NPO and the oracle; LoRA r16 for the main cells, full fine-tune with 8-bit Adam for the H3 cells.

**Baselines.** The retrain-from-scratch oracle is the reference point on every axis; the "do nothing" checkpoint is the floor.

**Ablations.** Forget-set size, number of unlearning steps, and retain-data availability (the "data constraints" setting of the TMLR 2026 paper). Layer-wise: which layers' deltas carry the unlearning (mirroring DisCEdit/ModHiFi).

**Moved to stretch (2026-10-18).** The f-INE-style hypothesis test (needs 3–5 oracle retrains for the seed-to-seed null; the most expensive arm); the Stable Diffusion 1.5 / SD-Turbo concept-erasure arm.

**The expected-to-fail control.** The unmodified checkpoint goes through the entire pipeline. It must show MIA AUC well above chance on the forget set, full probe accuracy, zero relearning steps, zero attribution drop and a zero delta spectrum. If the harness reports "forgetting" for it, the harness is wrong.

---

## 4. The reasoning component

The audit rests on three approximations we make explicit and test.

**Influence and one-step unlearning.** The influence of training point $z$ on test point $z_{\mathrm{test}}$ is

$$I(z, z_{\mathrm{test}}) = -\nabla \ell(z_{\mathrm{test}}, \hat\theta)^{\top} H_{\hat\theta}^{-1} \nabla \ell(z, \hat\theta),$$

and the one-step unlearning estimate is $\theta_{-S} \approx \hat\theta + \tfrac{1}{n} H^{-1} \sum_{z \in S} \nabla \ell(z, \hat\theta)$. Both assume a convex, well-conditioned loss. For deep networks $H$ is indefinite and must be damped, $(H + \lambda I)$, and influence estimates are known to be fragile ([Basu et al.](https://arxiv.org/abs/2006.14651)); we report how the attribution ranking changes with $\lambda$ and with a LoRA-restricted parameter subspace, and we use TRAK-style random projection to 4,096 dimensions so that per-example gradients (about 143 MB each in fp32 for a 35M-parameter UNet) are never stored whole.

**Training randomness.** Two training runs from different seeds differ by about as much as removing a single example; that is the premise of f-INE's hypothesis-testing framing. The full f-INE-style test needs 3–5 oracle retrains and is a stretch arm; at minimum scale we report attribution drops with bootstrap CIs over retain queries and say explicitly that the seed null is not estimated.

**The low-rank mechanism (H3).** Write the unlearned model as $\hat\theta + \Delta$ and ask what $\Delta$ does. If $\Delta$ is close to rank-1 per layer, it acts as a steering vector that subtracts the forget-class direction from the output while leaving feature geometry intact. That single property explains all three symptoms at once: near-chance MIA (forget-example loss is pushed up to the held-out level), high probe accuracy (features unchanged) and fast relearning (a rank-1 update is undone by a rank-1 update). Two cautions shape the test. First, under LoRA the rank of $\Delta$ is at most $r$ for every method including the oracle, so the comparison is meaningless there; H3 is run only on full-parameter updates. Second, a method that moves the weights less will relearn faster regardless of rank ([From Dormant to Deleted](https://arxiv.org/html/2505.22310) predicts relearning from L2 distance; [FRAG](https://arxiv.org/abs/2608.25429) argues selectivity matters more than distance), so the test is a partial correlation controlling for $\|\Delta\|_F$. For the diffusion track the same argument applies to the score network: class unlearning that only shifts the conditional score near the forget class's modes leaves the unconditional density unchanged, which we check by measuring how far forget-class samples drift under unconditional generation.

---

## 5. Compute budget (hard cap)

Diffusion track: small class-conditional UNet DDPM on CIFAR-10 (5–10M parameters; 3–6 T4-hours; FID in the tens, reported as such). LLM track: Qwen2.5-0.5B via OpenUnlearning, LoRA r16 (fp32 on T4, which has no bf16) plus full-FT cells with 8-bit Adam. **The 2026-10-18 review found the original DDPM line (a 35M-parameter model in 6 h) unrealistic**; the table below is revised. Kaggle's free quota is commonly reported as ~30 GPU-hours per week (not an official figure).

| Experiment | Model | Runs × seeds | GPU-h (revised) | Kaggle weeks |
|---|---|---|---|---|
| D1 Base DDPM training (5–10M params) | CIFAR-10 DDPM | 1 × 5 h | 5 | — |
| D2 Oracle retrains on retain set | DDPM | 3 × 5 h | 15 | 0.5 |
| D3 Unlearning: 4 methods × 3 seeds × 3 forget sizes (matched steps) | DDPM | 36 × 0.25 h | 9 | 0.3 |
| D4 MIA (loss, LiRA-style, CMIA; multi-timestep) + projected-gradient attribution | DDPM | all | 5 | 0.2 |
| D5 Relearning + linear probes + $\Delta\theta$ spectra and norms | DDPM | 36 × 0.1 h | 3.5 | 0.1 |
| L1 Fine-tune on TOFU-style set (LoRA + one full-FT) | 0.5B | 2 × 1.5 h | 3 | 0.1 |
| L2 Oracle retrains (retain only; LoRA + one full-FT) | 0.5B | 4 × 1.5 h | 6 | 0.2 |
| L3 Unlearning: 4 methods × 3 seeds (LoRA) + 3 methods × 2 seeds (full-FT, for H3) | 0.5B | 18 × 0.4 h | 7 | 0.25 |
| L4 Evaluation, MIA, attribution, relearning, probes | 0.5B | all | 6 | 0.2 |
| **Total full version** | | | **~60** | **~2** |
| S1 (stretch) f-INE seed null: 2 more oracle retrains | DDPM | 2 × 5 h | 10 | 0.3 |
| S2 (stretch) SD-1.5 / SD-Turbo concept erasure, 2 methods × 2 concepts | SD 1.5 | 4 × 2 h | 8 | 0.3 |

**Minimum viable version (~17 GPU-hours, one Kaggle week).** DDPM on a 3-hour schedule (D1 = 3 h); a single oracle retrain (D2 = 3 h); 3 methods (gradient ascent, label annealing, oracle) × 2 seeds × 1 forget size (D3 = 1.5 h); D4 and D5 on 2k-image subsets (3 h). LLM side: L1 LoRA only, one oracle (L2 = 1.5 h), 3 methods × 2 seeds (L3 = 2.4 h), L4 on a 200-example evaluation set (1.5 h). If the DDPM is still too slow, CelebA at 32×32 with a class-conditional attribute as the forget target, or a SmolLM-135M LLM track, are acceptable substitutes.

*What survives at minimum scale:* H2 as a two-method comparison with bootstrap CIs over retain queries (no seed null); H3 for the method-level spectrum contrast on the DDPM (the partial-correlation test is under-powered with 6 cells and reported as a trend); H1 on all three axes with loss-based MIA (LiRA/CMIA need the full version). The full version is required for the H3 correlation across enough cells, the full-FT LLM arm, the forget-set-size and retain-availability ablations, and the stronger MIA baselines.

**Caching rules.** Save the base checkpoint and every unlearned checkpoint (full UNet weights; LoRA adapters or full weights for the LLM). Precompute per-example losses at a fixed set of 5 diffusion timesteps once per checkpoint and reuse for MIA. Compute per-example gradients once per checkpoint on a training-set subset and a few timestep draws, store them random-projected to 4k dimensions, so that attribution across all checkpoint pairs is a CPU matrix product. Reuse the same evaluation noise seeds and prompts for every checkpoint so FID and attribution are paired.

---

## 6. How to run (Kaggle)

Each experiment is one notebook in `notebooks/` (from `tools/make_notebooks.py`) that clones the repo and calls `python -m src.run`. Kaggle: Accelerator **GPU T4 x2**, Internet **On**. Results land in `/kaggle/working/results`; save as a Kaggle dataset and attach to the next notebook.

| Order | Notebook | Runs | GPU-h (minimum) |
|---|---|---|---|
| 0 | `00_smoke_test` | tiny UNet + 2 methods + audit | 0.1 |
| 1 | `01_D1_train_base_and_classifier` | `train_base`, `train_classifier` | 3–5 |
| 2 | `02_D2_oracle_retrain` | `train_oracle` | 3–5 |
| 3 | `03_D3_unlearn_sweep` | `sweep --minimum` (unlearn + audit per cell) | ~6 |
| 4 | `04_D4_D5_audit` | `analyze` | 0 |
| 5 | `05_L1_L2_llm_finetune_oracle` | `llm_finetune`, `llm_oracle` | 3 |
| 6 | `06_L3_L4_llm_unlearn_audit` | `llm_sweep --minimum` | ~4 |
| 7 | `07_analysis_and_figures` | `analyze` | 0 |

Locally: `pip install -r requirements.txt && python tests/test_audit_cores.py` runs the numpy/sklearn-only tests (schedule, AUC, spectrum, TRAK, probe, H3 partial correlation) without a GPU.

**Status of the code (2026-10-20).** All modules are written and compile; the numpy/sklearn cores are unit-tested; the torch parts (DDPM training, unlearning, the audit axes, the LLM track) have not been executed on a GPU. Every audit axis is implemented, including membership inference (`src/audit/mia.py`: loss-based, SecMI t-error, offline LiRA with the oracle retrain as the OUT distribution, and class-geometry MIA) and the full LLM audit runner (`src/llm/audit.py`: eval, MIA, TRAK-style attribution, probe, relearning, spectrum). `mia_lira_auc` is NaN when no oracle is available. Run the smoke test first, then the do-nothing control.

## 6. Plan

The week-by-week plan with task lists lives in [`ROADMAP.md`](ROADMAP.md). Summary: W1–2 reading and rewrite of the pre-registration in your own words; W3–4 base models, oracle and the "do nothing" control; W5–7 minimum-viable sweeps, DDPM first; W8–10 analysis, 4-page write-up, one-page summary for Prathosh A.P.

---

## 7. Deliverables

1. **Preprint** (4-page workshop version first, then 6–8 pages): *"Suppressed or erased? An influence-and-membership audit of unlearning in a diffusion model and a small LLM."*
2. **This repository as a library:** MIA, attribution, relearning, probe and spectrum modules that accept any pair of original/unlearned checkpoints, with the diffusion arm and the influence axis as the additions over OpenUnlearning.
3. **One-page summaries** for two readers only. Prathosh A.P.: the variational method's position, the influence result, and where the f-INE-style test would go. Bhattacharyya: the component-level (which layers' deltas carry the unlearning) analysis mirroring DisCEdit/ModHiFi.

---

## 8. Research protocol

- **Pre-register before running.** Hypotheses, tests, models, seeds and date live in `PREREGISTRATION.md`, frozen before the first unlearning run; changes go in its Amendments appendix with a date and a reason.
- **Experiment log.** Every run in `EXPERIMENT_LOG.md` records config hash, git commit, expectation, observation and next change; it feeds the negative-results section.
- **Seeds.** Fixed for training, forget-set sampling, noise and prompts; three seeds (0, 1, 2) behind any plotted number.
- **Uncertainty.** Mean plus 95% bootstrap CI; paired across checkpoints on identical prompts/noise seeds.
- **Matched compute.** Every approximate method gets the same number of gradient steps; $\lambda$ is fixed and reported.
- **Negative results.** The preprint states which hypotheses were refuted and what we now believe instead.
- **Expected-to-fail control.** The "do nothing" baseline must show no forgetting on any axis before any figure is frozen.
- **Figures** come only from single-command-reproducible runs. A CIFAR-10 result from a 5–10M-parameter DDPM is reported as exactly that.
- **"First" claims** are made only after a Semantic Scholar sweep recorded in `REVIEW.md`.

---

## 9. Repository layout

```
unlearning-audit/
├── README.md
├── PREREGISTRATION.md            # frozen 2026-10-13; amendments appendix
├── REVIEW.md                     # independent review, 2026-10-18
├── ROADMAP.md                    # milestones as task lists (progress tracker)
├── EXPERIMENT_LOG.md             # append-only run log
├── READING.md                    # ordered reading list
├── requirements.txt
├── configs/                      # _base_*.yaml + diffusion/llm base and unlearn sweeps
├── src/                          # common/, diffusion/, llm/, audit/, analysis/, run.py, smoke.py (see src/README.md)
├── tests/                        # numpy/sklearn-only tests for the audit cores
├── tools/make_notebooks.py       # regenerates notebooks/
├── notebooks/                    # one Kaggle notebook per experiment, 00_smoke_test first
├── results/                      # checkpoints, projected gradients, per-timestep losses (git-ignored)
└── paper/                        # preprint source, one-pagers/, notes/
```

---

## 10. Related work to read first

See [`READING.md`](READING.md) for the ordered list with reasons. The must-reads: Ho et al. (DDPM), Carlini et al. (LiRA), Duan et al. (SecMI), Koh and Liang plus Basu et al. (influence functions and their fragility), Park et al. (TRAK), Golatkar et al. and Kurmanji et al. (relearn time, SCRUB), Maini et al. (TOFU), Zhang et al. (NPO), Lynch et al. and Łucki et al. (robust unlearning evaluation), Gao et al. (An Illusion of Unlearning?), Panda et al. (TMLR 2026) and f-INE (ICLR 2026).

---

## 11. Faculty this work is aimed at

| Faculty | Dept | Paper this extends | Role |
|---|---|---|---|
| Prathosh A.P. | ECE | *Unlearning in Diffusion Models under Data Constraints* (TMLR 2026); *f-INE* (ICLR 2026); label annealing (UAI 2025) | Primary reader |
| Chiranjib Bhattacharyya | CSA | DisCEdit (NeurIPS 2024); ModHiFi (NeurIPS 2025) | Second reader, via the per-layer component analysis |

---

## 12. License and citation

Code: MIT. Data and figures: CC BY 4.0.

```bibtex
@misc{unlearning-audit-2026,
  title  = {Suppressed or erased? An influence-and-membership audit of unlearning in a diffusion model and a small LLM},
  author = {Arush Badhe},
  year   = {2026},
  note   = {Pre-registered 2026-10-13, amended 2026-10-18. https://github.com/Arubad/unlearning-audit}
}
```
