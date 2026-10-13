# Suppressed or erased? Auditing diffusion and LLM unlearning with influence estimation and membership inference

**Status:** Pre-registration · Week 0 of 5 · Pre-registered 2026-10-13 · Models: DDPM on CIFAR-10 (32×32), Qwen2.5-0.5B + LoRA rank 16 · Seeds: 0, 1, 2

**Problem in one sentence.** We determine whether published approximate-unlearning methods erase the influence of forgotten data or merely suppress it, and identify the property of the parameter update that explains the difference.

**Why this matters.** Unlearning papers typically certify success with a single number: membership-inference AUC on the forget set drops to chance, or the forgotten class is no longer generated. Neither says whether the model's internal representation of that data is gone. If an attacker can relearn the forgotten class in a handful of steps, or a linear probe still reads it off frozen features, then "unlearning" was a surface edit, and privacy or copyright claims built on it do not hold. An audit that measures the gap to the retrain-from-scratch oracle along several independent axes, and explains the gap mechanically, is a prerequisite for trusting any of these methods in deployment.

---

## 1. Research proposition

### The gap

IISc has unusually dense coverage of both sides of this problem, methods and measurement tools, without an audit that joins them.

On the **methods side**, Prathosh A.P.'s group has published a sequence of unlearning papers across model families: *"Adapt then Unlearn"* for GANs (TMLR 2025), *"Partially Blinded Unlearning: Class Unlearning from Bayesian Perspective"* (AAAI 2025), *"Concept Forgetting via Label Annealing"* (UAI 2025, with Ananda Theertha Suresh) and *"Unlearning in Diffusion Models under Data Constraints: A Variational Inference Approach"* (TMLR 2026) ([prathosh.in/publications](https://prathosh.in/publications.html)). VAL's *"Concept Siever: Controllable Erasure of Concepts from Diffusion Models"* (TMLR 2025) and Soma Biswas's *"GenErase: Generalizable and Semantically-Aware Concept Erasure in Diffusion Models"* (CVPR 2026) are the vision-side erasure methods; both labs also debias text-to-image models (*Balancing Act*, CVPR 2024; unsupervised self-debiasing, WACV 2026) ([VAL publications](https://val.cds.iisc.ac.in/publications.html); [Biswas publications](https://sites.google.com/iisc.ac.in/somabiswas/publications)).

On the **measurement side**, Prathosh's *"f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness"* (ICLR 2026) and *"Efficient Subset Selection Using Text-Guided Data Attribution"* (TMLR 2026) are exactly the attribution tools needed to check whether influence has vanished. Danish Pruthi's STAMP (ICML 2025, with Pratyush Maini) proves dataset membership via watermarked rephrasings, and his EMNLP 2026 Findings paper quantifies a privacy–hallucination trade-off in differentially private LMs, making membership inference native in his group ([Pruthi publications](https://danishpruthi.com/publications/)). Chiranjib Bhattacharyya's DisCEdit (NeurIPS 2024) and ModHiFi (NeurIPS 2025 Spotlight) identify which components carry predictive fidelity for model modification ([ML Lab publications](http://mllab.csa.iisc.ac.in/publications)), and Anirban Chakraborty's DynEval (ECCV 2026) evaluates text-to-image models holistically in the wild ([VCL publications](http://visual-computing.in/publications/)).

What all of these leave open is the audit itself: when a published method reports that the forget set is "gone", does the influence of that data on the model's outputs actually vanish, or has it merely been suppressed in a way that a relearning attack or an influence estimator still detects? And if it is suppression, what about the parameter update makes it so?

### The research question

For class/concept unlearning in diffusion models and example-level unlearning in small LLMs, how far does each method move the model toward the retrain-from-scratch oracle, measured jointly by membership inference, influence/attribution estimates and relearning speed, and which mechanistic property of the unlearning update explains the gap?

### Why this is an audit, not a new method

The contribution is the protocol and the mechanism, not a competitor to any lab's method. That is what makes it credible to six labs at once: every method is evaluated by the same matched-compute rules against the same oracle, and the explanation (H3) is a property any method can be checked against.

---

## 2. Pre-registered hypotheses

Full text in [`HYPOTHESES.md`](HYPOTHESES.md); the tests are reproduced verbatim.

**H1 — suppression, not erasure.** Gradient-ascent-style and label-annealing unlearning push forget-set MIA AUC to about 0.5 but leave intermediate representations of the forgotten class largely intact, so a linear probe on frozen features still recovers it and few-shot relearning is an order of magnitude faster than for the oracle.
*Confirmed* if, for gradient-ascent and label-annealing methods, MIA AUC on the forget set is ≤ 0.55 while linear-probe accuracy on the forgotten class remains ≥ 80% of its pre-unlearning value and relearning steps-to-90% are ≤ 1/5 of the oracle's, in at least 2 of 3 seeds. *Refuted* if probe accuracy falls below 50% of original or relearning speed is within 2× of the oracle.

**H2 — influence survives.** Gradient-based influence of forget examples on retain-set generations drops far less under approximate unlearning than under retraining, and f-INE-style tests under training randomness still flag many forget examples as influential.
*Confirmed* if the mean attribution score of forget examples on retain generations drops by less than 50% under approximate unlearning versus more than 90% under retraining, with non-overlapping 95% CIs. *Refuted* if the approximate drop is within 10 percentage points of the retrain drop.

**H3 — low-rank mechanism.** The parameter delta from approximate unlearning is near-low-rank, dominated by a few directions that cancel forget-class logits or scores, whereas retrain-minus-original is full-rank; low-rankness predicts relearning speed. The variational method is predicted to sit between the extremes.
*Confirmed* if the top-5 singular values capture more than 70% of $\|\Delta\theta\|_F$ for approximate methods versus less than 30% for the oracle delta, and Spearman(effective rank, relearning steps) ≥ 0.7 across method × seed cells. *Refuted* if $|\rho| < 0.3$ or the spectra are indistinguishable.

---

## 3. Experimental design

| Role | Variable | Levels |
|---|---|---|
| Independent | Unlearning method | gradient ascent; gradient difference; label annealing (UAI 2025 style); variational/Bayesian (TMLR 2026 style); NPO-style preference unlearning (LLM track); retrain-from-scratch oracle |
| Independent | Forget-set size | 1%, 5%, 10% of training data |
| Independent | Retain-data availability | full, none |
| Dependent | Membership-inference AUC | loss-based and LiRA-style; multi-timestep for diffusion; forget vs held-out |
| Dependent | Influence/attribution | mean score of forget examples on retain-set outputs |
| Dependent | Forgetting | forget-class generation rate (diffusion); forget-set ROUGE/accuracy (LLM) |
| Dependent | Retain utility | FID on retain classes; benchmark accuracy (LLM) |
| Dependent | Relearning | steps to 90% recovery when fine-tuned on 1–5% of the forget set |
| Dependent | Linear-probe accuracy | frozen intermediate features |
| Dependent | Effective rank of per-layer $\Delta\theta$ | singular spectrum |
| Controlled | Base checkpoints | one DDPM on CIFAR-10 at 32×32; Qwen2.5-0.5B + LoRA rank 16 on a TOFU-style fictitious-author set |
| Controlled | Seeds | same 3 across methods |
| Controlled | Compute | matched gradient steps per method |
| Controlled | Evaluation | identical prompts / noise seeds; Hessian damping $\lambda$ fixed and reported |
| Control (expected to fail) | "Do nothing" baseline | original checkpoint run through the full audit; must show no forgetting on any axis |

**Tracks.** *Diffusion:* a small UNet DDPM on CIFAR-10 (a few T4-hours), one class unlearned with four methods, plus an optional Stable Diffusion 1.5 / SD-Turbo concept erasure on a benign object concept (16 GB with 8-bit optimisers). *LLM:* Qwen2.5-0.5B with LoRA on a TOFU-type fictitious-author benchmark, with gradient ascent, gradient difference, NPO-style unlearning and the oracle.

**Baselines.** The retrain-from-scratch oracle is the reference point on every axis; the "do nothing" checkpoint is the floor.

**Ablations.** Forget-set size, number of unlearning steps, and retain-data availability (the "data constraints" setting of the TMLR 2026 paper). Layer-wise: which layers' deltas carry the unlearning (mirroring DisCEdit/ModHiFi).

**The expected-to-fail control.** The unmodified checkpoint goes through the entire pipeline. It must show MIA AUC well above chance on the forget set, full probe accuracy, zero relearning steps and a zero delta spectrum. If the harness reports "forgetting" for it, the harness is wrong.

---

## 4. The reasoning component

The audit rests on three approximations we make explicit and test.

**Influence and one-step unlearning.** The influence of training point $z$ on test point $z_{\mathrm{test}}$ is

$$I(z, z_{\mathrm{test}}) = -\nabla \ell(z_{\mathrm{test}}, \hat\theta)^{\top} H_{\hat\theta}^{-1} \nabla \ell(z, \hat\theta),$$

and the one-step unlearning estimate is $\theta_{-S} \approx \hat\theta + \tfrac{1}{n} H^{-1} \sum_{z \in S} \nabla \ell(z, \hat\theta)$. Both assume a convex, well-conditioned loss. For deep networks $H$ is indefinite and must be damped, $(H + \lambda I)$; we report how the attribution ranking changes with $\lambda$ and with a LoRA-restricted parameter subspace.

**Training randomness.** Two training runs from different seeds differ by about as much as removing a single example; that is the premise of f-INE's hypothesis-testing framing. We estimate the seed-to-seed null with 3–5 oracle retrains at CIFAR scale and report which forget examples remain "influential" at a controlled false-discovery rate.

**The low-rank mechanism (H3).** Write the unlearned model as $\hat\theta + \Delta$ and ask what $\Delta$ does. If $\Delta$ is close to rank-1 per layer, it acts as a steering vector that subtracts the forget-class direction from the output while leaving feature geometry intact. That single property explains all three symptoms at once: near-chance MIA (forget-example loss is pushed up to the held-out level), high probe accuracy (features unchanged) and fast relearning (a rank-1 update is undone by a rank-1 update). The test is to compare the effective rank of $\Delta$ across methods and correlate it with relearning speed; the oracle is the full-rank control. For the diffusion track the same argument applies to the score network: class unlearning that only shifts the conditional score near the forget class's modes leaves the unconditional density unchanged, which we check by measuring how far forget-class samples drift under unconditional generation.

---

## 5. Compute budget (hard cap)

Diffusion track: small UNet DDPM on CIFAR-10 (about 6 T4-hours for a usable FID; a 3-hour schedule is acceptable for relative comparisons). LLM track: Qwen2.5-0.5B with LoRA. Cost assumptions: Colab Pro about US$0.20 per T4-hour; Kaggle about 30 free GPU-hours per week; rented 3090/4090 at US$0.30–0.50 per hour; Lightning AI free monthly credit.

| Experiment | Model | Runs × seeds | GPU-h (full) | Colab Pro | Kaggle free | Rented 3090/4090 | Lightning free |
|---|---|---|---|---|---|---|---|
| D1 Base DDPM training | CIFAR-10 DDPM | 1 × 6 h | 6 | ~$1.20 | $0 | ~$2.50 | within quota |
| D2 Oracle retrains on retain set (seed null for f-INE-style test) | DDPM | 3 × 6 h | 18 | ~$3.60 | $0 | ~$7.50 | no |
| D3 Unlearning: 4 methods × 3 seeds × 3 forget sizes (matched steps) | DDPM | 36 × 0.25 h | 9 | ~$1.80 | $0 | ~$3.60 | partial |
| D4 MIA (multi-timestep losses) + gradient attribution | DDPM | all | 4 | ~$0.80 | $0 | ~$1.60 | within quota |
| D5 Relearning + linear probes + $\Delta\theta$ spectra | DDPM | 36 × 0.1 h | 3.5 | ~$0.70 | $0 | ~$1.40 | within quota |
| L1 LoRA fine-tune on TOFU-style set | 0.5B | 1 × 1 h | 1 | ~$0.20 | $0 | ~$0.40 | within quota |
| L2 Oracle retrains (retain only) | 0.5B | 3 × 1 h | 3 | ~$0.60 | $0 | ~$1.20 | within quota |
| L3 Unlearning: 4 methods × 3 seeds | 0.5B | 12 × 0.3 h | 3.6 | ~$0.70 | $0 | ~$1.50 | within quota |
| L4 Evaluation, MIA, attribution, relearning, probes | 0.5B | all | 5 | ~$1 | $0 | ~$2 | within quota |
| S1 Optional SD-1.5 / SD-Turbo concept erasure (2 methods × 2 concepts) | SD 1.5 | 4 × 2 h | 8 | ~$1.60 | $0 (16 GB ok with 8-bit optimiser) | ~$3.20 | no |
| **Total full version** | | | **~61 (53 without S1)** | **~$12** | **$0 over ~2–3 weeks** | **~$25** | — |

**Minimum viable version (~17 GPU-hours, ≤US$4 on Colab Pro or free on Kaggle).** DDPM on a 3-hour schedule (D1 = 3 h); a single oracle retrain (D2 = 3 h); 3 methods (gradient ascent, label annealing, oracle) × 2 seeds × 1 forget size (D3 = 1.5 h); D4 and D5 on 2k-image subsets (3 h). LLM side: L1, one oracle (L2 = 1 h), 3 methods × 2 seeds (L3 = 1.8 h), L4 on a 200-example evaluation set (1.5 h). S1 dropped. If the DDPM is still too slow, CelebA at 32×32 with a class-conditional attribute as the forget target, or a SmolLM-135M LLM track, are acceptable substitutes.

*What survives at minimum scale:* H1 (MIA AUC, probe accuracy and relearning speed are all large-effect comparisons); H3 for the method-level contrast (approximate versus oracle spectra); H2 as a two-method comparison without the training-randomness null. The full version is required for the f-INE-style hypothesis test (3–5 oracle retrains), the forget-set-size and retain-availability ablations, the H3 correlation across enough cells, and any Stable Diffusion claim.

**Caching rules.** Save the base checkpoint and every unlearned checkpoint (LoRA adapters for the LLM, full UNet weights for the DDPM). Precompute per-example losses at a fixed set of 5 diffusion timesteps once per checkpoint and reuse for MIA. Compute per-example gradients once per checkpoint and store them random-projected to 4k dimensions, so that attribution across all checkpoint pairs is a CPU matrix product. Reuse the same evaluation noise seeds and prompts for every checkpoint so FID and attribution are paired.

---

## 6. Week-by-week plan

- [ ] **W1 — Train base DDPM and LoRA LLM; one oracle retrain per track.** Artifact: training curves, FID/accuracy table; this pre-registered README committed.
- [ ] **W2 — Implement the unlearning methods and the matched-compute protocol.** Artifact: table of forget-rate and retain-utility per method.
- [ ] **W3 — MIA and attribution modules.** Artifact: ROC curves per method; attribution-drop bar chart with CIs.
- [ ] **W4 — Relearning, linear-probe and $\Delta\theta$-spectrum analysis.** Artifact: relearning-curve figure; rank-vs-relearning scatter.
- [ ] **W5 — Additional oracle seeds for the f-INE-style null (full budget only); negative results; preprint v0.** Artifact: Figure 1 and the audit library's README with a "plug in any checkpoint pair" example.

---

## 7. Expected results and how to read either outcome

The expected headline is that MIA-based "success" is a weak certificate: most approximate methods will pass MIA while failing the relearning and probe tests, with the variational method partially closing the gap at a utility cost.

If instead the approximate methods also pass the relearning and influence tests, that is a positive and surprising result worth reporting as "cheap unlearning is enough at this scale". The explanation would be that the forget set is small relative to model capacity, which the forget-set-size ablation would confirm or deny.

If H3's correlation is absent, the spectra will still tell us something: identical spectra across methods would mean low-rankness is not the discriminating property and we would look instead at *which* layers carry the delta.

Either way, the contribution is the audit protocol and the mechanism, and every refuted hypothesis is written up with the explanation we now believe.

---

## 8. Deliverables

1. **Preprint:** *"Suppressed or erased? An attribution-and-membership audit of unlearning in diffusion models and small LLMs."*
2. **This repository as a library:** MIA, attribution, relearning, probe and spectrum modules that accept any pair of original/unlearned checkpoints.
3. **Per-professor one-pagers.** Prathosh: the variational method's position and the f-INE-style testing. Venkatesh Babu and Biswas: the concept-erasure results and the unconditional-drift measurement. Pruthi: membership inference and the privacy framing. Bhattacharyya: the component-level (which layers' deltas carry the unlearning) analysis mirroring DisCEdit/ModHiFi. Chakraborty: holistic post-intervention evaluation in the DynEval spirit.

---

## 9. Research protocol

- **Pre-register before running.** `HYPOTHESES.md` is committed before the first unlearning run.
- **Experiment log.** `logs/EXPERIMENT_LOG.md` records config hash, git commit, expectation, observation and next change for every run.
- **Seeds.** Fixed for training, forget-set sampling, noise and prompts; three seeds (0, 1, 2) behind any plotted number.
- **Uncertainty.** Mean plus 95% bootstrap CI; paired across checkpoints on identical prompts/noise seeds.
- **Matched compute.** Every approximate method gets the same number of gradient steps; $\lambda$ is fixed and reported.
- **Negative results.** The preprint states which hypotheses were refuted and what we now believe instead.
- **Expected-to-fail control.** The "do nothing" baseline must show no forgetting on any axis before any figure is frozen.
- **Figures** come only from single-command-reproducible runs. A CIFAR-10 result is reported as a CIFAR-10 result.

---

## 10. Repository layout (planned)

```
unlearning-audit/
├── README.md
├── HYPOTHESES.md                 # pre-registration, frozen 2026-10-13
├── configs/                      # diffusion/, llm/, one YAML per method × forget-size × seed
├── src/
│   ├── train/                    # base DDPM, LoRA fine-tune, oracle retrains
│   ├── unlearn/                  # gradient ascent, gradient difference, label annealing, variational, NPO
│   └── audit/                    # the library: mia/, attribution/, relearn/, probe/, spectrum/
├── notebooks/                    # weekly artifacts; "plug in any checkpoint pair" demo
├── results/                      # checkpoints, projected gradients, per-timestep losses, figure CSVs
├── logs/EXPERIMENT_LOG.md
└── paper/                        # preprint source, one-pagers/
```

---

## 11. Related work to read first

**IISc papers this audit builds on**

- Prathosh A.P. et al., *Partially Blinded Unlearning: Class Unlearning from Bayesian Perspective*, AAAI 2025; *Concept Forgetting via Label Annealing*, UAI 2025; *Adapt then Unlearn*, TMLR 2025; *Unlearning in Diffusion Models under Data Constraints: A Variational Inference Approach*, TMLR 2026; *f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness*, ICLR 2026; *Efficient Subset Selection Using Text-Guided Data Attribution*, TMLR 2026. [prathosh.in/publications](https://prathosh.in/publications.html)
- R. Venkatesh Babu et al. (VAL), *Concept Siever: Controllable Erasure of Concepts from Diffusion Models*, TMLR 2025; *Balancing Act*, CVPR 2024. [VAL publications](https://val.cds.iisc.ac.in/publications.html)
- S. Biswas et al., *GenErase: Generalizable and Semantically-Aware Concept Erasure in Diffusion Models*, CVPR 2026. [Biswas publications](https://sites.google.com/iisc.ac.in/somabiswas/publications)
- D. Pruthi, P. Maini et al., *STAMP*, ICML 2025; privacy–hallucination trade-off, EMNLP 2026 Findings. [Pruthi publications](https://danishpruthi.com/publications/)
- C. Bhattacharyya et al., *DisCEdit*, NeurIPS 2024; *ModHiFi*, NeurIPS 2025 (Spotlight). [ML Lab publications](http://mllab.csa.iisc.ac.in/publications)
- A. Chakraborty et al., *DynEval*, ECCV 2026. [VCL publications](http://visual-computing.in/publications/)

**Canonical external papers**

- P. Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*, 2024, arXiv:2401.06121.
- R. Zhang et al., *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*, 2024, arXiv:2404.05868.
- P. W. Koh and P. Liang, *Understanding Black-box Predictions via Influence Functions*, ICML 2017, arXiv:1703.04730.
- N. Carlini et al., *Membership Inference Attacks From First Principles* (LiRA), IEEE S&P 2022, arXiv:2112.03570.
- S. M. Park et al., *TRAK: Attributing Model Behavior at Scale*, ICML 2023, arXiv:2303.14186.
- R. Gandikota et al., *Erasing Concepts from Diffusion Models*, ICCV 2023, arXiv:2303.07345.

---

## 12. Faculty this work is aimed at

| Faculty | Dept | Paper this extends |
|---|---|---|
| Prathosh A.P. | ECE | Label annealing (UAI 2025); variational diffusion unlearning (TMLR 2026); f-INE (ICLR 2026) |
| R. Venkatesh Babu | CDS | Concept Siever (TMLR 2025); Balancing Act (CVPR 2024) |
| Soma Biswas | EE | GenErase (CVPR 2026); self-debiasing (WACV 2026) |
| Danish Pruthi | CDS | STAMP (ICML 2025); privacy–hallucination trade-off (EMNLP 2026 Findings) |
| Chiranjib Bhattacharyya | CSA | DisCEdit (NeurIPS 2024); ModHiFi (NeurIPS 2025) |
| Anirban Chakraborty | CDS | DynEval (ECCV 2026) |
| Sriram Ganapathy | EE | Self-influence guided data reweighting (EMNLP 2023) |
| C. S. Seelamantula | EE | Score matching (ICASSP 2025), if the diffusion-score analysis is emphasised |

---

## 13. License and citation

Code: MIT. Data and figures: CC BY 4.0.

```bibtex
@misc{unlearning-audit-2026,
  title  = {Suppressed or erased? An attribution-and-membership audit of unlearning in diffusion models and small LLMs},
  author = {<Your Name>},
  year   = {2026},
  note   = {Pre-registered 2026-10-13. https://github.com/Arubad/unlearning-audit}
}
```
