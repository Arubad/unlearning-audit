# The Retrain Never Saw the Label: an Ill-Posed Reference in Class-Unlearning Audits of Conditional Diffusion Models, and a Calibrated Alternative

*Working draft v2, started 2026-11-23. It replaces v1 ("Suppressed or Erased?", archived at `paper/archive/DRAFT_v1_depth_ladder.md`), whose headline was falsified by our own pre-registered study C1.*

*Every number comes from `paper/RESEARCH_JOURNAL.md` §7.12–7.14 and `results/figures/label_diag.json`. Every prediction tested is in `PREREGISTRATION.md` with its registration time. Study C3 (C3-A, C3-B) was scored on 2026-11-24. The exploratory follow-up C3-K (§5.4.1) ran on 2026-11-24.*

---

## Abstract

**The practice.** Class unlearning in class-conditional generative models is evaluated against *exact retraining*: a model trained from scratch without the forgotten class. The unlearned model is called successful when its outputs or losses under the forgotten label are close to the retrain's.

**The problem.** In models whose label enters through a learned input embedding, the retrain's row for the removed label is never trained.
- It receives no gradient. Under Adam(W) without weight decay it stays *exactly* at its random initialisation.
- Querying the retrain with the removed label therefore measures its response to a seed-specific random vector, not a counterfactual.

**What we show, on a class-conditional DDPM trained on CIFAR-10 with three independent retrains per class:**
- The removed-label "retrain reference" is heterogeneous across retrain seeds, with oracle-pair distances differing by up to 16×.
- It is uncalibrated: a held-out exact retrain scored against the others falls anywhere from −1.9 to 2.2 on a closeness scale where an exact retrain should score 1.
- The null (unconditional, classifier-free-guidance) label is trained in every model and gives a well-posed reference. Its seed floor is far more homogeneous.
- With the null-label reference, held-out retrains score 1.00 / 1.01 / 0.99 (automobile), 0.94 / 1.13 / 0.94 (ship) and 1.08 / 0.97 / 0.96 (truck, a class drawn at random).
- The null-label trace is class-level (it replicates on unseen test images of the class), specific (absent on other classes), and localised at low-to-mid noise.

**How the evidence was collected.**
- The ship results come from a study registered before its data existed (C2). One of its registered predictions, homogeneity at every t ≤ 400, failed narrowly, and we report it as failed.
- The diagnosis grew out of a pre-registered *negative* result (C1): an earlier, replicated single-class finding of ours did not generalise, because of this mechanism.

**Consequence (pre-registered, C3-A).**
- A zero-step edit that changes no weight that ever saw data passes the standard removed-label audit, on closeness *and* behaviour. The edit re-draws the removed label's embedding row.
- It passes on 6/20, 9/20 and 12/20 random draws for ship, automobile and truck. Which draws pass is luck.
- The null-label audit scores every such edit 0 ("nothing removed"), correctly. Re-fitting that one row restores the class to the original's generation rate (≥ 96 %, on all three classes), against ≤ 25 % for exact retrains (exploratory).
- None of five approximate unlearning methods is retrain-equivalent under the null-label audit. Label annealing moves *away* from the retrain, as its mechanism predicts.
- All 12 registered C3-A predictions held.

---

## 1. Introduction

**Retraining is the reference.** Machine unlearning asks for a model that behaves as if some data had never been seen (Cao & Yang 2015; Bourtoule et al. 2021). For deep networks the only exact method is to retrain without the data. Approximate methods are therefore judged by how close they come to a retrain. For class unlearning in diffusion models, the comparison is usually made *under the forgotten label*: generation rates, FID, or losses for prompts or labels of the removed class, against the same quantities for the retrain. Examples include SalUn (Fan et al. 2024), Selective Amnesia (Heng & Soh 2023), EraseDiff (Wu et al. 2024), Score Forgetting Distillation (Chen et al. 2024) and Zheng, Tai & Yeh (2026).

**The question.** What does a retrain do when it is queried with a label it never saw?
- In a class-conditional model with a learned embedding table, that label's row receives no gradient during retraining. It is the random vector it was initialised with. Under Adam(W) with zero weight decay, it is that vector to the bit.
- The retrain's "behaviour on the forgotten class" is then the network's reaction to an untrained random input. It depends on the seed, and nothing in training shapes it.
- Comparing an unlearned model to it is comparing to noise with structure: the network's learned readout of the nine trained rows, applied to a tenth random vector.

**This is a concrete mechanism inside a known concern.** Retrain-referenced evaluation is already known to be fragile:
- Average-case membership tests are miscalibrated (Hayes et al. 2024).
- Retrain outcomes vary across seeds (Triantafillou et al. 2024; Lanyon et al. 2025).
- Output-distribution methods average over several retrains (Ebrahimpour-Boroojeny, Wang & Sundaram 2025; Georgiev et al. 2025).

We identify a specific, checkable cause of the fragility for class removal, we quantify it, and we give a reference that does not have it.

**Contributions.**
1. **Mechanism, verified.**
   - In every one of nine exact retrains (three classes × three seeds), the removed-label row moved ≈ 1e-7 of its norm from initialisation (float32 round-off). Trained rows moved ≈ 0.97 in absolute terms.
   - Under the removed label, retrain seeds disagree heterogeneously across noise levels: max/min oracle-pair distance 3.2–16.1 at t ≤ 400.
   - A held-out retrain is not recognised as a retrain (§5.2).
2. **A calibrated alternative.**
   - Querying the null label, which is trained in every model by classifier-free-guidance dropout, gives a far more homogeneous seed floor.
   - Leave-one-retrain-out calibration holds on both classes with enough signal: automobile (post-hoc) and ship (pre-registered) (§5.3).
3. **Pre-registered evidence, including our own failures.**
   - C1 falsified the general version of an earlier finding of ours: a high-noise-only trace. It turned out to be this mechanism on one class.
   - C2 confirmed the mechanism and the calibration on a fresh class, and failed one homogeneity prediction narrowly.
   - Both are reported with their registered rules applied as written.
4. **Consequence for audits.**
   - The removed-label audit is passed by a zero-step embedding re-draw on every class tested, in 30–60 % of draws. The null-label audit is not fooled.
   - All 12 registered predictions held (§5.4).
   - The null-label calibration replicates on a fourth class drawn at random (truck, §5.5).

---

## 2. Background and related work

**Class-conditional diffusion with classifier-free guidance.**
- A denoiser $\varepsilon_\theta(x_t, t, y)$ is trained with $y$ replaced by a null token with probability $p_\text{uncond}$ (Ho & Salimans 2022). One network thus learns the conditional pathways $y = 1..K$ and an unconditional pathway $y = \varnothing$.
- When $y$ enters through `nn.Embedding(K + 1, d)`, every row, the null row included, is a learned vector initialised i.i.d. $\mathcal N(0, I)$.
- Theory places class commitment at high noise ("speciation", Biroli et al. 2024; Sclocchi, Favero & Wyart 2025). At low and mid noise, denoisers rely on local statistics largely shared across classes (Kadkhodaie et al. 2024; Kamb & Ganguli 2024).

**Unlearning in diffusion models and how it is evaluated.**
- Methods: fine-tuning that redirects a concept (Selective Amnesia, Heng & Soh 2023), saliency-masked ascent (SalUn, Fan et al. 2024), bi-level and distillation approaches (EraseDiff, Wu et al. 2024; SFD, Chen et al. 2024), and others.
- Class-forgetting evaluations compare generation under the forgotten label, or losses, with a retrain. None we found discusses what the retrain's forgotten-label pathway is.
- We also implement "label annealing" as *our reading* of Panda et al. (UAI 2025): the forget label is annealed to the null token. That paper is not shown to concern diffusion models, so we make no claim about the authors' method.

**Fragility of the retrain reference.**
- Hayes et al. (2024) show that retrain-referenced membership tests are miscalibrated in the average case.
- Triantafillou et al. (2024) and Lanyon et al. (2025) study seed distributions of retrains.
- Ebrahimpour-Boroojeny et al. (2025) build class-unlearning targets from retrain outputs.
- In classifiers the *output* row of an absent class does receive gradient, through the softmax normaliser. Zheng et al. (2026) analyse the head bias of an absent class. Our mechanism is specific to *input-side* conditioning.

**Untrained embedding rows elsewhere.** Tokens absent from LLM training data keep near-initial embeddings ("Fishing for Magikarp", Land & Bartolo 2024). We are not aware of this observation being connected to unlearning references.

**Novelty check.** It is described in `RESEARCH_JOURNAL.md` §9.2. It covered arXiv abstracts and a web search only, not full texts.

---

## 3. The mechanism

**Setup.**
- $\theta_0 \sim A(D)$ is the original, $O^{(s)} \sim A(D \setminus D_k; s)$ the exact retrain with seed $s$ on the data minus class $k$, and $e_k^{(s)}$ the retrain's embedding row $k$.

**Proposition (untrained row).**
- If label $k$ never occurs in $D \setminus D_k$, the loss does not depend on $e_k$, so $\nabla_{e_k} L \equiv 0$ at every step.
- With Adam(W), the moments of $e_k$ stay 0. Its update $\eta\, \hat m/(\sqrt{\hat v} + \epsilon)$ is 0, and with weight decay λ the row evolves as $e_k(T) = (1 - \eta\lambda)^T e_k(0)$.
- So $e_k^{(s)}$ is the seed's initial draw, possibly shrunk deterministically. It is independent of the data, and different for every seed.
- Gradient clipping and EMA do not change this. EMA moves the row only by arithmetic drift.

**Consequence.**
- For any query $(x_t, t)$, $\varepsilon_{O^{(s)}}(x_t, t, k)$ is the retrain's response to an untrained random conditioning vector.
- Across seeds these responses differ for two reasons: the networks differ (as for any label), *and* the vectors are independent random draws that no training has aligned.
- The second source does not exist for trained labels. It inflates and destabilises the seed floor in a way that depends on noise level.
- An audit that compares an unlearned model with "the retrain" under label $k$ compares it with one draw from this distribution.

**Prediction.** Removed-label retrain references show (i) a heterogeneous oracle-pair floor, and (ii) poor leave-one-retrain-out calibration. Null-label references should show neither, because the null row is trained in every model.

---

## 4. Audit statistic and calibration test

**Losses and distance.**
- $\ell_m(q, t)$ is the per-example ε-MSE of model $m$ on query image $q$ at timestep $t$, conditioned on label $y$ (forget label or null).
- Noise is fixed per (example, timestep) across models (common random numbers).
- $d(a, b)$ is the RMS difference of losses over queries and audit timesteps $t \in \{50, 200, 400, 600, 800\}$.

**Closeness.**
$$E(u) = \frac{d(\theta_0, O) - d(u, O)}{d(\theta_0, O) - d_\text{floor}},$$
- $d(\cdot, O)$ is averaged over the reference retrains, and $d_\text{floor}$ is the mean retrain-pair distance.
- $E = 0$ for doing nothing and $E = 1$ for a model as close to the references as they are to each other.
- $E$ is reported as undefined when $\text{SNR} = d(\theta_0, O)/d_\text{floor} < 1.2$. That means removal left no measurable trace on these queries.
- 95 % CIs come from a bootstrap over queries.
- A *standardised* variant divides each timestep's losses by that timestep's floor before computing $d$ (journal §7.7).

**Calibration test (leave-one-retrain-out).**
- With three retrains, score each one against the other two.
- If the reference is well-posed, a held-out exact retrain is exchangeable with the references and must score $E \approx 1$.
- This is a falsifiable property of the *reference*, independent of any unlearning method.

**Heterogeneity.**
- At each timestep we compute the max/min ratio of the three retrain-pair distances.
- Under a homogeneous floor every pair is equally far apart, giving a ratio near 1.

---

## 5. Results

### 5.0 Setup
**Model.**
- Class-conditional DDPM, 6.48M-parameter U-Net (channels 64·(1, 2, 2), attention at 16×16), 1000 linear-β steps.
- Trained 40k steps, batch 128, AdamW (lr 2e-4, **weight decay 0**), EMA 0.9999, CFG dropout 0.1.
- CIFAR-10. The label embedding is `nn.Embedding(11, 256)` with the null token at index 10.

**Retrains.**
- Three exact retrains per removed class (seeds 0, 1, 2), same recipe, on CIFAR-10 minus the whole class.
- Classes: automobile (1) and cat (3), where the diagnosis was found post-hoc; ship (8), pre-registered study C2; truck (9, drawn at random), study C3-B.

**Queries.** 256 training images of the removed class, 256 test images of it, 256 images from semantically near classes and 256 from far classes. Every model and condition sees the same images and noise.

**Compute.** Free-tier GPUs (Kaggle 2×T4, ≈ 3.2 GPU-h per retrain). Audits ran on a laptop GPU (Apple M4).

### 5.1 The removed-label row is untrained (all twelve retrains)
| class | removed row moved from init (relative) | trained rows moved (mean, absolute) |
|---|---|---|
| automobile | ≈ 1e-7 | ≈ 0.97 |
| cat | ≈ 1e-7 | ≈ 0.97 |
| ship | 2.1–3.4e-6 absolute, ≈ 1e-7 relative | 0.96–0.99 |
| truck | 1.8–3.2e-6 absolute, ≈ 1e-7 relative | 0.94–0.99 |

Re-initialising each retrain from its seed reproduces the row to float32 round-off between torch builds. A side observation: trained rows move only ≈ 6 % of their norm. A trained embedding is its random init plus a small learned offset, and the network learns to read the vectors it was given.

### 5.2 The removed-label reference is ill-posed
| | automobile | cat | ship (pre-registered) | truck (pre-registered, random draw) |
|---|---|---|---|---|
| max/min retrain-pair distance, t ≤ 400 | 16.1 | 4.4 | 3.2 (and 16.2 at t = 900) | 6.6 |
| held-out retrain, plain E | undefined, 2.19, −0.97 | undefined ×3 | undefined, 0.61, −0.78 | undefined, −2.08, undefined |
| held-out retrain, standardised E | −1.14, undefined, −0.63 | 0.92, −0.12, 1.20 | 1.81, 1.14, −1.92 | 0.56, −0.95, 0.86 |

- For ship (C2) and truck (C3-B), both predictions were registered: heterogeneity ≥ 2.0 at some t ≤ 400 (C2-F) and at least one held-out E outside [0.6, 1.4] or undefined (C2-G). Both held on both classes.
- No class gives a reference that recognises its own retrains.

### 5.3 The null-label reference is well-posed and calibrated
| | automobile | cat | ship (pre-registered) | truck (pre-registered) |
|---|---|---|---|---|
| max/min retrain-pair distance, t ≤ 400 | 1.23 | ≤ 1.2 | 1.42 (≤ 1.24 at t ≤ 300) | 1.49 (at t = 10; ≤ 1.19 at 50–400) |
| plain SNR, train / test images | 3.74 / 3.47 | 1.21 / 1.15 | 1.85 / 1.54 | 2.44 / 2.20 |
| held-out retrain plain E [95 % CI] | 1.00 [0.97, 1.03], 1.01 [0.98, 1.04], 0.99 [0.96, 1.02] | undefined, undefined, 0.80 | 0.94 [0.86, 1.03], 1.13 [1.04, 1.22], 0.94 [0.87, 1.02] | 1.08 [1.03, 1.13], 0.97 [0.92, 1.02], 0.96 [0.91, 1.01] |
| near / far classes SNR | 1.14 / 1.07 | 1.05 / 1.05 | 1.08 / 1.03 | 1.01 / 0.91 |
| per-timestep signal peak | 5.45 at t = 200 | — | 2.06 at t = 200 | 3.01 at t = 100 |

**Registered C2 outcome (ship).**
- Held: trace present (C2-0, SNR 1.85 ≥ 1.5), calibration (C2-A), class-level (C2-C, test within 20 % of train), specificity (C2-D) and low/mid-noise localisation (C2-E).
- **Failed: homogeneity at every t ≤ 400 (C2-B).** The ratio is 1.42 at t = 400 against a 1.3 threshold.
- Under the registered conjunctive rule, claim (b) is therefore "not confirmed as registered".
- We draw the narrower conclusion the data support: the null-label floor is markedly *more* homogeneous than the removed-label floor at every t ≤ 400 (1.42 vs 2.13 at t = 400; 1.04 vs 3.23 at t = 200), and that sufficed for calibration.

**Statistic choice.**
- The plain statistic calibrates on both classes.
- The standardised one does not on ship (held-out E 0.58, 1.33, 0.57) or on truck (0.57, 1.15, 0.93); both are exploratory. Standardising up-weights high-noise timesteps, where the null trace is absent and three-retrain floor estimates are noisy.
- We recommend plain closeness on the null label.

**Power is class-dependent.** Trace sizes are automobile 3.7, truck 2.4, ship 1.9, cat 1.2. For cat, the null-label audit cannot resolve removal at n = 256.
- One reading: the unconditional denoiser at low and mid noise uses local statistics shared across classes, so removing a class leaves a trace only where the other classes do not cover its statistics. Cat is plausibly covered by the other animals.
- This is a conjecture. Four classes cannot test it.

### 5.4 Consequence for auditing unlearning methods (pre-registered, C3-A)
**Design** (registered 2026-11-23, before any C3 data).
- For each class we audit:
  - doing nothing;
  - 20 zero-step *embedding re-draws* (row $k \leftarrow \mathcal N(0, I)$; every other weight is the original's);
  - five approximate methods, gradient ascent (GA), gradient difference (GD), label annealing (LA), variational EWC (VEWC) and retain fine-tune (RF), each with 2 seeds at 2000 steps and lr 3e-5.
- A model *passes the removed-label audit* if removed-label closeness $E \in [0.5, 1.5]$ AND its gated forget-class generation rate is ≤ the worst retrain's + 0.05.
- The statistic is standardised E, or plain E for automobile, whose standardised signal is too weak (SNR 1.18 < 1.2; amendment registered before data).

| | ship | automobile | truck |
|---|---|---|---|
| retrains' gated forget-class rate | 0.000–0.005 | 0.000–0.010 | 0.000–0.035 |
| **re-draws passing the removed-label audit (C3-A1, ≥ 1)** | **6 / 20** | **9 / 20** | **12 / 20** |
| re-draws' removed-label E: SD (range) (C3-A2, ≥ 0.3) | 4.77 (−15.2 … 1.18) | 0.72 (0.31 … 2.64) | 3.03 (−9.6 … 1.52) |
| re-draws' null-label E | 0 (all 20; by construction) | 0 | 0 |
| best method null-label E, lower CI (C3-A3, < 0.8) | −0.85 | −0.29 | 0.16 (GD s1: 0.24 [0.16, 0.32]) |
| LA null-label E, s0 / s1 (C3-A4, < 0) | −1.87 / −1.87 | −0.79 / −0.81 | −1.33 / −1.34 |

**All 12 registered predictions held.**

**Reading.**
- *The removed-label audit can be passed without unlearning.*
  - A re-draw changes 256 numbers that no data ever shaped, yet on every class it passes the retrain-referenced audit in 30–60 % of draws.
  - Its losses under the removed label sit as close to the retrains as the retrains sit to each other, and it no longer generates the class under that label.
  - This is exactly what §3 predicts. A retrain's removed-label response is a network's reaction to a random row, and a re-draw is another sample from the same distribution.
  - Which re-draws pass is a coin flip, because the reference itself is one random draw.
- *The null-label audit is not fooled, but that part is by construction.* The re-draw leaves the null pathway untouched, so its null-label E is 0. §5.4.1 tests directly whether the forgotten class is still *recoverable* from such a model. It is: on all three classes, re-fitting the single row restores ≥ 96 % forget-class generation, against ≤ 25 % for retrains.
- *Methods.*
  - Under the null-label audit, no method is retrain-equivalent.
  - LA moves the unconditional pathway away from the retrain, because it trains the null label on the forget images.
- *Exploratory, not registered: the methods at these hyper-parameters.*
  - GA and VEWC reach a zero forget rate by collapsing the model (|E| in the hundreds to thousands on both audits).
  - GD collapses on the removed label but is the closest method on the null label (−45.7 … 0.24).
  - LA and RF barely forget (gated rate 0.89–1.00).
  - So the methods' rankings here say little about the methods in general. The registered claims are about the *audit*.

#### 5.4.1 Is the class still in a re-drawn model? (exploratory; ship, automobile, truck)
**Test** (predictions written before running; `EXPERIMENT_LOG.md`, C3-K).
- Freeze every weight. Optimise only the removed label's row (256 numbers; Adam, lr 0.05, 400 steps, batch 64) on 1,000 class images disjoint from the audit set.
- Then measure the gated forget-class generation rate under that label.
- The models are the first three re-draws of each class that **passed** the removed-label audit in C3-A, and the three exact retrains.

| class | original | passing re-draws: before → after re-fitting one row | exact retrains: before → after | unconditional rate: original vs retrains |
|---|---|---|---|---|
| ship | 0.995 | 0.000, 0.005, 0.000 → **0.985, 0.980, 0.980** | 0.000–0.005 → 0.125–0.190 | 0.083 vs 0.013–0.019 |
| automobile | 0.985 | 0.000, 0.010, 0.035 → **0.985, 0.985, 0.990** | 0.000–0.010 → 0.085–0.190 | 0.030 vs 0.003–0.007 |
| truck | 0.975 | 0.000, 0.005, 0.000 → **0.960, 0.975, 0.955** | 0.000–0.035 → 0.155–0.245 | 0.029 vs 0.002–0.006 |

**Reading.**
- On every class, re-fitting 256 numbers restores the forgotten class to the original's rate (9 / 9 re-draws ≥ 0.955), in models the removed-label audit had certified as retrain-like.
- The same re-fit brings exact retrains only to 8.5–24.5 % (9 / 9 ≤ 0.245).
- The unconditional pathway, which the null-label audit queries, carries the class at 3.4–14× the retrains' rate.
- So the null-label audit's "nothing removed" verdict for re-draws is correct, not merely true by construction. The data's influence is intact and one row away from full recovery.
- Limits: three re-draws per class, one re-fit budget, exploratory.

### 5.5 A fourth class drawn at random (pre-registered, C3-B)
- Truck was drawn by `numpy.random.default_rng(20261123)` from the unused classes, before any data. Three new retrains were trained.
- The C2 rules applied unchanged, except that the homogeneity rule C2-B was replaced, before truck data existed, by C3-B-B: at every t ≤ 400, the null-label pair ratio is ≤ 1.5 and below the removed-label ratio.
- **Results** (columns "truck" in §5.1–5.3):
  - The removed row is untrained (1.8–3.2e-6).
  - The removed-label reference is heterogeneous (6.6) and uncalibrated (standardised E 0.56, −0.95, 0.86).
  - The null-label trace is present (SNR 2.44), class-level (test 2.20) and specific (near/far 1.01/0.91).
  - The null-label reference is calibrated: held-out E 1.08, 0.97, 0.96, every CI containing 1.
  - C3-B-B held, with a thin margin: 1.49 at t = 10 against 1.5. Every other t ≤ 400 is ≤ 1.19.
- **Claim (b) is confirmed on truck.** The original C2-B rule (≤ 1.3) also fails on truck, as on ship.
- Across the four classes, the null-label audit had power and was calibrated on three (automobile, ship, truck) and lacked power on one (cat).

### 5.6 How we got here: a pre-registered negative result
**The earlier finding.**
- Our earlier headline (F1) said the exact-retrain trace of class removal lives only at high noise, so per-timestep standardisation is needed to see it.
- It came from cat with two, then three, retrains and replicated on test images.

**The pre-registered test (C1, fresh class automobile).**
- No high-noise trace (C1-1 to C1-4 failed).
- Even on cat, the standardised statistic did not recognise a held-out retrain (E −0.18).

**The diagnosis.** Diagnosing C1 led to §3: the cat pattern was trained-vs-random conditioning on that class, not a property of class removal.

**The lesson.** A clean, replicated single-class pattern with null controls can still be an artefact of the reference. Only a fresh-class confirmatory test exposed it.

### 5.7 Secondary findings (exploratory)
- **Behaviour is easy, closeness is not.**
  - On cat, 14 of 20 zero-step re-draws of the removed row reduce the gated cat-generation rate into the retrains' range.
  - Only 3 of 20 land in the retrains' standardised-closeness band (journal §7.8).
- **Classifier-based forgetting rates must be gated for degenerate outputs.** Gradient-difference noise outputs are labelled "cat" 65 % of the time by the CIFAR classifier (journal §7.11).

---

## 6. Discussion and recommendations

**For auditors of class unlearning in conditional diffusion models:**
1. **Do not use the retrain's removed-label pathway as a reference.** It can be matched by re-drawing one embedding row (§5.4). If a removed-label comparison is reported, give the retrain-seed spread, a leave-one-retrain-out calibration, and the pass rate of zero-step re-draws as a negative control.
2. **Audit the unconditional pathway against several retrains, and check calibration.** A held-out retrain should score ≈ 1. Where SNR < 1.2, report "no measurable trace" rather than a number.
3. **State the conditioning architecture and optimiser.**
   - Input-side learned embeddings are affected, with or without weight decay: decay shrinks the row but does not train it.
   - Frozen text encoders (text-to-image) are not affected in this way, because the prompt embedding is trained by no one.
   - Output-side class heads in classifiers receive gradient and are not affected.

**Alternatives we did not test.**
- Share the removed row's initialisation across retrain seeds. That removes the seed dependence, but the row is still untrained.
- Average over many retrains, as distribution-level references do.

---

## 7. Limitations
- **Scale.** One architecture (6.5M-parameter U-Net), one dataset (CIFAR-10), one training length. Three retrains per class means the floor is estimated from three pairs.
- **Classes.** The mechanism is verified on twelve retrains of four classes. Calibration of the null-label audit holds on the three classes with enough signal, two of them confirmatory (ship, truck).
- **Power.** The null-label audit has low power when the removed class is covered by the others (cat).
- **The homogeneity prediction failed on ship as registered.** The revised rule held on truck only narrowly (1.49 against 1.5). Our "more homogeneous" claim is weaker than the original registered one.
- **Methods.** The five approximate methods ran at one hyper-parameter setting, where they either collapse or barely forget. The C3-A claims concern the audit, not method rankings.
- **The null-label audit is blind to label-only edits by construction.** It certifies that the *unconditional* pathway is unchanged. An edit confined to the removed label's row is reported as "nothing removed". This is the correct verdict for data removal, but the audit says nothing about label-conditioned behaviour, which must be checked separately.
- **Platform.** Retrains ran on Kaggle, the original on Colab, with different torch builds. The relative-movement check absorbs float32 round-off, and neither difference affects which rows are trained.

## 8. Pre-registration and reproducibility
- All confirmatory studies (C1, C2, C3) and post-hoc amendments are in `PREREGISTRATION.md`, with timestamps relative to data collection.
- Scoring code (`src/analysis/c1_report.py`, `label_diag.py`, `c3_consequence.py`) is unit-tested on hand-built inputs (`tests/test_audit_cores.py`).
- Experiment provenance is in `EXPERIMENT_LOG.md`.

---

## References (arXiv IDs in `RESEARCH_JOURNAL.md` §8 and §9.2)
Biroli et al. 2024 · Bourtoule et al. 2021 · Cao & Yang 2015 · Chen et al. 2024 (SFD) · Ebrahimpour-Boroojeny, Wang & Sundaram 2025 · Fan et al. 2024 (SalUn) · Georgiev et al. 2025 (KLOM) · Hayes et al. 2024 · Heng & Soh 2023 · Ho & Salimans 2022 · Kadkhodaie et al. 2024 · Kamb & Ganguli 2024 · Land & Bartolo 2024 · Lanyon et al. 2025 · Panda et al. 2025 (UAI; our reading) · Sclocchi, Favero & Wyart 2025 · Triantafillou et al. 2024 · Wu et al. 2024 (EraseDiff) · Zheng, Chen, Guo & Xiao 2026 · Zheng, Tai & Yeh 2026
