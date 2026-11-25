# Research journal — *Suppressed or erased?* Auditing approximate unlearning in diffusion models

A living research record: the formal question, the theory behind every measurement, predictions written **before** the corresponding data existed, every design decision with its reason, an honest account of what went wrong in the process, and results as they arrive. It complements the other documents rather than replacing them:

| Document | Role |
|---|---|
| [`PREREGISTRATION.md`](../PREREGISTRATION.md) | The frozen hypotheses and numeric tests, plus dated amendments |
| [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) | One entry per GPU run: config hash, expectation, observation, next change |
| [`REVIEW.md`](../docs/history/REVIEW_2026-10-18.md) | Novelty and citation check against the literature |
| **this file** | The argument: why each measurement means what we claim it means, and what the results say |

Last updated: 2026-11-24. All earlier **[pending]** markers are resolved; each now points to the section with the result.

---

## 1. The question, stated formally

Let $A$ be a (randomised) training algorithm, $D$ the training set, $D_f \subset D$ the forget set and $D_r = D \setminus D_f$ the retain set. Training gives $\theta_0 \sim A(D)$. An unlearning method $U$ maps $\theta_0$ (and possibly $D_f$, $D_r$) to $\theta_u = U(\theta_0, D_f, D_r)$.

**Exact unlearning** means the distribution of $\theta_u$ equals that of $A(D_r)$, the *oracle* retrain (Cao & Yang 2015; Ginart et al. 2019; Bourtoule et al. 2021). **Certified approximate unlearning** relaxes equality to $(\varepsilon,\delta)$-indistinguishability (Guo et al. 2020; Sekhari et al. 2021). Every practical method for deep networks (gradient ascent, gradient difference, label remapping, EWC-regularised variants, NPO) carries no such guarantee; it is judged empirically, usually by one statistic of the forget set.

Two consequences frame the whole project:

1. **The right reference is the oracle *distribution*, not the original model and not zero.** A statistic "after unlearning" means nothing until it is compared with what a retrain produces, and a retrain is itself random. With two oracle seeds we can estimate the seed-to-seed spread (the *noise floor*), which is the smallest gap that can be called a gap. This is the premise of f-INE (ICLR 2026) and of KLOM (Georgiev et al. 2024), and it is why the 2026-11-18 amendment moved the second oracle seed from stretch into the core design.
2. **Unlearning cannot be verified from one statistic.** Thudi et al. (2022) show that, from weights alone, approximate unlearning is not auditable in general; Hayes et al. (2024) show that average-case MIA evaluations give a false sense of privacy. Our design responds by measuring *many* functionals of the model at increasing "depth" and asking, for each, how close the unlearned model is to the oracle relative to the floor.

**Research question.** For class unlearning in a class-conditional DDPM (CIFAR-10, forget class 3 = cat), at which depth does each method stop being indistinguishable from a retrain, and what property of the update $\Delta = \theta_u - \theta_0$ explains how quickly the forgotten class comes back?

---

## 2. The depth-of-erasure ladder (the paper's organising idea)

Each audit axis tests a strictly stronger notion of "forgotten". A method can pass the shallow rungs and fail the deep ones; the claim of the paper is about *where* each method fails.

| Rung | Measurement | Passes if… | What passing does **not** rule out |
|---|---|---|---|
| 0. Behaviour | forget-class generation rate (classifier on samples) | the model stops producing cats | the knowledge is still in the weights, only gated |
| 1. Average-case membership | loss-based MIA, SecMI | forget-example losses look like held-out losses | per-example memorisation; class knowledge |
| 2. Calibrated membership | offline LiRA with the oracles as the OUT distribution | per-example losses are where a retrain would put them | representation-level knowledge |
| 3. Representation | linear probe on null-conditioned mid-block features | cat-vs-rest separability is no higher than in the oracle | latent knowledge cheaply re-exposed by fine-tuning |
| 4. Recoverability | relearning time on 5 % of the forget set; alignment $\cos(g_f(\theta_u), \Delta)$ | relearning is as slow as from the oracle | the downstream *imprint* of the data on other classes |
| 5. Influence / fingerprint | oracle-normalised closeness $E$ on near/far retain classes | other classes moved the way a retrain moves them | — (closest to the definition of exact unlearning) |

Rung 5 is the operational version of the definition in §1: exact unlearning implies indistinguishability of *every* functional, including the model's behaviour on data it was never asked to forget.

### 2.1 Rung 0–2: why a passing MIA is weak evidence

Loss-based MIA (Shokri et al. 2017; Yeom et al. 2018) thresholds the per-example loss. Unlearning methods are typically tuned until this test is at chance, so passing it is close to circular. LiRA (Carlini et al. 2022) replaces the global threshold with a per-example likelihood ratio against models that did not train on the example; we use the oracle retrains as those OUT models (an "offline" LiRA), so the test asks whether each forget example's loss is where a retrain would put it, not merely whether it is high. For diffusion models, SecMI (Duan et al. 2023) and the extraction results of Carlini et al. (2023) establish that DDPMs memorise individual training images, which is what gives these attacks power.

**A risk we predict, with its reason (written before results).** Kadkhodaie et al. (2024) show that diffusion models trained on enough data enter a *generalisation regime* in which models trained on disjoint subsets learn nearly the same score function, and memorisation (hence MIA power) collapses. Our base model has 6.5M parameters and saw ≈ 102 epochs of 50k images, far less than the 800k-step models in which Carlini et al. found strong memorisation. **Prediction P1:** loss-MIA AUC on the *original* model (do-nothing cell) will be modest, ≈ 0.52–0.62. If it is ≤ 0.55 the pre-registered expected-to-fail control fails, H1's "MIA ≤ 0.55" criterion becomes vacuous (every model, including the untouched one, satisfies it), and we report this as a finding about evaluation rather than tuning the model to memorise more. Rungs 3–5 do not depend on memorisation and remain informative either way.

**Corollary (class regime).** When the whole class is removed, the oracle has never seen *any* cat, so it treats training cats and test cats alike. A method that truly reproduces the oracle should therefore move held-out cats as much as forget cats: $E_{\text{heldout}} \approx E_{\text{forget}}$. **Prediction P2:** methods that optimise the forget-set loss directly (gradient ascent, gradient difference) will show $E_{\text{forget}} > E_{\text{heldout}}$, meaning they act on the specific training images rather than on the class. That is a signature of targeting the certificate, not the counterfactual.

### 2.2 Rung 3: probes need an oracle reference

A linear probe (Alain & Bengio 2016) measures linearly decodable information, not information the model *uses* (Belinkov 2022). A DDPM that never saw a cat can still separate cat images from other images in its features, because cats share low-level statistics with dogs and other animals. "Probe accuracy ≥ 80 % of the original" is therefore not evidence of suppression unless the probe also beats the **oracle's** probe. The 2026-11-18 amendment adds that requirement. Features are extracted with null-class conditioning, because conditioning on the true label would leak the label into the features: this was a bug fixed in review.

### 2.3 Rung 4: why relearning is fast after suppression, a first-order argument

Write $\theta_u = \theta_0 + \Delta$. Relearning runs gradient descent on the forget loss $L_f$ from $\theta_u$. To first order, one step moves by $-\eta\, g_f(\theta_u)$ with $g_f = \nabla L_f$. How far relearning must travel depends on how much of $\Delta$ lies along directions the forget gradient points back through:

$$a(\Delta) = \cos\big(g_f(\theta_u),\, \Delta\big).$$

If unlearning climbed the forget loss along a few directions, as gradient ascent on a single class does, then $\Delta$ points up the forget loss and $g_f(\theta_u)$ still points the same way ($a \to +1$). The descent step $-\eta\, g_f$ then walks straight back along $-\Delta$, so relearning is a short walk back to $\theta_0$. A retrain lands in an unrelated region of weight space: $\theta_{\text{oracle}} - \theta_0$ is the difference of two independent 40k-step runs, and is not aligned with $g_f$.

This is the mechanism behind three literatures that report the same phenomenon in other settings:
- fine-tuning learns minimal "wrappers" on top of capabilities, which are removed by a little further training (Jain et al. 2024);
- safety alignment is "a few tokens deep" (Qi et al. 2024);
- LLM unlearning methods leave the information recoverable from the weights (Deeb & Roger 2024; Łucki et al. 2024; Hu et al. 2025).

Relearning time as an unlearning metric goes back to Golatkar et al. (2020). *From Dormant to Deleted* (2025) predicts relearning from the $L_2$ distance $\|\Delta\|$, which is why every comparison in H3 is made at matched norm.

### 2.4 Rung 4, mechanism: why the update should be low-rank (H3)

The claim is that approximate unlearning updates are effectively low-rank per layer, and that low rank (at matched norm) predicts fast relearning. It rests on three results:
- **Optimisation concentrates in a tiny subspace.** Gur-Ari et al. (2018) show gradients quickly align with the top Hessian eigenspace. Li et al. (2018) and Aghajanyan et al. (2021) show tasks are solvable in a low intrinsic dimension, which is why LoRA (Hu et al. 2021) works.
- **The gradient structure of a single-class objective.** In a linear layer, the per-example gradient is an outer product $\delta\, x^\top$. When every example comes from one class and passes through one class embedding, these outer products share directions, so a short run of ascent accumulates a sum dominated by a few of them.
- **A low-rank update is undone by a low-rank update.** If $\Delta$ is rank-$k$ and aligned with $g_f$, relearning only has to recover $k$ directions. A retrain's delta is full-rank and unaligned.

Two confounds shape the test.
- **Rank and norm are confounded.** A small update relearns fast whatever its rank, so we use partial Spearman controlling for $\|\Delta\|_F$, plus a matched-steps `retain_finetune` control: plain descent on retain data, same steps and learning rate. Any short optimisation run yields a low-rank delta; H3 needs approximate methods to be lower-rank than *that*.
- **Correlation across about a dozen cells confounds rank with method identity.** We therefore add an intervention: replace $\Delta$ by its per-layer rank-$k$ truncation, rescale to the same total Frobenius norm, and re-measure forgetting and relearning for $k \in \{1, 4, 16, 64, \text{full}\}$. This varies rank while holding method and norm fixed, which is the causal version of H3.

Effective rank is $\exp H(p)$ with $p_i = s_i / \sum_j s_j$ (Roy & Vetterli 2007), aggregated across layers weighted by norm.

### 2.5 Rung 5: why the original influence statistic had to be replaced (H2)

The pre-registered H2 statistic was the drop in mean |TRAK score| of forget examples on retain queries, predicted to be > 90 % under retraining. TRAK (Park et al. 2023) estimates, *at the checkpoint where it is evaluated*, the effect on query $q$ of up-weighting training example $f$. That quantity is defined and generally non-zero for a model that never saw $f$: at the oracle it estimates the effect of *adding $f$ back*.
- Forget examples are fitted by the original model, so their gradients are small.
- They are unfitted by the oracle, so their gradients are larger.

So |TRAK| is expected to *grow* under exact retraining. The pre-registered prediction could not be met by the oracle itself, and the statistic could not discriminate methods. The smoke test confirmed the sign: `attr_drop = −1.95` at the oracle. The underlying problem is the known fragility of influence functions in non-convex deep networks (Basu et al. 2021): Bae et al. (2022) show they approximate a proximal response, not leave-one-out retraining.

**Replacement: oracle-normalised closeness.** For a query set $Q$, let $\ell_m(q,t)$ be the per-example $\epsilon$-MSE of model $m$ at a fixed timestep $t$ and fixed noise draw. Define

$$d(a,b) = \sqrt{\tfrac{1}{|Q||T|}\textstyle\sum_{q,t} \big(\ell_a(q,t) - \ell_b(q,t)\big)^2}, \qquad E(u) = \frac{d(\theta_0, O) - d(\theta_u, O)}{d(\theta_0, O) - d_{\text{floor}}}.$$

Here $O$ are the oracle retrains (distances averaged over them) and $d_{\text{floor}} = d(O_0, O_1)$.
- **Values.** $E = 0$ means no closer to a retrain than doing nothing. $E = 1$ means as close as an independent retrain. $E < 0$ means moved *away* from the retrain (over-forgetting, collateral damage), which the old statistic could not express.
- **Query sets.** $E$ is computed on four sets: forget (members); heldout (test cats); **near** (retain images of bird, deer, dog, frog, horse); **far** (airplane, automobile, ship, truck).
- **Confidence intervals.** These bootstrap over query examples.
- **SNR guard.** When $d(\theta_0, O)/d_{\text{floor}} < 1.2$, removing the forget set left no measurable trace on that query set and $E$ is reported as undefined. The smoke test showed $E$ at ±600 without this guard.

**Why "near" is the headline.** A retrain without cats changes the features cats shared with the other animals: dog fur and texture statistics were partly learned from cat images. That imprint on the *near* classes is a fingerprint of the data's influence which a retrain carries and pure suppression does not. **H2 (as amended):** approximate methods reach substantial $E_{\text{forget}}$ (the certificate they are tuned for) but low $E_{\text{near}}$.
- **Prediction P3.** $E_{\text{near}} < 0.5$ for all approximate methods, while $E_{\text{forget}} - E_{\text{near}} \ge 0.25$.
- **Prediction P4.** $d(\theta_0, O)$ is larger on near than on far classes, so $E_{\text{far}}$ is more often undefined by the SNR guard. Removing cats barely touches trucks.

The influence-function prediction survives as a *signed* secondary readout. TRAK at the original model predicts each near-query's loss change from removing the forget set, and we correlate that prediction with the actual change, $\ell_u - \ell_0$, and with the oracle's change. corr(oracle change, prediction) is an LDS-style validity check of the attribution tool itself (Ilyas et al. 2022; Park et al. 2023). If it is near zero, the paper reports that influence estimates are uninformative at this scale, rather than building on them.

This construction is the free-tier analogue of KLOM (Georgiev et al. 2024, *Attribute-to-Delete*), which compares output distributions to many oracles. We use two oracle seeds and losses instead of margins.

**Properties that hold by construction (used as checks, not findings):**
- **Do-nothing cell:** $E = 0$ on every set.
- **Oracle-seed-$s$ cell:** it is compared with the *other* oracle only. The floor is then 0 and $E = 1 - d(O_s, O_{s'})/d(\theta_0, O_{s'})$, so this cell reports the signal-to-floor ratio rather than 1.

---

## 3. Hypotheses, mapped to rungs, with predictions made before the data

| | Claim | Rungs | Pre-registered test (amended 2026-11-18) |
|---|---|---|---|
| **H1** | Suppression, not erasure (replication in diffusion; class regime) | 1, 3, 4 | GA and label annealing: MIA ≤ 0.55 AND probe ≥ 80 % of do-nothing AND probe > oracle's probe AND relearn ≤ 1/5 of the oracle's, in ≥ 2 of 2 seeds |
| **H2** | The data's influence on other classes survives (headline) | 5 | $E_{\text{near}} < 0.5$ with CI below 0.5, AND $E_{\text{forget}} - E_{\text{near}} \ge 0.25$ with non-overlapping CIs; refuted if $E_{\text{near}} \ge 0.8$ |
| **H3** | A low-rank, gradient-aligned update predicts fast relearning at matched norm | 4 | top-5 energy > 0.7 (approx.) vs < 0.3 (oracle) AND ≥ 0.2 above `retain_finetune`; partial Spearman(rank, relearn \| norm) ≥ 0.7; plus the truncation intervention |

**Per-method predictions (class regime)**, recorded 2026-11-19 before any sweep cell had finished. Each prediction comes with its reason.

| Method | $E_{\text{forget}}$ | $E_{\text{near}}$ | Probe vs oracle | Relearn vs oracle | Top-5 energy | Reason |
|---|---|---|---|---|---|---|
| gradient_ascent | possibly **< 0** | ≈ 0 or < 0 | above | ≪ 1/5 | high | Unbounded ascent overshoots: forget losses are pushed *past* where a retrain puts them (SalUn, Fan et al. 2024, documents GA instability in diffusion). Retain classes receive collateral damage, not the oracle's imprint. |
| gradient_difference | 0–0.5 | ≈ 0 | above | ≪ 1/5 | high | Same ascent direction, but the retain term limits collateral damage; still no mechanism to reproduce the imprint |
| label_annealing | 0.2–0.6 | low | above | fast | moderate | Remaps cat to the null-class (unconditional) prediction. The oracle's class-3 embedding is *untrained*, so label 3 behaves like an arbitrary, unused condition rather than the unconditional model. This is close but not equal, which is why E is partial. |
| variational_ewc | between GA and oracle | low | above | faster than oracle | high | Fisher-weighted anchoring (Kirkpatrick et al. 2017; cf. Selective Amnesia, Heng & Soh 2023) protects retain behaviour; the pre-registered secondary prediction is "in between" |
| retain_finetune (control) | ≈ 0 | ≈ 0, slightly > 0 | ≈ do-nothing | ≈ 0 steps | a short-run baseline | Never touches cats; only extra training on retain data. Any small $E_{\text{near}} > 0$ is "more training on retain", which is also part of what a retrain does. |
| do_nothing | 0 (by construction) | 0 | = original | 0 steps | n/a ($\Delta = 0$) | Expected-to-fail control |

**Prediction P5 (H3 intervention).** For gradient ascent, rank 1–4 truncations at matched norm reproduce most of the forgetting (a low forget-class rate), and relearning time does not decrease as k grows. If instead forgetting needs high k, the update is not low-rank in the functional sense, and H3's mechanism is wrong even if the spectra look concentrated.

**Exploratory predictions for Amendment A3** (written 2026-11-21 12:50 IST, while run v5 was queued and no A3 data existed; scored like P1–P5 but labelled exploratory, since the amendment itself is post-hoc):

- **A3-a — reachability.**
  - *Prediction.*
    - Gradient ascent reaches the target (cat rate ≤ 0.10, retain accuracy ≥ 0.90) only at the lower learning rates, inside a narrow window.
    - The step at which the cat rate first drops to 0.10 or below precedes the step at which retain accuracy falls below 0.90 by at most 2 evaluation intervals (≤ 400 steps).
    - Gradient difference and variational EWC reach the target more easily: gradient difference at lr 3e-4, variational EWC at ewc_weight ≥ 1e3.
  - *Why.*
    - The ascent objective is unbounded. Once the forget loss is pushed up, the shared denoiser degrades for every class. This is the "catastrophic collapse" of gradient ascent reported in LLMs (NPO, Zhang et al. 2024) and the instability in diffusion reported by SalUn (Fan et al. 2024).
    - A retain-loss term or a Fisher anchor bounds the collateral damage (Kirkpatrick et al. 2017; Heng & Soh 2023).
  - *Would falsify.*
    - Gradient ascent holds retain accuracy ≥ 0.90 for more than 400 steps after reaching cat rate ≤ 0.10.
    - Or gradient difference never forgets at either lr.
- **A3-b — MIA stays at chance at the operating point.**
  - *Prediction.* Loss-, SecMI- and LiRA-AUC are within 0.5 ± 0.05 for every selected point.
  - *Why.* The oracle itself is at chance (P1 held in v4: AUC 0.52–0.54). The model is in the generalisation regime (Kadkhodaie et al. 2024), so per-example losses carry almost no membership signal for unlearning to remove.
  - *Would falsify.* Any selected point with AUC > 0.6. That would mean the unlearning update *created* a membership signal: "over-forgetting" visible to the attacker (cf. Hayes et al. 2024, Inexact unlearning needs more careful evaluations).
- **A3-c — matched outputs, unmatched model.**
  - *Prediction.*
    - Even at a matched cat rate, the gradient methods sit far from the oracle: $d(u,O)/d_\text{floor} > 3$ on the forget set.
    - Their relearning is faster than the oracle's: ratio < 0.5, against the oracle's 500–550 steps.
  - *Why.*
    - The oracle "forgets" because its class-3 embedding was never trained (v4 §7.1). Gradient methods instead push the shared weights away from cat features.
    - That is a low-rank, reversible change, so the knowledge stays recoverable. This is the relearning-attack argument (Łucki et al., TMLR; Hu et al. 2025).
  - *Would falsify.* A gradient method at the operating point with relearn ratio ≥ 0.8 and $d/d_\text{floor} \le 3$. That method would be oracle-like on two independent axes.
- **A3-d — the embedding is the locus.**
  - *Prediction.*
    - `embedding_reset` (zero gradient steps) brings the cat rate to ≤ 0.15 with retain accuracy within 0.02 of the original.
    - Its retain-class distance to the original is at the floor, since nothing that retain classes use changes.
  - *Why.*
    - Under classifier-free guidance, the conditional prediction is the only path by which "cat" enters sampling (Ho & Salimans 2022).
    - A random embedding row is an untrained condition, which is what the oracle has. So output-level forgetting should follow from this one row.
  - *Would falsify.* A cat rate > 0.3 after the reset, meaning cat generation is not carried by the embedding row. For example, the network could map an arbitrary condition back to frequent classes.
- **A3-e — but it is the shallowest forgetting.**
  - *Prediction.*
    - `embedding_reset` relearns fastest of all methods: ratio vs oracle < 0.2.
    - Its weight-change spectrum is trivially concentrated (one row).
  - *Why.* Every cat feature in the denoiser is untouched, so relearning only has to recover one 256-d vector.
  - *Implication.* If A3-d and A3-e both hold, we have a method that passes output-level audits as well as the oracle does, yet fails the relearning audit badly. That is the cleanest possible demonstration of the paper's thesis that audits on different rungs disagree.
  - *Caveat.* Its weight-change norm (~16) reflects replacing one row with a fresh random draw, not a large update. It is excluded from any norm-matched comparison.

**Amendment A4 (post-hoc, exploratory; predictions written 2026-11-21 ≈ 16:00 IST, before any computation): where in noise level does the retrain differ?**

- *Motivation.*
  - In v4, the oracle and the original are indistinguishable on every per-example loss statistic: SNR ≈ 1 on all four query sets, cats included.
  - Yet they differ completely in conditional generation: cat rate ≈ 0.05 for the oracle vs ≈ 0.9 for the original.
  - The loss-based rungs (1, 2, 5) therefore cannot see the one thing unlearning was supposed to do.
- *Experiment.*
  - We decompose the closeness signal by timestep and by conditioning, on CPU/MPS from the existing checkpoints, so no Kaggle quota is used.
  - $\varepsilon$-MSE is measured on training cats and on near-class images, with common random numbers, at 12 timesteps spanning 10–990.
  - Each is measured under two conditions: the true label and the null label.
  - Models: the original, oracles s0 and s1, `embedding_reset` (built locally: zero steps), and the v4 gradient-ascent and label-annealing models.

**Predictions.**
- **A4-a — locality.**
  - *Prediction.* On cats with the cat label, the per-timestep signal-to-floor ratio $d_t(\theta_0,O)/d_t(O_0,O_1)$ is < 1.2 for all $t \le 400$, and exceeds 1.2 only at high noise ($t \ge 600$), if anywhere.
  - *Why.*
    - At low noise, the optimal denoiser is local: the posterior mean is fixed by a small pixel neighbourhood, shared across classes (Kadkhodaie et al. 2024; Kamb & Ganguli 2024; Niedoba et al. 2024).
    - Class identity is a global property that only matters when the noise has erased it.
    - This is also why guidance helps mainly in a middle-to-high noise interval (Kynkäänniemi et al. 2024).
- **A4-b — the label is where the original and the oracle differ.**
  - *Definition.* The label benefit is $B_m(t) = \mathbb{E}_x[\ell_m(x,\varnothing,t) - \ell_m(x,\text{cat},t)]$ on training cats.
  - *Prediction.* $B_{\theta_0}(t) > 0$ at high $t$ (the cat label helps the original), while $B_O(t) \approx 0$ or $< 0$ for both oracles (their cat embedding was never trained).
  - *Why.* The oracle's cat row only received its random initialisation. Since the null row is shared, the oracle's whole difference from the original on cats should be carried by the label.
- **A4-c — dilution.**
  - *Prediction.* Even where the signal-to-floor ratio exceeds 1.2, the timestep-averaged distance used by $E$ (5 timesteps, 50–800) has SNR < 1.2. This reproduces v4's undefined $E$.
  - *Why.* A label effect concentrated at a few high-noise timesteps, with the per-example variance of $\varepsilon$-MSE spread over all timesteps, is averaged away.
  - *Consequence.* A loss-based closeness audit designed in the usual way is structurally blind to class unlearning in a generalising model.
- **A4-d — embedding reset reproduces the oracle's label profile.**
  - *Prediction.* $B_{\text{reset}}(t)$ tracks $B_O(t)$, within the oracle–oracle spread, at every $t$. On null-conditioned losses, `embedding_reset` equals the original exactly (by construction).
  - *Contrast.* Gradient ascent instead changes null-conditioned losses on all classes, which is the collateral damage behind its $d/d_\text{floor}\approx 1600$ on retain classes in v4.
- **Falsifiers.**
  - Signal-to-floor ratio > 1.2 at $t \le 200$ would falsify A4-a.
  - $B_O(t) \approx B_{\theta_0}(t)$ would falsify A4-b. That would mean the oracle forgets cats somewhere other than the label pathway, e.g. through retain-class features.

**A4-e (written 2026-11-21 ≈ 17:30 IST, after A4-a…d were scored, before any multi-draw computation).**
- *Experiment.* 20 independent `embedding_reset` draws (seeds 100–119) on the original network. For each draw, measure:
  - timestep-standardised $E_\text{std}$ on cats;
  - the gated cat rate (A5) over 200 samples.
- *Prediction.*
  - The outcome is draw-dependent and broad: SD of $E_\text{std}$ across draws ≥ 0.3, and gated cat rates spanning more than 0.2 between the 10th and 90th percentile.
  - Fewer than half of the draws land inside the oracle reference band, i.e. $E_\text{std} \ge 0.5$ with gated cat rate ≤ 0.15.
- *Why.*
  - The original network was trained only with its 10 trained rows and the null row.
  - Its response to a foreign vector is set by where that vector falls relative to the trained rows. A random direction in 256-d is nearly orthogonal to all of them, yet the projections vary from draw to draw.
- *Implication if it holds.* A random re-draw is not a reliable unlearning method, but a *selected* re-draw is cheap. It would then be fair to compare it with the oracle only at the distribution level, and that is the paper's point about the target itself being a distribution.

**Operational tests (fixed 2026-11-21 ≈ 15:30 IST, while run v5 was running and before any of its outputs existed; implemented in `src/analysis/a3_report.py`).**
- *How verdicts are assigned.*
  - With 2 seeds a seed-level CI is meaningless, so a prediction **holds** only if it holds in every seed where it can be evaluated.
  - It is **untestable** where the quantity is undefined, e.g. a relearn ratio when the oracle's relearning is undefined, or the truncation forgetting fraction when the full update does not forget.
- **P5.**
  - Definition: forgetting fraction $f_k = (\text{pre} - \text{rate}_k)/(\text{pre} - \text{rate}_\text{full})$.
  - Holds if $f_k \ge 0.8$ for some $k \le 4$, AND Spearman(k, relearn steps) ≥ 0 over $k \in \{1, 4, 16, 64, \text{full}\}$.
- **A3-a.** All of:
  - Gradient ascent does not meet the target at lr 3e-5.
  - For every ascent entry that forgets, first step with retain < 0.90 minus first step with cat rate ≤ 0.10 is ≤ 400 steps.
  - The gradient difference and variational EWC selections have `reached_target = true`.
- **A3-b.** $|AUC − 0.5| \le 0.05$ for loss-MIA, SecMI and LiRA, on every audited A3 record.
- **A3-c.** Every gradient-method A3 record has $d(u,O)/d_\text{floor}$ (forget set) > 3 AND a relearn ratio vs the same-seed oracle < 0.5.
- **A3-d.**
  - Holds if embedding_reset has cat rate ≤ 0.15, AND retain accuracy ≥ the same seed's original − 0.02, AND |mean near-class loss shift vs original| ≤ $d_\text{floor,near}$.
  - The last condition is a proxy for "retain-class distance at the floor".
- **A3-e.** embedding_reset's relearn ratio is < 0.2 and the smallest of all A3 records in that seed.

---

## 4. Design decisions and why

Each item lists the decision, then the reason. The full list with commit references is in the PREREGISTRATION amendments.

**Regimes**
- **Two forget regimes.** `forget_frac 0.10` removes all 5,000 cats (the class regime: the oracle never sees a cat). `0.05` removes 2,500 random cats (the subset regime: the oracle still generates cats). Relearning a class the oracle still generates is undefined, so H1 and H3's relearning parts run in the class regime only. The subset regime tests H2 when the question is "forget these images", not "forget this concept".
- **The forget set is fixed per fraction (`split_seed 0`).** It is independent of the run seed, so oracle seeds are replicates of *one* counterfactual. Without that, the floor would mix two sources of variance.
- **Oracle initialisation uses a different seed stream from the base model's.** Otherwise oracle seed 0 would share the base model's initialisation, and the two oracles would not be exchangeable.

**Fingerprint**
- **Fixed noise and fixed query sets** (query seed 12345, noise per position and timestep) for every checkpoint. Then the differences in $d$ are differences between models, not between Monte-Carlo draws: a common-random-numbers design.

**Attribution**
- **CountSketch instead of a dense Rademacher projection.** The dense projection was 6.5M × 4,096 and was regenerated per batch. CountSketch (Charikar et al. 2002; Weinberger et al. 2009) preserves inner products in expectation with the same Johnson–Lindenstrauss-type guarantee, at O(nnz) cost; a unit test checks the inner-product preservation.
- **Per-example gradients are computed in vmap chunks of 4.** A batch of 16 used about 14 GB, the whole T4. Chunking changes memory, not values.

**Statistics and controls**
- **Censored relearning.** A cell that never reaches 90 % recovery within the step budget is right-censored. Dropping such cells would select on the outcome, so rank statistics tie them at the maximum.
- **The `retain_finetune` control and the oracle probe reference** are explained in §2.2 and §2.4.

**Training**
- **The pre-registered 40k training steps were kept** even after Colab was found to be GPU-bound at 0.30 s/step. Changing the schedule mid-study to save time would have been an undisclosed deviation, and the oracles must match the base model's schedule exactly.

---

## 5. Process log (what actually happened, including failures)

All times are IST. The log is kept even where nothing scientific depends on it, because it explains the provenance of every checkpoint.

- **2026-11-18 — design review before any GPU run.** Found and fixed seven bugs that would have silently corrupted results:
  1. every seed ran as seed 0;
  2. `forget_frac` never reached the oracle's data split;
  3. oracle checkpoints for different fractions shared one file;
  4. MIA config missing;
  5. oracle cells skipped the audit, so H1 was unconfirmable;
  6. the probe saw the true label;
  7. the NPO reference was wrong, and censored relearning cells were dropped.

  The H2 statistic was found to be unable to discriminate methods (§2.5) and was replaced. Amendment written.
- **2026-11-18 — smoke test on Colab T4.** All stages ran (432 s) on a 0.7M-param model. It revealed the unbounded-E problem, which led to the SNR guard, and a 0/0 relearn ratio that had counted as a pass. CIFAR was switched to the Hugging Face parquet mirror because the Toronto server delivered about 50 kB/s.
- **2026-11-18 23:35 — 2026-11-19 04:30 — Colab run.** Classifier at 88.6 %. Base DDPM finished (3.32 h). Oracle seed 0 reached step 16,000. Colab then reclaimed the free session, and every later GPU request was refused ("Service Unavailable"), consistent with the free tier's usage cap.
  - A recovery script ("keeper") was meant to detect the loss and rebuild the session. It had a bug: when the session was gone, the ssh error text was read as a healthy response, so the loss went unnoticed for about 30 minutes.
  - An earlier sync bug had compared files by size only, and a re-saved checkpoint has the same size, so newer checkpoints were never copied. It was fixed to compare modification times before any data were lost.
- **2026-11-19 09:00 — moved to Kaggle (2× T4, private kernel `arushb/ua-runner`).**
  - One pipeline process per GPU (seed 0 and seed 1).
  - A barrier stage blocks each sweep until *both* oracles exist, because every audit needs the floor.
  - An 11 h wall-clock deadline makes training save and exit cleanly before Kaggle's 12 h kill.
  - State moves between runs as a private dataset (`arushb/ua-state`).
  - A 15-min harness test verified resume, both GPUs, the deadline and output persistence.
- **2026-11-19 09:08 — 12:41 — run v2.** Both class-regime oracles were completed. The first sweep cell then crashed with CUDA out-of-memory in TRAK per-example gradients; the smoke model was too small to expose it. The driver treated Kaggle's "COMPLETE" status as success and began relaunching; this was caught and stopped by hand.
  - **Fixes:** chunked vmap with an OOM fallback; attribution made non-fatal (it is a secondary readout, and a failure there should not discard the primary axes); per-axis timings; the runner writes its exit codes and the driver stops on any crash.
  - **Lesson for the paper's reproducibility section:** smoke tests must run at the real parameter count for every memory-heavy stage, not just at a toy scale.
- **2026-11-19 12:45 — run v3 — crashed after 12 min with the same error.** The kernel had mounted the previous dataset version, which still had the old code, even though the dataset was reported "ready" when the run was pushed. The driver now proves the served code is the published code: it downloads the file back and compares SHA-256, and the runner refuses mismatched code. **Lesson:** on hosted platforms, verify artefact identity end to end; a status flag is not proof.
- **2026-11-19 13:07 — run v4 launched** with verified code `bab2d7ed64cc`: the class-regime sweep, rank truncation, then the subset-regime oracles and sweep. → class sweep done (§7); truncation crashed and re-ran in v5 (§7.10).

**Threats to validity introduced by the process.**
- Seed 0's oracle was resumed twice. Resumes restore the optimiser, scaler and RNG state, but the data loader is re-seeded, so it is a different but equally valid draw of training randomness.
- The base model (Colab) and the oracles (Kaggle, torch 2.11) ran on different software stacks.

Neither affects distributional claims. Both are logged in `EXPERIMENT_LOG.md`.

---
- 2026-11-19 15:20 — compute-budget incident. `kaggle quota` showed 2.58 of 30 weekly GPU-h left, so run v4 (pushed with an 11 h budget) will hit the quota at about 17:50 IST, long before its own deadline. Measured charge rate: about 1 GPU-h per wall-hour on T4x2. A session killed by quota exhaustion may save no output, so v4's results are at risk. Fix (driver v2): before each push the driver reads the remaining quota, waits for the weekly refresh if less than 3 h remain, and bakes a budget of min(11 h, remaining − 0.5 h) into the runner so every run ends with its own clean deadline inside the quota. Lesson: on quota-metered compute, the time budget must come from the meter, not from the platform's session cap.
- 2026-11-19 18:00 — non-GPU work while the quota resets (next run Sat 21 Nov ≈ 05:45 IST). Ported $E$ to the LLM audit. Fixed two LLM protocol bugs, the seed-dependent held-out split and the GradDiff retain leak (see §6). Changed the analysis to report each track separately and to keep the do-nothing control check for every (track, regime, seed) cell; before, only the last one read was kept.
- 2026-11-19 19:38 — run v4 completed the class sweep: 14/14 cells audited, about 52 min per cell, 40 min of it TRAK. Kaggle let the session run about 1.5 h past the exhausted quota. Both processes then crashed in `truncate_c`: `cell_name` depended on dict insertion order, so the truncation stage looked in `methodgradient_ascent_forget_frac0p1_seed0/` while the sweep had written `forget_frac0p1_methodgradient_ascent_seed0/`. Fixed by sorting keys in `cell_name`; the sorted order equals the sweep's order, so existing directories are unchanged. Results are in §7.
- 2026-11-20 — A3 pre-flight. A local CPU smoke on a 60-step toy model exercised every new stage end to end: truncation with the fixed cell names, trajectories, selection, the selected-point audits including `embedding_reset`, and the pipeline continuing past a deliberately failing stage while still exiting non-zero. All passed.
- 2026-11-21 12:46 — **run v5 launched** with verified code `07a0d9e15cd8` and an 11 h budget of a fresh 30 GPU-h quota. The driver should have launched at about 05:45. It did not, because its `sleep` does not advance while macOS is asleep. It was restarted by hand; nothing was lost. Lesson: on a laptop, long waits must be checked against wall-clock time, not a single timer. Housekeeping: the smoke had appended three throwaway entries to `EXPERIMENT_LOG.md` (relative log path), and they were removed. Predictions A3-a … A3-e were written down (§3) before any A3 data. → results in §7.10.
- 2026-11-21 15:30 — **analysis bug caught before it could matter.** `load_records` globbed every `results/*/*/audit_record.json`. The A3 records (`results/D6_oppoint_audit/`) reuse method names (`gradient_ascent`, …) at other learning rates and step counts. So the end-of-run `analyze` would have pooled exploratory cells into the pre-registered H1–H3 tests. Fix: records are now tagged with their experiment directory, and `D6_*` experiments are excluded by default. A3 is scored separately by `src/analysis/a3_report.py` against the thresholds fixed in §3. The hypothesis report generated on Kaggle at the end of v5 predates this fix and will be regenerated locally. Lesson: exploratory and confirmatory records must be separated by construction, not by naming convention.
- 2026-11-24 11:50 — **test-runner bug found in the end-of-study audit.** In `tests/test_audit_cores.py` the `__main__` runner sat above the last 7 tests, so they were defined but never run, and the file still exited 0. Those tests cover the closeness statistic, the degeneracy gate, the operating-point selection rule, the LLM closeness suite, and the C2, C3-A and C3-B scoring rules. The runner was moved to the end of the file. All 26 tests now run and pass, so the registered verdicts computed with that code stand. Lesson: a hand-rolled runner should assert the number of tests it ran.

## 6. Experiment inventory and status

### Diffusion track (primary; the current pipeline)

| ID | What | Units | Status |
|---|---|---|---|
| C0 | CIFAR-10 classifier (generation-rate metric) | 1 training | ✅ 88.6 % |
| D1 | Base class-conditional DDPM, 6.48M params, 40k steps | 1 training | ✅ |
| D2-c | Oracle retrains, class regime (no cats), seeds 0, 1 | 2 trainings | ✅ |
| D3–D5-c | Class-regime sweep: 7 methods (oracle, do_nothing, GA, GD, label annealing, variational EWC, retain_finetune) × 2 seeds, each with the full audit (forgetting + utility, 4 MIA variants, E on 4 query sets, attribution, probe, spectrum, alignment, relearning) | 14 cells | ✅ run v4 (§7) |
| H3-int | Rank-truncation intervention on GA cells: k ∈ {1, 4, 16, 64, full} × 2 seeds | 10 variants | ✅ run v5 (§7.10; P5 held but uninformative) |
| D6-traj (A3, exploratory) | Operating-point trajectories: GA × 3 lr, GD × 2 lr, VEWC × 3 ewc_weight, seed 0, eval every 200 steps | 8 runs | ✅ run v5 (§7.10) |
| D6-audit (A3, exploratory) | Audit at the selected operating point per method + `embedding_reset`, seeds 0, 1, TRAK off | ≤ 8 cells | ✅ run v5 (§7.10) |
| D2-s | Oracle retrains, subset regime (2,500 cats removed), seeds 0, 1 | 2 trainings | ✅ run v5 (§7.10) |
| D3–D5-s | Subset-regime sweep, 7 methods × 2 seeds | 14 cells | ✅ run v5, 14/14 (§7.10) |
| A | Hypothesis report and figures | — | ✅ reports per study (§7.10–7.17); paper figures in `paper/figures/` (`python -m src.analysis.paper_figures`) |

That leaves 2 trainings, 28 audited cells and 10 interventions on the diffusion side. On 2× T4 this is about one to one-and-a-half 11 h Kaggle runs, depending on the per-cell audit time measured in run v3.

### LLM track (corroborating; code written and smoke-tested, not yet run at scale)

| ID | What | Units |
|---|---|---|
| L1 | Qwen2.5-0.5B fine-tune on the TOFU-style set (LoRA r16; plus one full-fine-tune arm for H3) | 2 trainings |
| L2 | Oracle retrains on retain only (LoRA + full-FT) | 2–4 trainings |
| L3 | Unlearning: GA, GradDiff, NPO, oracle, do_nothing × 3 seeds (LoRA; minimum version 4 methods × 2 seeds) + 3 methods × 2 seeds full-FT | 15 + 6 cells (8 + 6 minimum) |
| L4 | Audit: eval, NLL / Min-K% / LiRA / CMIA MIA, attribution, probe, relearning (ROUGE-L), spectrum | per cell |

**Ported 2026-11-19 (no GPU).** $E$ now runs on per-example mean answer-token NLL, a single loss column in place of the diffusion per-timestep losses. The same `closeness_np` core is used, with the same SNR guard and bootstrap. The LLM query sets are:
- *forget*: TOFU forget QA;
- *heldout*: the held-out retain slice, never trained on;
- *near*: retain-train QA about other fictitious authors with the same template, which every model has seen;
- *far*: `world_facts`, which comes from pre-training.

*far* is not a retained fine-tuning set, because TOFU has none. It therefore measures collateral damage to general ability rather than downstream fingerprint, and the SNR guard should usually leave $E_\text{far}$ undefined. This is expected, not a failure. The sweep config now lists two oracle seeds (`oracle_checkpoints`), so the LLM noise floor is estimated as in the diffusion track. The LLM smoke test asserts that $E$ is present with 2 reference oracles; it has not yet been run on a GPU.

**Protocol bugs found while porting, fixed before any LLM data existed.**
1. The held-out (MIA non-member) slice was drawn with a seed derived from the *training/cell* seed. The base model is trained only at seed 0, so the "non-members" of a seed-1 or seed-2 cell were mostly base-model training examples. That would have biased loss-MIA AUC towards 0.5, exactly the direction of prediction P1, so it would have been a confound for our own hypothesis. The two oracle seeds would also have been trained on different retain sets, which inflates the seed-to-seed floor $d_\text{floor}$ with a data-set difference that is not seed noise. The split now uses one fixed seed everywhere (`HELDOUT_SPLIT_SEED`).
2. GradDiff's retain term sampled from the whole retain split, including the held-out slice. That trains on the MIA non-members and makes the MIA look like forgetting. The retain term now uses only the retain-train slice.

Both fixes change the protocol, not a hypothesis or threshold. They are logged here because they were found by reading code, not by looking at data.

### Pre-registered secondary analyses and stretch

- Unconditional-sample drift of the forget class; per-layer localisation of $\|\Delta_\ell\|$; Hessian-damping sensitivity λ ∈ {1e-3, 1e-2, 1e-1}; LoRA-subspace vs full-parameter gradients.
- S1: f-INE seed null with more oracle seeds (3–5 oracles, FDR 0.1).
- S2: Stable Diffusion 1.5 / SD-Turbo concept erasure.
- Ablations: retain availability ("data constraints", Panda et al. 2026) and forget size.

**Count.** Counted as distinct runs, the repo proposes:
- 9 core experiment blocks (C0, D1–D5, L1–L4), of which C0, D1 and D2-c are done;
- 2 stretch blocks;
- about 5 secondary analyses or ablations.

In units of work, about 28 diffusion cells, 10 interventions, 2 oracle trainings, and about 20–30 LLM cells with 4–6 LLM trainings remain.

---

## 7. Results — class regime (run v4, 2026-11-19; 7 methods × 2 seeds; truncation and subset regime: see §7.10)

Source: `results/D3_ddpm_unlearn/*/audit_record.json` (14 records), Kaggle run v4, code `bab2d7ed64cc`.
- **Two values per cell** are seeds 0 / 1.
- **Generation rates** come from 500 samples per class.
- **MIA** uses 1,000 members and 1,000 non-members.
- **Distances** are computed on 256 queries × 5 timesteps.

### 7.0 The table

| Method | Cat gen. rate (pre 0.83 / 0.82) | Retain gen. acc. | Loss-MIA | LiRA | Probe | Relearn steps (ratio to oracle) | $d(u,O)/d_\text{floor}$ forget · near · far | $\|\Delta\|_F$ | Top-5 energy | Alignment $a$ |
|---|---|---|---|---|---|---|---|---|---|---|
| do_nothing | 0.83 / 0.82 | 0.95 / 0.96 | 0.516 | 0.539 | 0.73 / 0.76 | 0 | 1.00 · 0.99 · 0.98 | 0 | — | — |
| oracle | 0.01 / 0.10 | 0.96 / 0.95 | 0.51 | 0.48 / 0.53 | 0.73 / 0.77 | 550 / 500 | (reference) | 153 | 0.24 | −0.001 |
| gradient_ascent | 0.01 / 0.03 | **0.11 / 0.11** | 0.503 | **0.66** | 0.70 / 0.74 | 100 / 50 (0.18 / 0.10) | **175 · 1746 · 1592** | 3.5 | **0.93** | **0.043** |
| gradient_difference | 0.65 / 0.73 | 0.94 / 0.93 | 0.500 | 0.63 / 0.56 | 0.75 | 100 / 50 (0.18 / 0.10) | **150** · 1.7 · 1.4 | 4.2 | 0.72 | 0.046 |
| label_annealing | 0.83 / 0.78 | 0.96 / 0.97 | 0.518 | 0.54 | 0.75 / 0.77 | 0 (never forgot) | 1.1 · 2.0 · 1.6 | 2.9 | 0.25 | 0.002 |
| variational_ewc | 0.00 / 0.01 | **0.10 / 0.12** | 0.503 | **0.66** | 0.71 / 0.73 | 100 / 100 (0.18 / 0.20) | **168 · 1666 · 1518** | 3.8 | **0.94** | **0.035** |
| retain_finetune | 0.92 / 0.87 | 0.94 / 0.96 | 0.516 | 0.53 | 0.76 / 0.77 | 0 | 1.1 · 1.9 · 1.6 | 2.5 | 0.22 | 0.003 |

### 7.1 The do-nothing control — and the central finding: retraining without the class is invisible to every loss-based statistic

**P1 held.** The loss-MIA AUC of the untouched original is **0.516**; SecMI gives 0.519, LiRA 0.539 and CMIA 0.514. The pre-registered expected-to-fail control therefore **fails on its MIA check, as predicted**. A 6.5M-parameter DDPM trained for about 100 epochs on CIFAR-10 is in the generalisation regime of Kadkhodaie et al. (2024): its denoising loss on a training cat is barely different from its loss on a test cat. H1's "MIA ≤ 0.55" criterion is therefore met by every model, including the untouched one, and carries no evidence. This is a statement about the evaluation, and we do not tune the model to memorise more.

The control's second failed check, probe accuracy ≥ 0.8, is **a mis-calibrated harness threshold, not forgetting**. It was set as an absolute number before any data existed. The original model's cat-vs-rest probe reaches only 0.73 / 0.76, and the oracle's reaches 0.73 / 0.77. The pre-registered H1 criterion is *relative* ("≥ 80 % of do-nothing"), which is the right one. The absolute check in `control_assertions` will be replaced by the relative one, a harness fix logged as such.

**The stronger observation is the noise floor itself.** Every query set has $d(\theta_0,O)/d_\text{floor} \approx 1$:

| Set | forget | heldout | near | far |
|---|---|---|---|---|
| SNR | 1.00 | 0.96 | 0.99 | 0.98 |

So the original model is exactly as close to a cat-free retrain, per example, as one cat-free retrain is to another.

**Removing all 5,000 cats** raises the mean forget-set loss by only +0.0056: 0.0377 for the original, 0.0408 and 0.0458 for the two oracles. That is smaller than the per-example seed-to-seed RMS floor of 0.0076, and the two oracles differ from each other in mean by 0.005. On the near and far classes the mean shift is +0.0000 and −0.0001.

The signed-fingerprint check agrees. The changes of the two oracles relative to the original correlate at 0.52 across near queries, and **0.5 is the value expected from pure seed noise**: both changes contain the original's own idiosyncrasy, so $\mathrm{corr}(O_0-\theta_0, O_1-\theta_0) = 1/2$ for exchangeable independent draws. There is no near-class imprint beyond that.

**Interpretation.** A generalising DDPM denoises cats almost as well without ever having seen one, because the score it learns is built from geometry shared across natural images (Kadkhodaie et al. 2024). Yet the oracle *cannot generate* cats on label 3 (rates of 0.01 and 0.10), because its class-3 embedding was never trained. **In this regime, "forgetting a class" lives almost entirely in the conditional routing, not in the denoiser's knowledge of cat images.** Exact unlearning here looks like "an untrained class embedding". It does not look like "a denoiser that is bad at cats".

### 7.2 H1 — suppression, not erasure

**Gradient ascent and variational EWC "forget" by destroying the model.** The cat rate falls to 0.00–0.03, but generation accuracy on the *other nine classes* collapses to 0.10–0.12, which is chance. Their forget-set loss is 1.30–1.37, against 0.04 for the oracle. On the distance scale they sit **165–175× the noise floor away from the retrain on cats, and about 1,600× on the retain classes**. They are not on the way to the oracle; they are far beyond it in another direction.

**The utility-preserving methods barely forget at the matched budget.**
- *Gradient difference* keeps retain accuracy (0.93–0.94) and moves the cat rate only from 0.83 to 0.65–0.73. Even so, its forget loss overshoots (150× the floor), while its near-class distance stays at 1.7×.
- *Label annealing* (our reading of Panda et al. 2025) did not forget at all (0.78–0.83).

At 2,000 steps and learning rate 3e-5, **no method reaches the oracle's operating point** (cat rate ≈ 0.05, retain accuracy ≈ 0.95). The methods sit at the two ends of a forgetting–utility trade-off.

**Recoverability, rung 4, is the axis that separates methods from the retrain.**
- From the oracle it takes 500–550 steps on 5 % of the cats to reach 90 % of the original cat rate.
- From gradient ascent and variational EWC, which start from a rate of 0.00–0.03, it takes **50–100 steps, a ratio of 0.10–0.20**.

That meets H1's relearning criterion (≤ 1/5) in both seeds, the second seed of variational EWC exactly at the bound. Gradient difference's equally short relearning is not evidence, because it starts close to the 90 % target. The first-order account of §2.3 predicted this. A short ascent run leaves a displacement that the forget gradient points straight back along. A retrain has to train the class pathway from scratch.

**Formal verdict: H1 is NOT supported as pre-registered.** It requires the unlearned model's probe to exceed the oracle's, and the probe does not discriminate at all: original 0.73 / 0.76, oracle 0.73 / 0.77, gradient ascent 0.70 / 0.74. This is the risk §2.2 anticipated. A cat-free DDPM separates cats linearly as well as the original does, so a probe cannot tell suppression from erasure here. The relearning half of H1 holds; the representation half is untestable with a probe.

### 7.3 H2 — the closeness ladder

**H2 is untestable as designed, and its premise failed.** The SNR guard (1.2) left $E$ undefined on every set and every cell, as it should: there is no trace of the forget set for a method to reproduce. The ±600 values the smoke test showed without the guard would have been reported as findings otherwise.
- **P3:** untestable, because there is no near-class imprint to reproduce (§7.1).
- **P4:** not supported. Near and far are equally at the floor (0.99 vs 0.98).
- **P2**, read through distances, failed. Gradient ascent's distance is the same for training cats and test cats (175 vs 175, 166 vs 166), so it acts on the class, not on the specific images.

  The calibrated attack disagrees in part. LiRA reaches 0.66 on gradient ascent and variational EWC, against 0.54 on the original. Over-unlearning makes the forget examples *more* identifiable than doing nothing, the "Streisand effect" noted by Hayes et al. (2024) for inexact unlearning. The per-example deviations from the oracle are therefore not exactly the same for members and non-members.

**What the distances do show** is a clean ordinal ladder in units of the retrain noise floor:
- label annealing and retain fine-tune: 1–2× everywhere, effectively unmoved;
- gradient difference: 150× on cats, 1.7× elsewhere;
- gradient ascent and variational EWC: 170× on cats, about 1,600× elsewhere.

$d(u,O)/d_\text{floor}$ is well defined even when $E$ is not, and it is the statistic the data support. We will report it as a post-hoc, clearly labelled readout. It is not a pre-registered test.

**Influence estimates.** The planned validity check failed for TRAK, as §2.5 allowed. The correlation between TRAK's predicted near-query loss change and the oracle's actual change is −0.03 / −0.05, i.e. zero. TRAK *does* correlate with gradient ascent's changes, 0.17 / 0.19, and with variational EWC's, 0.12 / 0.15. This is the expected asymmetry. TRAK is a linearised, first-order estimate of the effect of a gradient step on the forget set, which is literally what ascent methods take. Retraining is not a small step, and Bae et al. (2022) show influence functions approximate a proximal response, not leave-one-out retraining. **Influence estimates describe gradient-based unlearning, not exact unlearning.** TRAK also cost 40 of each cell's 52 minutes.

### 7.4 H3 — spectra and alignment (truncation: §7.10)

**The spectral criteria hold for the ascent-type methods.**

| Method | Top-5 energy | Effective rank |
|---|---|---|
| gradient ascent | 0.93 | 43 |
| variational EWC | 0.94 | 41 |
| gradient difference | 0.72 | 70 |
| oracle | 0.24 | 89 |
| retain fine-tune (matched steps) | 0.22 | 106 |
| label annealing | 0.25 | 103 |

The pre-registered thresholds are "> 0.7 for approximate methods, < 0.3 for the oracle, and ≥ 0.2 above retain fine-tune". Gradient ascent and variational EWC pass all three. Gradient difference passes at the margin. Label annealing fails, and its update looks like plain fine-tuning, consistent with it not having forgotten.

**Alignment.** $a = \cos(g_f(\theta_u), \Delta)$ is 0.035–0.049 for the three ascent-type methods. It is ±0.003 for the controls and −0.001 for the oracle. In absolute terms it is small, far from the idealised $a \to 1$ of §2.3. But a random direction in 6.5M dimensions has $|\cos| \approx 1/\sqrt{6.5\text{M}} \approx 4\times10^{-4}$, so this is about 100× chance and the ordering is as predicted.

**What is still missing.** The rank-versus-relearning correlation at matched norm cannot be estimated on these cells: the methods that forgot (rank low, fast relearning) are exactly those that broke the model, and the low-rank methods that did not forget have relearning time 0. Method identity and outcome are confounded, which is the situation the truncation intervention (P5) was designed for. It did not run in v4, because a cell-naming bug made it look in the wrong directory (§5). It is fixed and runs first in the next GPU session.

### 7.5 What we now believe

1. **At this scale, loss-based membership and closeness statistics cannot audit class unlearning.** This holds even for the exact counterfactual: retraining without the entire class is at the seed-noise floor on every loss-based statistic. Any audit that certifies forgetting from such statistics in a generalising diffusion model is certifying nothing. This is the strongest, most general result of the run.
2. **The difference between erasure and suppression shows up in recoverability, not in losses.** Relearning is 5–10× faster from ascent-type methods than from a retrain, and those updates are low-rank and aligned with the forget gradient.
3. **Matched-budget comparisons put methods at the two ends of a trade-off.** Ascent destroys and annealing does nothing. Comparing methods at a common operating point (matched forgetting or matched utility) is needed before rung-by-rung claims about *methods* are meaningful. This is a design gap we did not anticipate.
4. **Class unlearning in a generalising DDPM is a conditioning problem.** The oracle differs from the original in the class pathway, not in the denoiser. This motivates an embedding- and cross-attention-level audit, and suggests that the ascent methods' destruction is avoidable by construction.

### 7.6 Prediction scorecard (class regime)

| Prediction | Outcome |
|---|---|
| P1 MIA control may fail (AUC 0.52–0.62) | **Held** (0.516) |
| P2 $E_\text{forget} > E_\text{heldout}$ for ascent | **Failed / untestable** ($E$ undefined; distances equal for training and test cats; LiRA partly disagrees) |
| P3 $E_\text{near} < 0.5$ | **Untestable** (no imprint exists above the floor) |
| P4 near signal > far signal | **Failed** (both at the floor) |
| P5 truncation | **Held by the rule, uninformative** (§7.10: every rank breaks the model at matched norm) |
| Per-method: GA overshoots | **Held** (175× the floor, retain collapse) |
| Per-method: label annealing partial $E$ | **Failed** (no forgetting at the matched budget) |
| Per-method: variational EWC between GA and oracle | **Failed** (indistinguishable from GA; Fisher anchor too weak at weight 1.0) |
| Per-method: retain fine-tune ≈ 0 | **Held** |

---

### 7.7 Amendment A4 results — where the retrain differs (exploratory; computed 2026-11-21 on Apple MPS from the v4 checkpoints, 0 GPU-h)

**Reproducibility check.** The local loss pipeline reproduces the Kaggle audit's $d(u,O)$ and $d_\text{floor}$ to 3 decimals for all 14 class-regime cells, e.g. gradient ascent s0: 1.3306 vs 1.3299; floor 0.0076 vs 0.0076. Same query seed, different hardware and torch build.

**Per-timestep signal-to-floor ratio** $d_t(\theta_0,O)/d_t(O_0,O_1)$ on 256 training cats, cat label, $n=256$, common random numbers:

| t | 10 | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 990 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cats, cat label | 0.86 | 1.10 | 0.72 | 0.86 | 1.24 | 1.83 | 2.50 | 2.17 | 2.36 | 2.27 | 2.56 | 1.58 |
| near classes, own label | 1.10 | 1.09 | 0.98 | 1.02 | 1.03 | 1.04 | 0.80 | 0.75 | 0.82 | 0.91 | 0.95 | 0.82 |

The audit's distance (RMS over examples and the 5 audit timesteps) gives SNR **0.99** on cats. That is v4's undefined $E$, reproduced. The **timestep-standardised** distance, in which each timestep's differences are divided by its own oracle-pair floor (`standardized_closeness_np`), gives SNR **1.75** on cats (1.82 on the dense grid) and **0.98** on the near classes.

**Why the plain distance is blind.**
- Absolute ε-MSE differences between *any* two models are large at low noise and tiny at high noise. The oracle-pair floor is 0.011 at $t=50$ vs 0.0004 at $t=800$.
- So timesteps 50 and 200 carry **≈ 99 %** of the plain floor's squared mass ($1.25\text{e-}4 + 1.59\text{e-}4$ out of $2.88\text{e-}4$).
- The low-noise regime is exactly where models trained with and without cats agree. Denoising there is local and class-agnostic (Kadkhodaie et al. 2024; Kamb & Ganguli 2024; Niedoba et al. 2024).
- The retrain's trace is at $t \ge 300$. There it is 1.8–2.6× the floor but contributes almost nothing in absolute terms.
- This is the ε-parameterisation's implicit weighting. Its loss is a weighted ELBO whose weight on a timestep shrinks as noise grows, relative to the $x_0$ error (Kingma et al. 2021; Kingma & Gao 2023). An audit built on unweighted ε-MSE therefore inherits a blind spot over the noise levels where global, class-level structure is decided.

**Label benefit** $B_m(t)=\mathbb{E}_x[\ell_m(x,\varnothing,t)-\ell_m(x,\text{cat},t)]$ on cats (positive: the label helps):

| model | t=10 | 50 | 100 | 200 | 400 | 600 | 800 | 990 |
|---|---|---|---|---|---|---|---|---|
| original | +.0008 | +.0005 | +.0003 | +.0003 | +.0003 | +.0001 | .0000 | .0000 |
| oracle s0 | −.0024 | −.0046 | −.0032 | −.0036 | −.0021 | −.0011 | −.0008 | −.0008 |
| oracle s1 | −.0163 | −.0156 | −.0242 | −.0159 | −.0035 | −.0014 | −.0011 | −.0013 |
| embedding_reset s0 | −.0465 | −.0153 | −.0029 | −.0001 | .0000 | +.0001 | +.0001 | +.0001 |

**Embedding norms.**
- Weight decay is 0. The oracles' never-trained cat row keeps its initial norm: 16.3 and 15.8, vs 15.7–16.0 for their trained rows, and 16.0 expected for N(0,1) in 256-d.
- So the oracle's cat label is a random vector of the right scale.
- `embedding_reset` places a random vector of the same scale into the *original* network.

**Timestep-standardised closeness $E_\text{std}$** (cats, audit timesteps, bootstrap 95 % CI; v4 checkpoints):

| method | s0 | s1 |
|---|---|---|
| do_nothing | 0.00 | 0.00 |
| retain_finetune | −0.61 [−0.66, −0.57] | −0.62 |
| label_annealing | −0.63 | −0.65 |
| gradient_difference | −1657 | −1633 |
| variational_ewc | −2055 | −1972 |
| gradient_ascent | −2088 | −1976 |
| embedding_reset | **0.05** [0.03, 0.08] | **1.27** [1.24, 1.31] |
| *oracle (is one of the two references)* | *1.66, by construction* | *1.66* |

Near classes: SNR 0.98. $E_\text{std}$ is undefined, because retraining without cats leaves **no measurable trace on the other animal classes** at this scale, even per timestep.

**Scoring (thresholds fixed in §3 before computation).**
- **A4-a — failed as stated, direction held.**
  - The ratio exceeds 1.2 already at $t=300$ (1.24) and $t=400$ (1.83), so "< 1.2 for all $t \le 400$" is false.
  - The falsifier (ratio > 1.2 at $t \le 200$) did not occur: values there are 0.72–1.10.
  - The trace is confined to the upper ~70 % of the noise range; the onset is earlier than predicted.
- **A4-b — half held.**
  - Held: $B_O(t)<0$ at every $t$ for both oracles. The oracle's cat label actively *hurts* denoising of cats.
  - Failed: "$B_{\theta_0}>0$ at high $t$". The original's label benefit is ≈ 0 at high noise and only +0.0003–0.0008 at low noise.
  - Revised reading: the cat label carries almost no ε-MSE information even in the original. What distinguishes the oracle is a *harmful* foreign conditioning vector, not a missing helpful one.
- **A4-c — held.** Audit-distance SNR is 0.99, while the per-timestep ratio peaks at 2.56 and the standardised SNR is 1.75.
- **A4-d — failed.**
  - `embedding_reset`'s label profile matches the *original*'s, not the oracle's, at $t \ge 200$ (B ≈ 0 vs −0.001 to −0.016). It departs only at $t \le 100$, more strongly than the oracles.
  - Its standardised closeness depends on the random draw: E 0.05 for draw s0, 1.27 for draw s1.
  - So the same 256-d replacement can leave the model at "do nothing" or put it inside the oracle cloud. The network's response to a never-seen conditioning vector is draw-dependent.

**What this changes.**
1. **A methodological contribution.** Loss-based unlearning audits of diffusion models must weight timesteps in noise-floor units. The unweighted ε-MSE distance used in the MIA and closeness literature is dominated by the class-agnostic low-noise regime and is blind to class removal. Standardising recovers a signal of 1.75× the floor from the same losses at no extra cost.
2. **A conceptual point about the target.**
   - For a conditional model, what exact retraining does on the removed label is decided by an *untrained random vector*.
   - The two oracles already disagree: cat rates 0.01 vs 0.10, label benefit differing 5× at low $t$.
   - So "indistinguishable from retrain" on the forgotten label is a *distribution* over embedding draws, not a point. A zero-step embedding re-draw samples from a closely related distribution. Whether it lands inside the oracle cloud is a matter of the draw, which A4-e (below) quantifies.
3. **No imprint on near classes at this scale.** H2's headline query set has nothing to measure: SNR 0.98 even standardised. H2's premise that retraining without cats leaves an imprint on dogs and horses fails empirically in the generalisation regime. That is a finding about the regime, not about the methods.

### 7.8 A4-e results — 20 independent embedding-reset draws (exploratory; MPS, 0 GPU-h; `results/figures/a4e_reset_draws.json`)

**Setup.**
- Original network; the cat row was re-drawn from N(0, 1) with seeds 100–119.
- $E_\text{std}$ on 256 training cats at the audit timesteps.
- Gated cat rate (A5) from 200 samples with sampling seed 4242.
- References at the same sampling seed: original 0.81, oracle s0 0.015, oracle s1 0.095.

**Results.**
- **Behaviour is easy.**
  - Gated cat rate: median 0.095, 10th–90th percentile 0.03–0.18. 14/20 draws are ≤ 0.15, i.e. inside the oracles' range.
  - So a zero-step re-draw of one 256-d vector removes cat generation about as well as retraining, in most draws.
- **Loss-level closeness is not.**
  - $E_\text{std}$: mean 0.04, SD 1.15, median 0.25, range −3.53 to 1.27.
  - Only **3/20** draws are in the oracle band ($E_\text{std} \ge 0.5$ AND cat ≤ 0.15).
- **Behaviour and loss-closeness dissociate in both directions.** corr(E, cat) = 0.43.
  - Draw 116 is loss-close ($E_\text{std}$ 1.27) but still generates cats (0.59).
  - Draw 112 makes 88.5 % of cat samples degenerate. The raw classifier rate is 0.17; the gated rate is 0.015; $E_\text{std} = -3.53$ correctly flags it as far from any retrain.
  - The draw's cosine with the null row does not predict anything (r = −0.08).

**Scoring (thresholds from §3).**

| Prediction | Verdict | Numbers |
|---|---|---|
| SD of $E_\text{std}$ ≥ 0.3 | **held** | 1.15 |
| Gated cat rate spans > 0.2 between the 10th and 90th percentile | **failed** | 0.15; behaviour is *less* draw-dependent than predicted |
| Fewer than half of draws in the oracle band | **held** | 3/20 |

**Reading.**
- For a conditional model, matching the retrain *behaviourally* on the forgotten label is almost free.
- Matching it at the level of the denoiser's losses, measured per noise level against the seed floor, is not. A behavioural audit cannot tell the two apart.
- This strengthens F1's practical point: behaviour-based checks can be passed trivially, so the closeness measurement is what discriminates.
- It also shows the two axes measure different things. Draw 116 is loss-close but behaviourally not forgotten. A complete audit needs both.

### 7.9 A6 — predictions for the subset regime's per-timestep trace (written 2026-11-21 ≈ 21:50 IST, before computing; exploratory)

**Context.**
- In the subset regime, 2,500 of the 5,000 cats are removed, so the oracle still sees 2,500 cats and knows the class.
- v5's do-nothing cells give plain SNR 1.30 on the forget set (vs 0.99 in the class regime), and LiRA 0.615.
- So the subset oracle differs from the original in a way the plain average *can* see.

**Hypothesis.**
- This is a *per-example membership* trace. Prior MIA work locates membership at low-to-mid noise: Carlini et al. 2023, best $t \approx 50$–300; SecMI, $t$ 50–150; Matsumoto et al. 2023, peak at $\bar\alpha \approx 0.7$.
- The class regime's trace is at high noise (§7.7).
- So the two kinds of removal should leave traces at opposite ends of the noise range.

**Predictions** (same machinery as `timestep_decomp.controls`; forget set = the 2,500 removed cats, held-out set = test cats):

| ID | Prediction |
|---|---|
| A6-a | On the removed cats, the per-timestep ratio reaches its maximum at some $t \le 400$, and is ≤ 1.2 at every $t \ge 700$ |
| A6-b | On **test** cats (non-members of both models) the ratio is ≤ 1.2 at every $t$. Removing examples leaves no class-level trace, the mirror image of C1-4 |
| A6-c | On the *retained* cats (members of both), the ratio is ≤ 1.2 at every $t$ |

**Falsifier.** A ratio > 1.2 on test cats at high noise would mean the subset oracle carries a class-level trace too. Then the "two ends of the noise range" story is wrong.

### 7.10 Run v5 results (2026-11-21; code `07a0d9e15cd8`; 7.38 GPU-h; all stages exit 0)

**P5 — rank truncation, scored "held" by the pre-registered rule, but uninformative.**
- At matched total norm (3.5 / 3.6), *every* truncation of the gradient-ascent update destroys the model, rank 1 included.
- Retain accuracy is 0.10–0.12 at every k ∈ {1, 4, 16, 64, full}. Relearning is flat at 50–150 steps (Spearman(k, relearn) = 0.4 for s0 and 0.0 for s1).
- "Rank-1 reproduces the forgetting" here means "a rank-1 update of this norm is enough to break the network".
- The informative reading goes *against* H3's mechanism: at matched norm, rank does not matter; the size of the update does. This is consistent with Siddiqui et al. 2025's norm-based account of relearning.
- Caveat: these rates predate A5. Since every variant has retain ≈ 0.11, the gate would not change the verdict.

**A3 — operating points (raw rates; A5 gate applied where checkpoints exist).**
- *Trajectories.*
  - Gradient ascent collapses retain accuracy to 0.11–0.14 within the first 200 steps at every lr, 3e-6 included.
  - Variational EWC collapses identically at every anchor weight, 1e2–1e4.
  - Gradient difference at lr 1e-4 never forgets (cat 0.47 → 1.00).
  - Gradient difference at lr 3e-4 reaches the target at step 600 (cat 0.02, retain 0.94), but its trajectory is unstable: retain 0.24 at step 800, then 0.87–0.96.
- *Selected:* only gradient difference (lr 3e-4, 600 steps). Ascent and variational EWC have **no utility-preserving operating point** on this grid.
- *Audit of the selection.*
  - **Seed 0** forgets for real. A5-gated cat rate 0.056 with 0 % degenerate samples; retain 0.958.
  - **Seed 1 does not forget** (cat 0.794, retain 0.842). The operating point chosen on one seed's trajectory does not transfer to a second seed with a different data order. Gradient difference's operating point is not reproducible.
  - Even seed 0, which forgets in behaviour, sits **167× the floor** from the oracle on cat ε-MSE, and relearns *slower* than the oracle (700 vs ≈ 550 steps; ratio 1.27).
- *Scoring.*

| Prediction | Verdict | Reason |
|---|---|---|
| A3-a | **failed** | variational EWC never reached the target |
| A3-b | **failed** | LiRA 0.574–0.578 on gradient difference, just outside ±0.05 |
| A3-c | **failed** | seed 0's relearn ratio is 1.27; seed 1 does not forget at all |
| A3-d | **failed** | embedding_reset s1 cat rate 0.174 > 0.15. Seed 0 met every condition (0.128, retain 0.949, near shift 0) |
|  | *gate check* | A local re-sample (MPS, different sampler noise) gives **0.116** for the same s1 checkpoint. So the failure is within sampling noise of the threshold: n = 500, SE ≈ 0.017, gap ≈ 3 SE. The verdict stays "failed" on the registered measurement, but it is not robust. All four A3 checkpoints have **0 %** degenerate samples, so their raw rates are valid. |
| A3-e | **failed** | embedding_reset relearn ratio 0.46 / 0.80, not < 0.2 |

- *What A3-e's failure suggests.* This is a hypothesis, not tested.
  - Re-learning a single 256-d vector of norm ≈ 16 with Adam at lr 1e-4 is slow by construction: each step moves a coordinate by at most about 1e-4.
  - So the relearning time may measure the *optimiser's geometry* (where the knowledge sits in parameter space) as much as how deeply the knowledge was removed.
  - Relearning-based audits should be read with this confound in mind.

**Subset regime (pre-registered cells, 14/14).**
- Do-nothing: plain SNR on the forget set is 1.30, so $E$ **is defined** here (it was 0.99 in the class regime). LiRA is 0.615, so membership is detectable even for the unmodified model.
- Gradient ascent and gradient difference push LiRA to 0.95–0.98. Unlearning *creates* a membership signal, the over-unlearning effect Hayes et al. 2024 warn about, and sit 1,260–1,740× the floor.
- Label annealing: $E_\text{forget} = -5.7$, $d/d_\text{floor} = 3$.
- Full H1–H3 tables are in `results/figures/hypothesis_report.json`. They are regenerated locally and include only pre-registered records.

**A6 — where removing *examples* leaves its trace** (predictions §7.9; $n = 256$ per set; MPS).

| t | 10 | 50 | 100 | 200 | 400 | 600 | 800 | 900 | 990 | plain | std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| removed cats | **1.47** | **1.32** | 1.16 | 1.12 | 0.90 | 0.75 | 0.92 | 1.25 | 1.24 | **1.27** | 1.03 |
| retained cats | 1.13 | 1.15 | 1.07 | 1.02 | 0.84 | 0.73 | 0.94 | 1.26 | 1.26 | 1.11 | 0.96 |
| test cats | 1.08 | 1.13 | 1.02 | 1.14 | 0.94 | 0.79 | 0.93 | 1.29 | 1.26 | 1.12 | 1.00 |

- **A6-a — failed as stated.**
  - Held part: the removed set's maximum is at $t=10$, and its excess over retained and test cats is confined to $t \le 100$. That is the membership trace, at the *lowest* noise.
  - Failed part: the ratio is also 1.24–1.25 at $t \ge 900$.
- **A6-b and A6-c — failed.** Test and retained cats also reach 1.26–1.29 at $t = 900, 990$.
- *Reading.*
  - The very-high-noise excess (≈ 1.25) appears *equally* on removed, retained and test cats. So it is a weak **class-level** trace of halving the class's data, not a membership trace. It is small next to the class regime's 2.2–2.6.
  - So removal leaves two traces at the two ends of the noise range:
    - **membership** (which images were removed) at $t \le 100$;
    - **class-level** (how much of the class remains) at high $t$.
- *Consequence for audit design.* The plain average (dominated by low noise) *sees* membership (1.27) but is blind to class removal (0.99, §7.7). The standardised average sees class removal (1.75) but *dilutes* membership (1.03). Neither single weighting fits both kinds of unlearning. **Audits should report the per-timestep profile**, or at least one low-noise and one high-noise statistic.
- This refines F1 from "standardise" to "**where you look must match what was removed**". The C1 headline (class removal is high-noise only) is unaffected.

### 7.11 A5 results — gated re-score of all class-regime checkpoints (MPS; same sampler seeds and sizes as the audit; `results/figures/a5_behaviour_rescore.json`)

| model | cat raw | cat valid | cat degenerate | retain raw | retain valid | retain degenerate |
|---|---|---|---|---|---|---|
| do_nothing s0 / s1 | 0.87 / 0.83 | 0.87 / 0.83 | 0 / 0 | 0.97 / 0.97 | 0.97 / 0.97 | 0 / 0 |
| oracle s0 / s1 | 0.006 / 0.058 | same | 0 / 0 | 0.96 / 0.97 | same | 0 / 0 |
| retain_finetune | 0.92 / 0.91 | same | 0 / 0 | 0.96 / 0.96 | same | 0 / 0 |
| label_annealing | 0.84 / 0.82 | same | 0 / 0 | 0.96 / 0.98 | same | 0 / 0 |
| **gradient_difference s0** | **0.654** | **0.000** | **1.00** | 0.94 | 0.94 | 0 |
| gradient_difference s1 | 0.756 | 0.756 | 0 | 0.95 | 0.95 | 0 |
| gradient_ascent | 0.006 / 0.058 | **0 / 0** | **1.00 / 1.00** | 0.12 / 0.10 | **0 / 0** | **1.00 / 1.00** |
| variational_ewc | 0.000 / 0.002 | 0 / 0 | 1.00 / 1.00 | 0.11 / 0.11 | 0 / 0 | 1.00 / 1.00 |
| embedding_reset s0 / s1 | 0.120 / 0.116 | same | 0 / 0 | 0.97 / 0.97 | same | 0 / 0 |

**Reading.**
- The local raw rates reproduce the Kaggle audit to within sampler noise; the differences are ≤ 0.04 except oracle s1, 0.058 vs 0.100.
- **Gradient difference seed 0 is a different kind of model from seed 1, and the raw metric hides it.**
  - Seed 0 fully replaced cat generation with noise, *selectively*: other classes are untouched, retain 0.94.
  - The raw rate scores it as "barely forgot" (0.65); the gated rate says it removed cat generation entirely (0.00 valid cats).
  - Seed 1 really still generates cats.
  - So v4's statement "gradient difference barely forgets" was wrong for seed 0 and right for seed 1. The two seeds of the same method at the same budget land in qualitatively different states. This matches A3's finding that gradient difference's operating point does not transfer across seeds.
- **Gradient ascent and variational EWC are dead networks.**
  - 100 % of samples are degenerate for every class; valid retain accuracy is 0.00 rather than the raw 0.10–0.12.
  - The raw 0.10–0.12 is chance-level labelling of noise, not degraded generation.
- **Earlier wording corrected.**
  - Previous text (and the A5 amendment) said gradient difference's samples are noise "62–73 % labelled cat". That generalised from seed 0's sample grid; 73 % was seed 1's *real* cats.
  - Corrected in `PREREGISTRATION.md`, `src/diffusion/sample.py` and §9.

### 7.12 C1 results — the confirmatory test of F1 (registered 2026-11-21; scored 2026-11-22 with the registered code, unchanged; `results/figures/c1_report.json`)

**Data.**
- Four new exact retrains, all trained the full 40k steps on Kaggle T4 (3.1–3.3 h each): CIFAR-10 minus automobile, seeds 0, 1, 2; and CIFAR-10 minus cat, seed 2.
- Scored on MPS with $n = 256$ per query set.

**Per-timestep signal / floor, forget-train set, 3 oracles.** Timesteps are 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 990.

| class | snr_t | plain audit SNR | standardised audit SNR |
|---|---|---|---|
| automobile | 0.70 1.29 **1.70** 1.23 1.28 1.26 1.17 1.05 1.00 1.07 1.14 1.05 | 1.22 | 1.18 |
| cat | 1.21 0.78 0.90 1.10 1.47 1.95 **2.58** 2.34 2.58 2.51 2.22 1.84 | 0.95 | 1.83 |

**Controls (standardised SNR).**
- Automobile: test 1.18, near (airplane, ship, truck) 1.04, far 1.02.
- Cat: test 1.83, near 0.93, far 0.88.

**Leave-one-oracle-out calibration.** Each entry is the held-out retrain scored against the other two. An exchangeable retrain should score E ≈ 1.

| class | held-out s0 | held-out s1 | held-out s2 |
|---|---|---|---|
| automobile, plain E | undefined | 2.13 | −0.95 |
| automobile, standardised E | −1.07 | undefined | −0.60 |
| cat, plain E | undefined | undefined | undefined |
| cat, standardised E | 0.92 | **−0.18** | 1.18 |

**Verdicts**, applying the rules exactly as registered:
- Automobile: C1-1 failed, C1-2 failed (plain 1.22 ≥ 1.2), C1-3 failed, C1-4 failed, C1-5 held, C1-6 failed.
- Cat: C1-1 to C1-5 held, C1-6 failed.

**By the registered interpretation rule, F1 is class-specific, not general.**
- Automobile removal leaves no high-noise trace.
- Even for cats, the standardised statistic is not calibrated: one of three held-out exact retrains scores below the original.
- The cat replication (C1-1 to C1-5 with a new third oracle) shows the cat *pattern* is stable. What fails is the claim that the pattern is a property of class removal in general, and that the standardised E can recognise an exact retrain.

### 7.13 Why C1 failed: the exact retrain's embedding for the removed label is untrained (Amendment A6; post-hoc; 0 GPU-h; `src/analysis/label_diag.py`, `results/figures/label_diag.json`)

**Observation 1, by construction and verified.**
- A retrain on CIFAR-10 minus class $k$ never presents label $k$. Row $k$ of its class embedding therefore receives zero gradient. With Adam and weight decay 0 the update $m/(\sqrt v + \epsilon)$ is exactly 0, so the row keeps its random $\mathcal N(0, I)$ initial value.
- Re-initialising each retrain from its seed's init stream gives these movements from init:

  | | removed row $k$ | every trained row |
  |---|---|---|
  | model weights | 2–3e-6 (≈ 1e-7 relative to norm ≈ 16; float32 round-off between torch builds) | 0.96–0.99 |
  | EMA | 5–8e-4 (EMA arithmetic drift) | — |

- All 6 retrains (classes 1 and 3) behave this way.
- A side fact: trained rows move only about 6 % of their norm. A trained embedding is mostly its random init plus a small learned offset, and the network learns to *read* the random vectors it was given.

**Consequence.**
- A query $\varepsilon_\theta(x_t, t, y = k)$ to a retrain evaluates the network on a conditioning vector it was never trained to read: a different random vector in every seed.
- The "exact-retrain reference" for label-$k$ queries is therefore not a counterfactual. It is a seed-specific random-conditioning response. This predicts a heterogeneous seed floor and unstable calibration, and that is what C1 shows.

**Observation 2: removed-label vs null-label queries, same images and checkpoints.**
- The null label (index 10) is trained in every model through CFG dropout, so every model's unconditional pathway is a well-defined function of its training data.

| | max/min oracle-pair distance, $t \le 400$ | train plain SNR | test plain SNR | near / far | held-out retrain plain E |
|---|---|---|---|---|---|
| automobile, label $k$ | **2.1–16.1** | 1.22 | 1.20 | 1.08 / 1.06 | undefined, 2.19, −0.97 |
| automobile, **null** | **1.1–1.2** | **3.74** | **3.47** | 1.14 / 1.07 | **1.00 [0.97, 1.03], 1.01 [0.98, 1.04], 0.99 [0.96, 1.02]** |
| cat, label $k$ | 1.7–4.4 | 0.95 | 0.93 | 1.00 / 1.02 | (std) 0.92, −0.12, 1.20 |
| cat, null | 1.1–1.2 | 1.21 | 1.15 | 1.05 / 1.05 | undefined in 2 of 3 (SNR < 1.2) |

- Automobile null-label snr_t is 1.82, 3.35, 4.68, **5.45**, 4.42, 3.45, 1.72, 1.10, 0.93, 0.90, 0.80, 0.74. The trace is at low and mid noise, peaking at $t = 200$.
- Retrain-minus-original mean loss is positive (+0.0033 to +0.0049) at $t \le 200$ and ≈ 0 at $t \ge 600$. The retrains denoise automobiles worse, as expected for a class missing from their data.

**Theoretical reading.**
- *Why the removed-label response is arbitrary and noise-dependent.*
  - The class pathway enters every residual block through the embedding.
  - A never-trained random vector has no learned meaning. Its effect is its projection on readout directions the network learned for the other nine rows, and those directions differ per seed.
  - Conditioning matters most where the class is still undecided, at high noise (Sclocchi, Favero & Wyart 2025; Biroli et al. 2024). That is where the cat original (trained row) and the cat retrains (random rows) separated consistently.
  - For automobile, one retrain's random row produced a grossly different response even at low noise: s2's loss at $t = 10$ is 0.456 vs 0.32–0.36, which inflated the floor. Which pattern appears is a property of the draws, not of class removal.
- *Why the null-label trace is the data effect.*
  - The unconditional denoiser of a retrain models $p(x \mid y \ne k)$.
  - At low and mid noise a denoiser relies on local image statistics shared across classes (Kadkhodaie et al. 2024; Kamb & Ganguli 2024). It is worse on class-$k$ images to the extent that the retained classes do not cover those statistics.
  - At high noise the unconditional prediction is close to a dataset-level mean and barely depends on one class.
  - This predicts a low/mid-noise trace whose size depends on how distinctive the removed class is. Automobile is strong (3.7). Cat is weak (1.2), plausibly covered by the other animals, dog above all. This magnitude story is a conjecture from two classes; C2 tests it on ships at risk (C2-0).
- *An open observation.* At $t \ge 700$, snr_t < 1 (0.6–0.8) on every query set and under both labels. The original is closer to each retrain than the retrains are to each other. This is not label-specific and does not affect the claims. A candidate explanation is the original sitting nearer the "average" high-noise solution; it is not tested.

**Relation to earlier results.**
- **A4-e** (embedding-reset closeness draw-dependent, SD 1.15, 3/20 draws in band) is the same mechanism seen from the method side. Resetting row $k$ to a random draw makes the original's label-$k$ pathway look like *a* retrain's, but not like *these* retrains', whose own random rows are different draws.
- **F1** (§7.7) was a real, reproducible pattern for cats. It was mis-attributed to class removal; it is a property of trained-vs-random conditioning on that class.

**What this is.** A measurement-validity result for unlearning audits of class-conditional diffusion models:
- Retrain-referenced closeness under the removed label is ill-posed.
- The unconditional pathway gives a well-posed, calibrated reference.
- The claim was found post-hoc on two classes; confirmatory study C2 (ship, 3 new retrains) is registered and running.

**Caveats.**
- One architecture: learned `nn.Embedding` conditioning with $\mathcal N(0, 1)$ init, weight decay 0.
- With weight decay the untrained row would decay deterministically towards 0. That would be seed-independent in direction but still untrained.
- CIFAR-10, a 6.5M-parameter U-Net, 40k steps.

### 7.14 C2 results — confirmatory test of F4′ on ship (registered 2026-11-22 before data; scored 2026-11-23 with the registered code, unchanged; `results/figures/label_diag.json`, key `class8`; 6.51 GPU-h)

**Verdicts.**
- Claim (a), "the removed-label retrain reference is ill-posed": **confirmed**. C2-F and C2-G held.
- Claim (b), "the null label is a well-posed reference": **not confirmed as registered**. C2-0, C2-A, C2-C, C2-D and C2-E held; C2-B failed at a single timestep.

**Cross-class table** (forget-train, n = 256, three retrains each; classes 1 and 3 post-hoc, class 8 confirmatory):

| | automobile (1) | cat (3) | **ship (8)** |
|---|---|---|---|
| removed-label row moved (rel.) | ≈ 1e-7 | ≈ 1e-7 | ≈ 1e-7 |
| forget-label max pair ratio, t ≤ 400 | 16.1 | 4.4 | 3.23 |
| forget-label held-out std E | −1.14, NaN, −0.63 | 0.92, −0.12, 1.20 | 1.81, 1.14, −1.92 |
| null-label max pair ratio, t ≤ 400 | 1.23 | ≤ 1.2 | **1.42** (t = 400; ≤ 1.24 at t ≤ 300) |
| null-label plain SNR (train / test) | 3.74 / 3.47 | 1.21 / — | 1.85 / 1.54 |
| null-label held-out plain E | 1.00, 1.01, 0.99 | NaN, NaN, 0.80 | 0.94, 1.13, 0.94 |
| null-label held-out std E | 0.88, 1.05, 1.02 | NaN | 0.58, 1.33, 0.57 |
| null-label near / far SNR | 1.14 / 1.07 | — | 1.08 / 1.03 |
| null-label snr_t peak | 5.45 at t = 200 | — | 2.06 at t = 200 |

**Reading.**
1. **The flaw replicates on a class chosen in advance.**
   - Ship's row is untrained in every retrain.
   - Under the removed label, the retrains disagree by up to 3.2× at low/mid noise, and by up to 16× at t = 900.
   - No held-out retrain calibrates. Two of the three standardised E values fall outside [0.6, 1.4], with opposite signs.
2. **The null-label audit calibrates on a second class.**
   - Held-out plain E values are 0.94, 1.13 and 0.94, with all CIs overlapping 1 or close to it (s1's CI is [1.04, 1.22]).
   - The trace is class-level: it reappears on unseen test ships at 83 % of the train SNR. It is absent on other classes.
   - It peaks at t = 200, the same timestep as for automobile.
3. **The registered homogeneity prediction failed narrowly.** At t = 400 the ratio is 1.42 against a 1.3 threshold. We keep the failure as written.
   - Substantively, calibration held despite it, so homogeneity at every t ≤ 400 is not necessary for calibration here.
   - The null-label floor is clearly more homogeneous than the forget-label floor at the same t (1.42 vs 2.13 at t = 400; 1.04 vs 3.23 at t = 200).
   - The paper should state "more homogeneous", not "homogeneous".
4. **Exploratory: standardisation does not help the null-label audit.** On ship, standardised E (0.58, 1.33, 0.57) is miscalibrated while plain E is calibrated.
   - Per-timestep standardisation reweights toward timesteps with a small floor. Those are high-noise t, where the null trace is absent (snr_t < 1 at t ≥ 600) and floor estimates from three retrains are noisy.
   - The recommended statistic is therefore plain closeness on the null label.
5. **Trace magnitude is class-dependent:** automobile 3.74, ship 1.85, cat 1.21.
   - This is consistent with the distinctiveness conjecture (§7.13), but three classes cannot test it.

**What the paper can claim.**
- (i) Confirmed on a pre-registered fresh class: the removed-label retrain reference is ill-posed, by mechanism (untrained row) and by consequence (heterogeneous floor, uncalibrated closeness).
- (ii) Confirmed on two classes, one of them confirmatory: null-label plain closeness is calibrated against held-out retrains, class-level, and specific.
- (iii) Reported as failed: the strict homogeneity prediction.
- Cat shows the method's limit: when the trace is weak, the audit cannot resolve it.

### 7.15 C3 results — consequence for audits (C3-A) and a fourth, random class (C3-B) (registered 2026-11-23 ≈ 14:35 IST, before data; scored 2026-11-24 with the registered code, unchanged; 8.48 GPU-h for v8 + v9, scoring local)

**C3-B (truck, class 9, drawn by `default_rng(20261123)`): claim (b) confirmed.**
- Removed row untrained: 1.8–3.2e-6.
- Removed-label reference ill-posed: pair ratio 6.6 at t ≤ 400; held-out std E 0.56, −0.95, 0.86.
- Null label: SNR 2.44 train / 2.20 test; near / far 1.01 / 0.91; held-out plain E 1.08 [1.03, 1.13], 0.97 [0.92, 1.02], 0.96 [0.91, 1.01].
- C3-B-B held only narrowly: the null pair ratio is 1.49 at t = 10, against 1.5. The ratio is ≤ 1.19 at t = 50–400, always below the removed-label ratio.
- The original C2-B (≤ 1.3) fails again. That makes 2 of 2 confirmatory classes on which the strict homogeneity rule fails while calibration holds. Our reading: *calibration does not need strict homogeneity; it needs the floor to be far tighter than the removed-label floor.*

**C3-A (ship, automobile, truck): all 12 registered predictions held.**

| | ship | automobile | truck |
|---|---|---|---|
| re-draws passing the removed-label audit | 6/20 | 9/20 | 12/20 |
| SD of re-draw removed E | 4.77 | 0.72 | 3.03 |
| best method null-E lower CI | −0.85 | −0.29 | 0.16 |
| LA null E | −1.87, −1.87 | −0.79, −0.81 | −1.33, −1.34 |

**Reading.**
1. **The retrain-referenced removed-label audit is satisfiable by a zero-step edit**, in 30–60 % of random draws, on every class. This is the paper's strongest practical claim. It moves the result from "a reference is ill-posed" to "the standard audit certifies an intervention that removes nothing".
2. **Mechanism.** The re-draw is another sample from the same distribution the retrain's removed row was drawn from (§7.13). The audit therefore cannot tell them apart. The pass rate measures how often a random row lands "as close as" the retrains' random rows.
3. **Limit of the counter-claim.** The null-label score of 0 for re-draws holds by construction. Whether the class is *recoverable* from a re-drawn model is tested separately (C3-K, §7.16; exploratory).
4. **Methods (exploratory).** At the sweep hyper-parameters, GA and VEWC collapse; GD collapses on the removed label; LA and RF barely forget. No method reaches retrain-like behaviour without collapse, so no method ranking is claimed.
5. **Automobile** has the smallest re-draw spread (SD 0.72) and uses plain E, per the amendment. The pass count is still 9/20, so C3-A1 does not depend on the statistic.

**Compute accounting, this week (quota 30 GPU-h, refresh 2026-11-28):** v5 7.38, v6 ≈ 6.65, v7 6.51, v8 6.84, v9 1.64. That totals 29.03 h, confirmed by `kaggle quota`, with 0.97 h left.

### 7.16 C3-K — the class is one row away in a re-drawn model (EXPLORATORY; predictions in EXPERIMENT_LOG before running; `src/analysis/c3_knowledge.py`; `results/figures/c3_knowledge_c8.json`; 34 min local)
**Method.** Re-fit only row 8 (256 numbers) on 1,000 ship images, network frozen; then measure the gated ship rate.

**Results**
- Passing re-draws: 0.000, 0.005, 0.000 → **0.985, 0.980, 0.980**. The original's rate is 0.995.
- Retrains: 0.005, 0.005, 0.000 → 0.180, 0.125, 0.190.
- Unconditional ship rate: original 0.083, retrains 0.013–0.019.
- **K1, K2 and K3 held.**

**Reading**
- It closes the "by construction" gap of C3-A. The re-draws the removed-label audit certified still contain ship completely.
- The null-label audit's 0 is the right answer. It sees the unconditional pathway, which still makes ships at 5× the retrain rate.
- A secondary point: one row steers a ship-naive retrain to 12–19 % ships. Embedding-only optimisation can elicit unseen classes partially, much as textual inversion does. Relearning attacks that touch only the label row are therefore not a clean test either. What matters is the *gap* between the two (98 % vs ≤ 19 %).

**Extension to automobile and truck (2026-11-24, predictions logged first): the same pattern.**
- Re-draws: 0.985, 0.985, 0.990 (automobile) and 0.960, 0.975, 0.955 (truck).
- Retrains: 0.085–0.190 and 0.155–0.245.
- Unconditional rate: original 0.030 vs 0.003–0.007, and 0.029 vs 0.002–0.006.
- K1–K3 held on 3/3 classes.

**Limits.** 3 re-draws per class; one budget (a larger re-fit budget might lift the retrains); exploratory.

### 7.17 End-of-study audit of the write-up (2026-11-24; no new runs; post-hoc analyses are from existing JSON)

**Process.** Two independent agents read the write-up, neither given our conclusions. One checked every citation against arXiv and the full texts. The other checked every number in the workshop paper against the result JSONs and wrote a critical review (score 5/10, "weak accept if the overclaims are fixed").

**Errors found and corrected.** Each was verified by recomputation before it was changed.
- *Unconditional-rate ratio.* "3.4–14×" should read **4.3–14.5×** (a transposition). Corrected in DRAFT, the log (dated note) and the paper.
- *Truck null CIs.* "every truck null CI contains 1" is wrong: 7 of 9 null CIs contain 1 across classes. The registered C2-A rule uses the point estimate in [0.8, 1.2], so it still holds.
- *Removed-label calibration.* "A held-out retrain is not recognised" overstated the result. Under standardised E, 4/12 rotations fall in [0.6, 1.4]. The supported claim is that every class has at least one failing rotation.
- *Wrong theory link.* The null trace peaking at t = 100–200 was attributed to class commitment at high noise, which is backwards. The citation link was removed.
- *Cat.* Cat was called "no measurable trace", but its pooled null SNR is 1.21, just above the 1.2 definedness threshold. The correct statement is that it is below the registered power threshold of 1.5.
- *Re-draw wording.* "256 numbers no data shaped": the re-draw overwrites a trained row. "Which draws pass is luck": outcomes correlate across classes (ship–truck ρ = 0.52), because the same vectors are reused.
- *Pre-registration status.* The null-label E = 0 for re-draws was presented next to "pre-registered" without saying it holds by construction.
- *Fig. 1 caption* contradicted the plot. *Fig. 5* called t ≤ 400 the "audit range"; the audit also uses t = 600 and 800, where the null floor is also heterogeneous.

**Post-hoc analyses added (labelled as such).**
- Held-out exact retrains pass the C3-A rule in **3/9** leave-one-out rotations, against 27/60 re-draws. The removed-label audit cannot distinguish re-draws from retrains in either direction. This is a stronger and more honest framing than "re-draws pass".
- Wilson CIs for the pass rates: [0.15, 0.52], [0.26, 0.66], [0.39, 0.78].
- Re-draw medians: 0.40, 1.30, 0.88.
- Pass decomposition (E band / behaviour gate): 10/10, 10/17, 12/18.
- Retain-only fine-tuning already scores null E −0.24 to −0.74, so null E penalises generic drift. The null audit is shown to be specific, not sensitive.

**Weaknesses that remain (stated in the paper's limitations).**
- No positive control for the null-label audit.
- No causal isolation of the row's share of the heterogeneity.
- Rate-only audits have a homogeneous reference, and are passed by suppression for a different, known reason. The paper's scope is therefore narrowed to loss- and output-closeness audits.

**Cheapest strengthening (optional).** A *common-row control*, forward passes only, on the existing retrains and runnable on the laptop: give all three retrains the same removed-label row, then recompute the pair ratios and LOO E. If the heterogeneity collapses, the mechanism is causally established. Before running it, it should be registered as a short amendment.

### 7.18 Generality of the mechanism in public training code (checked 2026-11-24 from the `main` branches on GitHub)
- **DiT** (facebookresearch/DiT):
  - `models.py`: `LabelEmbedder` uses `nn.Embedding(num_classes + use_cfg_embedding, hidden_size)`, with a classifier-free-guidance null row.
  - `train.py`: `torch.optim.AdamW(..., lr=1e-4, weight_decay=0)`.
  - Same structure as our model: an absent class's row is untrained, and a trained null row exists.
- **ADM** (openai/guided-diffusion):
  - `unet.py`: `self.label_emb = nn.Embedding(num_classes, time_embed_dim)`.
  - `train_util.py` and `scripts/image_train.py`: AdamW with default `weight_decay=0.0`.
  - The mechanism applies. There is no built-in null row, so our null-label alternative would need CFG training.
- **EDM** (NVlabs/edm):
  - `networks.py`: labels enter as a one-hot vector through `map_label = Linear(label_dim, noise_channels)`, with `label_dropout` zeroing the one-hot.
  - `train.py`: `torch.optim.Adam`, no weight decay.
  - An absent class's column receives exactly zero gradient, because its one-hot input is always 0. Same mechanism; the "null" input is the zero vector, which is trained.

So the untrained-row mechanism holds *by construction* for the standard class-conditional recipes of all three code bases, without weight decay. What it does to their audits is not measured here.


### 7.19 C4 — mechanism test, specificity control and the removed-label Fréchet distance (registered 2026-11-24 12:53 IST, before data; laptop only; scored with the registered code)
**Why.** An independent review of the workshop draft (5/10) asked three things:
- whether the untrained row *causes* the heterogeneity (W7);
- whether null-label E ≈ 1 just means "a different network" (W2);
- whether any published metric is actually affected (W1). FID_O (Zheng, Tai & Yeh 2026) compares the unlearned model's samples with one retrain's "for all the classes", which includes the forgotten label.

**C4-R (row swap; failed, all three rules).**
- Transplanting retrain rows between retrains (a 3×3 grid per class) moves removed-label losses about as much as changing the network does (D_row/D_net 0.69–1.71 at t ≤ 400).
- A common row does not halve the floor's max/min ratio (16.1 → 9.3, 4.45 → 3.98, 3.23 → 4.42, 6.56 → 8.49), nor calibrate it.
- Our earlier explanation, that the independent draws add a second source of variation, is withdrawn.
- What remains is the weaker statement that each network reads an *untrained* input in its own way. Its evidence is the contrast with the trained null label, whose floor is homogeneous; it is not an intervention.
- A direct test would train the removed row in each retrain and re-measure the floor (not done).

**C4-X (cross-class; held).** Retrains that removed a different class score null-label E −0.41 to 0.19 (27/27 < 0.5). The null-label audit responds to the removal of class k, not to "being another network".

**C4-F (removed-label FD; F1 and F2 held, F3 failed).**
- Retrains disagree 5.5–16× more under the removed label than under airplane.
- 28–85 % of re-draws fall within the retrain-pair spread.
- The registered sanity check failed on every class: the unmodified original is no farther from at least one retrain than the farthest retrain pair. The instrument floor (0.11–0.18 under airplane) rules out measurement noise. The reference spread itself swallows the effect of doing nothing.
- For the FID_O argument, this means the forgotten-label component of a single-retrain FD comparison is dominated by which retrain was drawn.
- Limits: 768-d features; we did not compute FID_O itself, over all classes.

**What changed in the write-up.** The mechanism is now stated as "untrained input" in DRAFT and the workshop paper, and the generality (§7.18) and C4 results are in both.

### 7.20 C5 — real FID_O against one retrain (registered 2026-11-25 13:55 IST, before data; laptop only; scored with the registered code)
**Why.** The second review (6/10) found the link to published practice thin: C4-F measured a removed-label FD with 768-d features, not FID_O itself. C5 computes FID_O as Zheng, Tai & Yeh (CVPR 2026) define it (all classes, 2048-d Inception), adding a same-model sampling floor and a retrain-to-retrain floor. Their Table 5 reports retrain FID_O = 0.00.

**Result (n = 1,000 per label; floor 4.48).**
- **The retrain is not a fixed point (O2 held, 3/3).** Two exact retrains differ by FID_O 7.2–12.2 (ship), 16.4–24.1 (automobile) and 8.6–10.0 (truck), i.e. 2.1–4.3× the sampling floor.
- **The forgotten label drives much of it (O4 held, 3/3).** Over the 9 retained labels only, the same pairs give 5.8–8.8. Inflation is 1.49×, 3.02× and 1.28× (truck thin). The retained-only values still exceed the floor, consistent with general training-seed FID variance (Dufour et al. 2026).
- **Re-draws are scored retrain-like on 2 of 3 classes (O1 held, 2/3).** 57 % (ship) and 93 % (automobile) of re-draw–retrain values fall inside the retrain-pair range; on truck only 12 %, because its retrains agree closely.
- **Doing nothing is not reliably separated (O3 failed; held on truck only).** The unmodified original's FID_O to at least one retrain is below the largest retrain pair on ship (8.60 vs 12.18) and automobile (11.61 vs 24.13). This is the C4-F F3 failure at the level of the published metric; on truck it does separate (11.64 vs 9.96).

**Reading.** The single-retrain FID_O protocol has a reference spread of the same size as published method differences (SalUn 3.59–6.93, SPM 1.80–1.97; scale context only, as registered: different model, n and guidance). Whether a reported FID_O gap is meaningful depends on which retrain was drawn, and on two of three classes the spread swallows the effect of doing nothing. Truck shows the effect is class-dependent, not universal.

**Exploratory (not registered).** Against a fixed retrain, 25–75 % of re-draws have lower FID_O than the unmodified original. Re-draws do stop generating class k (C3-A), so this is expected at the output level; what FID_O cannot see is the one-row recovery (C3-K).

**Related work found 2026-11-25 (to cite).** Cho et al. 2025 (arXiv:2510.12981) use retrain-vs-retrain FADE as a floor on Stable Diffusion and find it small (≈ 0.3–1.9). This is consistent with our mechanism: a frozen text encoder means the forgotten prompt is never an untrained input. Dufour, Efros & Pérez 2026 (arXiv:2606.20536) quantify training-seed FID variance (≈ 1.3 % CoV, not unlearning).

## 9. Direction decision (2026-11-21 ≈ 21:30 IST)

**Goal.** One small but genuine finding, confirmed on data collected after the claim was fixed.

**Re-scan of what we have.**

| Candidate | Evidence | Status | Decision |
|---|---|---|---|
| F1: the exact-retrain trace sits only at high noise; unweighted ε-MSE audits are blind; per-timestep floor standardisation recovers it | cats: ratio 1.8–2.6 at $t \ge 300$, ≤ 1.1 at $t \le 200$; plain SNR 0.99 vs standardised 1.75; test cats 1.72 (class-level, not membership); near 0.98, far 0.91 | post-hoc, 1 class, 2 oracles | **headline; confirm as C1** |
| F3: classifier forget rates are corrupted by degenerate outputs (A5) | gradient-difference s0 noise (100 % degenerate) labelled "cat" 65 %; ascent and variational EWC 100 % noise for every class (§7.11) | solid but small; a reviewer may call it obvious | secondary (methods note + gate) |
| F4: the retrain's behaviour on the removed label is a random-embedding effect | two oracles disagree; reset draws E 0.05 vs 1.27 | exploratory; A4-e running | discussion |
| Pre-registered H1–H3 | H1 not supported as written; H2 untestable (no imprint at this scale); H3 pending truncation | confirmatory but mostly null | report honestly; secondary |
| A3 method comparison, subset regime, LLM track | selection compromised by the F3 metric problem; subset and LLM not central | — | appendix / future work |

**Novelty check** (background literature agent, 2026-11-21; not exhaustive).
- Nothing found that states F1, F3 or F4.
- *Adjacent work, which must be cited and positioned against:*
  - Matsumoto et al. 2023 (arXiv:2302.03262): averaging over timesteps dilutes the *per-sample membership* signal, which peaks at intermediate $t$, $\bar\alpha \approx 0.7$. Ours is a *class-removal* signal at higher noise.
  - Carlini et al. 2023 (best $t \approx 50$–300) and SecMI (best $t$ 50–150): membership lives at low-to-mid noise.
  - Park & Park 2025 (arXiv:2510.17917): unlearning is uneven over time. That is measured with gradient and guidance norms against the original, not against a retrain with a seed floor.
  - Joseph & Hoshen 2026 (arXiv:2610.40616): memorisation is localised in time. That is about membership inference; there is no unlearning.
  - Theory that class is decided at high noise: Sclocchi, Favero & Wyart (arXiv:2402.16991, PNAS 2025); Biroli et al. (arXiv:2402.18491, "speciation time"); Kynkäänniemi et al. 2024.
- *Corrections the check surfaced for our own draft.*
  - "Label annealing" (Panda et al.) appears to be a UAI 2025 paper not shown to involve diffusion. Their diffusion paper is "Variational Diffusion Unlearning" (arXiv:2510.04058).
  - f-INE (arXiv:2510.10510) is an influence-estimation test, not an unlearning metric.
  - Both descriptions in `paper/DRAFT.md` must be fixed before submission.
- *Verified locally, against a caveat it raised:* weight decay is 0 and the oracles' unused cat row keeps its initial norm (§7.7). So the "random draw" reading of F4 stands for this model.

**Decision.**
- The paper's contribution becomes F1, a measurement-validity result for unlearning audits of diffusion models. It is confirmed by C1 (`PREREGISTRATION.md`), on a fresh forget class (automobile, 3 new oracles) plus a third cat oracle for a leave-one-oracle-out calibration.
- Cost: 4 oracles × ≈ 3.2 GPU-h, run 2 at a time on T4×2 ≈ 6.4 wall-hours ≈ 6.5 GPU-h of the ≈ 19 h left this week.
- Everything else is reported as secondary with its label.
- The driver was switched to collect-only (20:29), so that run v5's end does not auto-launch the subset regime.

### 9.1 Direction update (2026-11-22 ≈ 11:00 IST, after C1)

- **C1 falsified F1 as a general claim** (§7.12). The diagnosis (§7.13) promotes F4 from "discussion" to the headline. It also gives a constructive fix: audit through the null label.
- **New headline (F4′).**
  - For class unlearning in class-conditional diffusion, the exact-retrain reference is ill-posed under the removed label: that row is an untrained random vector.
  - The unconditional pathway is a calibrated reference, with held-out retrains at E = 1.00 ± 0.02 for automobile.
  - The size of the data trace is class-dependent.
- **Confirmation.**
  - C2 is registered before data: ship, 3 new retrains, kernel v7, about 6.5 GPU-h, leaving about 9 h this week.
  - If C2-0 fails (ship's null-label trace is too weak, as for cat), calibration rests on automobile alone. The paper then states that, and frames trace magnitude as class-dependent.
- **The paper reports C1 as a pre-registered negative result**, with the cat pattern and its re-attribution.
  - That is itself a useful lesson: a clean, replicated single-class pattern (cat: 2 then 3 oracles, test-set replication, null controls) was still not a general effect. Only a fresh-class confirmatory test exposed it.

### 9.2 Novelty check for F4′ (background literature agent, 2026-11-22; arXiv abstracts and web search only, full texts not read)

**Verdict.**
- No paper found states that the retrain's removed-label input embedding is untrained, or that removed-label retrain comparisons are therefore ill-posed.
- Auditing class removal through the null-label pathway also appears new.
- The *surrounding* critique is known: retrain is a fragile, seed-dependent target. So F4′ is positioned as a specific mechanism and fix inside that critique.

**Must cite and position against.**
- Output-distribution reweighting for class unlearning. Ebrahimpour-Boroojeny, Wang & Sundaram, arXiv:2506.20893. *[Corrected 2026-11-24 after a full-text check: the TREW target approximates the retrain's forget-class distribution by tilting the **original** model's predictions by inter-class similarity. Retrains appear only as CMIA shadow models; there is no averaging over retrains.]*
- Head-bias gradient for an absent class in classifiers. Zheng, Chen, Guo & Xiao, arXiv:2605.08730.
- Untrained embedding rows stay at init (LLM tokenizers; the mechanism, outside unlearning). Land & Bartolo 2024, "Fishing for Magikarp", arXiv:2405.05417.
- Retrain-referenced membership inference is miscalibrated. Hayes et al. 2024, arXiv:2403.01218.
- Seed distributions of retrains. Triantafillou et al. 2024, arXiv:2406.09073; Lanyon et al. 2025, arXiv:2510.26714.
- Logit metrics vs representation similarity to retrain. arXiv:2503.06991.
- Instances of the practice in diffusion class forgetting:
  - SalUn, arXiv:2310.12508
  - Selective Amnesia, arXiv:2305.10120
  - EraseDiff, arXiv:2401.05779
  - Score Forgetting Distillation, arXiv:2409.11219
  - Zheng, Tai & Yeh, arXiv:2603.22870

**Scope caveats it raised, and our status.**
1. *The claim is input-side only.*
   - In a softmax classifier the absent class's output row still gets gradient (its logit is pushed down). It is biased, not random.
   - F4′ is stated for input-side conditioning embeddings.
   - It does not apply to text-to-image models with frozen text encoders either.
2. *"Exactly at init" depends on the optimiser.*
   - Under AdamW with weight decay > 0, the row would shrink deterministically toward 0.
   - Ours is AdamW with `weight_decay: 0.0` (`configs/diffusion_base.yaml`). The row is at init to float32 round-off (§7.13). This was checked against the config and the checkpoints.
   - The paper must state the optimiser condition. Under decay, the row is a shrunken copy of its init: still untrained, and still seed-specific in direction.

**Limits of the search.** A footnote or an OpenReview discussion in one of the diffusion papers could exist and would not surface in an abstract-level search.

## 8. References

*Citation re-check 2026-11-24 (arXiv API and full texts of the diffusion-unlearning papers). It found no wrong IDs, authors or years. Description corrections were applied to `paper/DRAFT.md`: Selective Amnesia and EraseDiff report no retrain baseline; Hayes, Lanyon and Land & Bartolo were reworded. Added: Chen et al. SFD (ICLR 2025); Kim, Cha & Kim 2026, "Are We Truly Forgetting?", arXiv:2503.06991; Ho, Jain & Abbeel 2020; Krizhevsky 2009.*

All arXiv IDs and metadata verified against arXiv and venue pages on 2026-11-19. Exceptions: the four entries without an arXiv ID were checked from prior knowledge only, and Hayes et al. has no confirmed peer-reviewed venue.

- Aghajanyan, Zettlemoyer, Gupta (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv:2012.13255
- Alain, Bengio (2016). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644
- Bae, Ng, Lo, Ghassemi, Grosse (2022). If influence functions are the answer, then what is the question? NeurIPS. arXiv:2209.05364
- Basu, Pope, Feizi (2021). Influence functions in deep learning are fragile. ICLR. arXiv:2006.14651
- Belinkov (2022). Probing classifiers: promises, shortcomings, and advances. Computational Linguistics. arXiv:2102.12452
- Biroli, Bonnaire, de Bortoli, Mézard (2024). Dynamical regimes of diffusion models. Nature Communications 15, 9957. arXiv:2402.18491
- Bourtoule et al. (2021). Machine unlearning. IEEE S&P. arXiv:1912.03817
- Cao, Yang (2015). Towards making systems forget with machine unlearning. IEEE S&P.
- Carlini et al. (2022). Membership inference attacks from first principles (LiRA). IEEE S&P. arXiv:2112.03570
- Carlini et al. (2023). Extracting training data from diffusion models. USENIX Security. arXiv:2301.13188
- Charikar, Chen, Farach-Colton (2002). Finding frequent items in data streams (CountSketch). ICALP.
- Deeb, Roger (2024). Do unlearning methods remove information from language model weights? arXiv:2410.08827
- Duan et al. (2023). Are diffusion models vulnerable to membership inference attacks? (SecMI). ICML. arXiv:2302.01316
- Fan et al. (2024). SalUn: empowering machine unlearning via gradient-based weight saliency in both image classification and generation. ICLR. arXiv:2310.12508
- Sclocchi, Favero, Wyart (2025). A phase transition in diffusion models reveals the hierarchical nature of data. PNAS 122(1), e2408799121. arXiv:2402.16991
- Siddiqui, Weller, Krueger, Dziugaite, Mozer, Triantafillou (2025). From Dormant to Deleted: tamper-resistant unlearning through weight-space regularization. NeurIPS. arXiv:2505.22310
- Georgiev, Rinberg, Park, Garg, Ilyas, Madry, Neel (2025). Attribute-to-Delete: machine unlearning via datamodel matching (KLOM). ICLR 2025, published there as "Machine Unlearning via Simulated Oracle Matching". arXiv:2410.23232
- Ginart, Guan, Valiant, Zou (2019). Making AI forget you: data deletion in machine learning. NeurIPS. arXiv:1907.05012
- Golatkar, Achille, Soatto (2020). Eternal sunshine of the spotless net: selective forgetting in deep networks. CVPR. arXiv:1911.04933
- Guo, Goldstein, Hannun, van der Maaten (2020). Certified data removal from machine learning models. ICML. arXiv:1911.03030
- Gur-Ari, Roberts, Dyer (2018). Gradient descent happens in a tiny subspace. arXiv:1812.04754
- Hayes et al. (2024). Inexact unlearning needs more careful evaluations to avoid a false sense of privacy. arXiv:2403.01218
- Heng, Soh (2023). Selective Amnesia: a continual learning approach to forgetting in deep generative models. NeurIPS. arXiv:2305.10120
- Ho, Salimans (2022). Classifier-free diffusion guidance. NeurIPS 2021 Workshop on Deep Generative Models. arXiv:2207.12598 (verified 2026-11-24)
- Ho, Jain, Abbeel (2020). Denoising diffusion probabilistic models. NeurIPS. arXiv:2006.11239
- Hu et al. (2021). LoRA. arXiv:2106.09685
- Hu, Fu, Wu, Smith (2025). Unlearning or obfuscating? Jogging the memory of unlearned LLMs via benign relearning. ICLR. arXiv:2406.13356
- Ilyas et al. (2022). Datamodels. ICML. arXiv:2202.00622
- Jain et al. (2024). Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks. ICLR. arXiv:2311.12786
- Kadkhodaie, Guth, Simoncelli, Mallat (2024). Generalization in diffusion models arises from geometry-adaptive harmonic representations. ICLR. arXiv:2310.02557
- Maini, Feng, Schwarzschild, Lipton, Kolter (2024). TOFU: a task of fictitious unlearning for LLMs. arXiv:2401.06121
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting (EWC). PNAS. arXiv:1612.00796
- Koh, Liang (2017). Understanding black-box predictions via influence functions. ICML. arXiv:1703.04730
- Li, Farkhoor, Liu, Yosinski (2018). Measuring the intrinsic dimension of objective landscapes. ICLR. arXiv:1804.08838
- Łucki, Wei, Huang, Henderson, Tramèr, Rando (2024). An adversarial perspective on machine unlearning for AI safety. TMLR. arXiv:2409.18025
- Panda, Suresh, Guha, Prathosh A.P. (2025). Concept forgetting via label annealing. UAI, PMLR 286:3346–3360. https://proceedings.mlr.press/v286/panda25a.html (no arXiv version found). A classifier paper (pseudo-labels from the model's own predictions, then fine-tuning, on MNIST, CIFAR-10 and CelebA); our null-token variant is our own diffusion adaptation.
- Panda, Varun M S, Jain, Maharana, Prathosh A.P. (2026). Unlearning in diffusion models under data constraints: a variational inference approach. TMLR. arXiv:2510.04058
- Park et al. (2023). TRAK: Attributing model behavior at scale. ICML. arXiv:2303.14186
- Qi et al. (2025). Safety alignment should be made more than just a few tokens deep. ICLR (oral). arXiv:2406.05946
- Roy, Vetterli (2007). The effective rank: a measure of effective dimensionality. EUSIPCO.
- Sekhari, Acharya, Kamath, Suresh (2021). Remember what you want to forget: algorithms for machine unlearning. NeurIPS. arXiv:2103.03279
- Shokri, Stronati, Song, Shmatikov (2017). Membership inference attacks against ML models. IEEE S&P. arXiv:1610.05820
- Thudi, Jia, Shumailov, Papernot (2022). On the necessity of auditable algorithmic definitions for machine unlearning. USENIX Security. arXiv:2110.11891
- Weinberger et al. (2009). Feature hashing for large scale multitask learning. ICML. arXiv:0902.2206
- Yeom, Giacomelli, Fredrikson, Jha (2018). Privacy risk in machine learning: analyzing the connection to overfitting. CSF. arXiv:1709.01604
- Panda, Tarsadiya, Sourav, Prathosh A.P., Karimireddy (2026). f-INE: a hypothesis testing framework for estimating influence under training randomness. ICLR. arXiv:2510.10510
- Gao, Unal, Rangamani, Zhu (2026). An illusion of unlearning? Assessing machine unlearning through internal representations. AISTATS, PMLR 300. arXiv:2604.08271
- Zhang, Lin, Bai, Mei (2024). Negative preference optimization: from catastrophic collapse to effective unlearning. COLM. arXiv:2404.05868 (from prior knowledge; not re-verified)
- Kamb, Ganguli (2025). An analytic theory of creativity in convolutional diffusion models. ICML 2025, PMLR 267. arXiv:2412.20292
- Kingma, Salimans, Poole, Ho (2021). Variational diffusion models. NeurIPS. arXiv:2107.00630 (from prior knowledge; not re-verified)
- Kingma, Gao (2023). Understanding diffusion objectives as the ELBO with simple data augmentation. NeurIPS. arXiv:2303.00848 (from prior knowledge; not re-verified)
- Kynkäänniemi, Aittala, Karras, Laine, Aila, Lehtinen (2024). Applying guidance in a limited interval improves sample and distribution quality in diffusion models. NeurIPS. arXiv:2404.07724 (from prior knowledge; not re-verified)
- Niedoba et al. (2024). Towards a mechanistic explanation of diffusion model generalization. arXiv:2411.19339 (from prior knowledge; not re-verified)
