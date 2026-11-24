# Suppressed or Erased? A Depth-of-Erasure Audit of Approximate Unlearning in Diffusion Models

*Working draft, started 2026-11-19. Sections 1–5 and 7 are written before any sweep result exists, on purpose: the method and the predictions are fixed first. Section 6 is a skeleton filled from `RESEARCH_JOURNAL.md` §7 when the runs finish. Citations are (Author Year); the full list with arXiv IDs is in `RESEARCH_JOURNAL.md` §8 (verified 2026-11-19).*

---

## Abstract (template; numbers pending)

Approximate unlearning methods for generative models are usually judged by one shallow statistic: whether the model still generates the forgotten concept, or whether a membership-inference attack still separates the forgotten examples. We argue that "forgotten" is not one property but a *ladder* of increasingly strict ones: behaviour, average-case membership, calibrated membership, representation, recoverability, and the downstream imprint of the data on what was retained. Exact unlearning, defined as indistinguishability from a retrain, implies all of them. We audit five approximate methods for class unlearning in a class-conditional DDPM on CIFAR-10 against two independent retrains, and measure each rung relative to the retrain-to-retrain noise floor. We introduce an *oracle-normalised closeness* $E \in (-\infty, 1]$ that is 0 for doing nothing, 1 for an independent retrain, and negative for overshooting. It replaces influence-function statistics, which we show are not monotone in unlearning quality. [RESULT: at which rung each method fails; $E_\text{forget}$ vs $E_\text{near}$; rank/alignment vs relearning, with the truncation intervention.] [RESULT: the MIA control under the generalisation regime.] A corroborating experiment on LLM unlearning (TOFU, Qwen2.5-0.5B) [RESULT].

---

## 1. Introduction

Machine unlearning asks for a model that behaves as if some training data had never been seen (Cao & Yang 2015; Bourtoule et al. 2021). For deep networks the only exact method is retraining. Practical methods therefore *approximate* it: gradient ascent on the forget set, ascent balanced by retain descent, label remapping, regularised variants (Golatkar et al. 2020; Heng & Soh 2023; Fan et al. 2024; Panda et al. 2025). These methods carry no guarantee. They are judged empirically, and the judgement is usually made on the statistic the method was tuned to move.

This is a weak test for two reasons. First, from weights alone, approximate unlearning cannot be verified in general (Thudi et al. 2022), and average-case evaluations overstate privacy (Hayes et al. 2024). Second, in the LLM literature a sequence of results shows that what unlearning removes is often an *access path*, not the information. Fine-tuning learns thin wrappers (Jain et al. 2024), safety behaviour is "a few tokens deep" (Qi et al. 2025), and unlearned knowledge is recovered by a few steps of fine-tuning or by probing the weights (Deeb & Roger 2024; Łucki et al. 2024). Whether the same holds for diffusion models has been argued case by case, not measured systematically against a retrain.

**Our position.** The right question is not *whether* a method forgets, but *how deep* its forgetting goes. We formalise a six-rung ladder (§3) in which each rung is a strictly stronger functional of the model, and exact unlearning passes every rung. For each rung we compare the unlearned model with the **distribution** of retrains, estimated from two independent oracle seeds, rather than with the original model or with zero.

**Contributions.**
1. **A depth-of-erasure audit protocol for diffusion models.** It covers behaviour, three MIAs (loss, SecMI, offline LiRA with the oracles as OUT models), a null-conditioned linear probe with an oracle reference, censored relearning time, gradient alignment, update spectra, and a closeness statistic on four query sets. Each rung is reported against the oracle-seed noise floor.
2. **Oracle-normalised closeness $E$.** It is a calibrated, signed distance to the retrain distribution, with an SNR guard and bootstrap CIs. We show why the natural influence-function alternative (the drop in |TRAK|) increases under exact retraining and cannot rank methods (§3.5).
3. **A causal test of the low-rank-update explanation for fast relearning.** It truncates the unlearning update to rank $k$ at matched Frobenius norm. This separates the update's rank from its norm and from the identity of the method.
4. **Pre-registered predictions and an honest control.** Hypotheses, thresholds and per-method predictions were fixed before the data. They include a prediction that the standard MIA control may *fail* in the generalisation regime of diffusion models (Kadkhodaie et al. 2024). We report it either way.

---

## 2. Related work

**Exact and certified unlearning.** Exact unlearning reproduces the retrain distribution (Ginart et al. 2019; Bourtoule et al. 2021). Certified unlearning bounds the divergence from it (Guo et al. 2020; Sekhari et al. 2021). Neither scales to the generative models practitioners unlearn today, which is why evaluation has carried the burden.

**Unlearning in diffusion models.** Methods range from fine-tuning objectives that redirect a concept (Heng & Soh 2023, Selective Amnesia; label annealing, Panda et al. 2025) to saliency-masked gradient ascent (Fan et al. 2024, SalUn), which documents the instability of unmasked ascent. Evaluations mostly report generation rates and FID. A few report MIA. To our knowledge, none compares several rungs against multiple retrains.

**Evaluating unlearning.** Relearning time as a metric dates to Golatkar et al. (2020). Recent work predicts relearning from the update norm (Siddiqui et al. 2025), which is why our rank analysis controls for norm. KLOM (Georgiev et al. 2025) compares output distributions with many retrains, and f-INE (Panda et al. 2026) frames unlearning evaluation as a test against the retrain distribution. Our $E$ is a two-oracle, loss-based instance of the same idea, built for a free-tier compute budget. In LLMs, adversarial evaluations show recoverable "unlearned" knowledge (Łucki et al. 2024; Deeb & Roger 2024; Hu et al. 2025; Gao et al. 2026).

**Memorisation and membership in diffusion models.** DDPMs can memorise and regurgitate training images (Carlini et al. 2023), and dedicated attacks exist (SecMI, Duan et al. 2023). Whether a given model memorises depends on its regime: with enough data, models trained on disjoint halves learn nearly the same score (Kadkhodaie et al. 2024). This bounds what any MIA can show, and it motivates rungs that do not depend on memorisation.

**Influence and attribution.** Influence functions (Koh & Liang 2017) are fragile in deep networks (Basu et al. 2021) and approximate a proximal response rather than leave-one-out retraining (Bae et al. 2022). TRAK (Park et al. 2023) and datamodels (Ilyas et al. 2022) make attribution scale. We use TRAK only as a signed, secondary readout whose validity we check against the oracle (§3.5).

**Low-dimensional structure of fine-tuning.** Gradient descent concentrates in a tiny subspace (Gur-Ari et al. 2018), and tasks have low intrinsic dimension (Li et al. 2018; Aghajanyan et al. 2021), which LoRA exploits (Hu et al. 2021). This motivates H3. Effective rank follows Roy & Vetterli (2007).

---

## 3. The depth-of-erasure ladder

**Setup.** $A$ is a randomised training algorithm, $D = D_f \cup D_r$ the training set, $\theta_0 \sim A(D)$ the original, $O \sim A(D_r)$ the oracle, and $\theta_u = U(\theta_0, D_f, D_r)$ the unlearned model with $\Delta = \theta_u - \theta_0$. Exact unlearning means $\theta_u \overset{d}{=} O$, so *every* functional $\phi$ satisfies $\phi(\theta_u) \overset{d}{=} \phi(O)$. We test six functionals, ordered so that passing a rung does not imply passing the next.

| Rung | Functional $\phi$ | What passing leaves open |
|---|---|---|
| 0 Behaviour | forget-class generation rate (CIFAR classifier, 88.6 % acc.) | knowledge gated, not removed |
| 1 Average-case membership | loss-MIA, SecMI | per-example memorisation |
| 2 Calibrated membership | offline LiRA with oracles as OUT | representation-level knowledge |
| 3 Representation | linear probe on null-conditioned features, vs the oracle's probe | cheap re-exposure by fine-tuning |
| 4 Recoverability | steps to 90 % recovery on 5 % of $D_f$; $\cos(g_f(\theta_u), \Delta)$ | the imprint on retained data |
| 5 Imprint | closeness $E$ on near and far retain classes | — |

### 3.1 Why shallow rungs are weak evidence
A method tuned until loss-MIA is at chance passes rung 1 almost by construction. LiRA (Carlini et al. 2022) asks instead whether each forget example's loss lies where *a retrain* would put it. In the generalisation regime (Kadkhodaie et al. 2024), even the original model may not be separable from held-out data. Rungs 1–2 then lose power for every method, including doing nothing (prediction P1, §5.4).

### 3.2 Representation needs an oracle reference
A probe measures decodable, not used, information (Belinkov 2022). A DDPM that never saw a cat still separates cats from other images through low-level statistics shared with other animals. Probe accuracy is therefore only evidence of suppression if it exceeds the **oracle's** probe accuracy. Features are extracted with null-class conditioning, so the label cannot leak into them.

### 3.3 Recoverability: a first-order account
Relearning descends $L_f$ from $\theta_u$; one step moves by $-\eta g_f(\theta_u)$. If $U$ climbed $L_f$ along a few directions, then $g_f(\theta_u)$ still points along $\Delta$, i.e. $a = \cos(g_f(\theta_u), \Delta) \to 1$. Descent then retraces $-\Delta$, a short walk back to $\theta_0$. An oracle's displacement from $\theta_0$ is the difference of two independent long runs, and is not aligned with $g_f$. We therefore predict fast relearning exactly when $a$ is high and $\Delta$ is concentrated. Relearning is right-censored at a step budget; censored cells are tied at the maximum rather than dropped.

### 3.4 Mechanism: low rank at matched norm (H3)
Per-example gradients of a linear layer are outer products $\delta x^\top$. In single-class ascent they share directions through one class embedding, so a short ascent run accumulates a low-rank sum. Two confounds are handled.
- **Rank vs norm.** We use partial Spearman controlling for $\|\Delta\|_F$, and a matched-steps `retain_finetune` control, since any short run is somewhat low-rank.
- **Rank vs method identity.** We intervene by replacing $\Delta$ with its per-layer rank-$k$ truncation, rescaled to the same total norm, for $k \in \{1,4,16,64,\text{full}\}$, and re-measure forgetting and relearning.

### 3.5 Imprint: oracle-normalised closeness
**Why not influence.** TRAK estimates, at the model where it is evaluated, the effect of up-weighting $f$. At the oracle, $f$ is unfitted, so its gradients are *larger* and |TRAK| grows under exact retraining. On our smoke model the "attribution drop" at the oracle was −1.95. A statistic that the exact solution fails cannot rank approximations.

**Definition.** Let $\ell_m(q,t)$ be the per-example $\epsilon$-MSE of model $m$ on query $q$ at timestep $t$, with noise fixed per (position, timestep) across models (common random numbers). Then
$$d(a,b)=\Big(\tfrac{1}{|Q||T|}\textstyle\sum_{q,t}(\ell_a-\ell_b)^2\Big)^{1/2},\qquad E(u)=\frac{d(\theta_0,O)-d(\theta_u,O)}{d(\theta_0,O)-d_\text{floor}},\quad d_\text{floor}=d(O_0,O_1).$$
- **Reference values.** $E=0$ for doing nothing and $E=1$ for an independent retrain. $E<0$ means the model moved away from the retrain, i.e. overshoot or collateral damage.
- **CIs.** These bootstrap over queries.
- **SNR guard.** When $d(\theta_0,O)/d_\text{floor}<1.2$, removing $D_f$ left no measurable trace on $Q$, and $E$ is reported as undefined rather than as noise.
- **Query sets.** *forget*, *heldout* (test cats), *near* (bird, deer, dog, frog, horse) and *far* (airplane, automobile, ship, truck). The near and far sets are drawn from retain images, which every model trained on.

**Why *near* is the headline.** A retrain without cats changes what the model learned about dogs and horses from cat images. That is the data's imprint, and suppression does not reproduce it. H2: approximate methods reach substantial $E_\text{forget}$ (the certificate they target) but low $E_\text{near}$.

**LLM instance.** The loss is the per-example mean answer-token NLL on TOFU. The sets are forget, a held-out slice never trained on, retain-train (near), and `world_facts` (far, pre-training knowledge).

---

## 4. Experimental setup

**Model and data.** The model is a class-conditional DDPM (Ho et al. 2020): a 6.48M-parameter UNet with channels 64·(1,2,2), 2 residual blocks per level, attention at 16×16, and 1,000 linear-β steps. It is trained for 40k steps at batch 128 with learning rate 2e-4, EMA 0.9999 and 10 % classifier-free-guidance dropout, on CIFAR-10. The forget class is 3 (cat). Samples use DDIM with 50 steps and guidance 2.0.

**Regimes.**
- *Class:* all 5,000 training cats are forgotten, so the oracle never sees a cat.
- *Subset:* a fixed 2,500 cats are forgotten; the oracle still sees the other 2,500.

The forget set is fixed per regime (`split_seed 0`), so the oracle seeds replicate one counterfactual. Oracle initialisation draws from a different seed stream than the base model's, so the two oracles are exchangeable.

**Oracles.** Two seeds per regime, with the same schedule as the base model. Their mean ε-MSE over the last 2k steps is 0.0287 and 0.0311 in the class regime, against 0.0294 for the base model.

**Methods** (2,000 steps each, learning rate 3e-5, batch 64, gradient clip 1.0):
- *gradient ascent* (per-step ascent loss clipped at 1.0);
- *gradient difference* (ascent on $D_f$ plus descent on $D_r$);
- *label annealing*, our reading of Panda et al. (2025): forget images trained with their label annealed from cat to the null class, plus retain descent;
- *variational EWC*: ascent plus a diagonal-Fisher quadratic anchor;
- *retain fine-tune*, a control that never touches $D_f$;
- *do nothing*, the expected-to-fail control;
- *oracle*.

Every cell is run with 2 seeds. Both "our reading" implementations are flagged as such in the pre-registration.

**Audit hyperparameters.**
- *MIA:* 1,000 members and non-members; timesteps {50, 200, 400, 600, 800}; SecMI at t = 200.
- *Closeness:* 256 queries per set over the same timesteps.
- *Probe:* 1,500 examples per class, features at t = 200.
- *Relearning:* 5 % of $D_f$, learning rate 1e-4, evaluated every 50 steps, budget 1,000 steps, target 90 % recovery of the original cat-generation rate.
- *TRAK:* CountSketch projection to 4,096 dimensions, damping 1e-2.
- *Truncation:* $k \in \{1,4,16,64,\text{full}\}$ on the gradient-ascent cells.

**Compute.** Everything runs on free-tier GPUs: the base model on Colab T4, the oracles and sweeps on Kaggle 2×T4, with one seed per GPU. Runs are chained under an 11 h session cap and a 30 GPU-h weekly quota, with atomic checkpoints every 2k steps. Provenance for every checkpoint is in `EXPERIMENT_LOG.md`.

**LLM corroboration.** Qwen2.5-0.5B-Instruct is fine-tuned on TOFU (Maini et al. 2024), with LoRA r = 16 and one full-fine-tune arm for H3. Oracles are trained on retain only, with 2 seeds. The methods are GA, GradDiff, NPO, oracle and do nothing. The audit covers ROUGE-L forgetting, NLL, Min-K% and LiRA MIAs, a probe, ROUGE-L relearning, spectra and $E$.

---

## 5. Hypotheses and pre-registered predictions

**Hypotheses** (thresholds in `PREREGISTRATION.md`, amended 2026-11-18 before any sweep data):
- **H1, suppression not erasure.** Gradient ascent and label annealing pass rung 1, but fail rungs 3–4. Their probe stays above the oracle's, and they relearn in under 1/5 of the oracle's steps.
- **H2, the imprint survives.** $E_\text{near}<0.5$ with its CI below 0.5, while $E_\text{forget}-E_\text{near}\ge 0.25$.
- **H3, low rank at matched norm predicts fast relearning.** Top-5 energy above 0.7 for approximate methods and below 0.3 for the oracle. Partial Spearman ≥ 0.7. The truncation intervention reproduces it.

**Specific predictions** (written 2026-11-19, before any cell finished):
- **P1.** Do-nothing loss-MIA AUC ≈ 0.52–0.62. If it is ≤ 0.55, the MIA control fails, and we report that as a finding about the generalisation regime.
- **P2.** In the class regime, ascent-type methods show $E_\text{forget}>E_\text{heldout}$: they act on the training images, not the class.
- **P3.** $E_\text{near}<0.5$ for every approximate method.
- **P4.** $d(\theta_0,O)$ is larger on near than on far classes, so $E_\text{far}$ is more often undefined.
- **P5.** Rank 1–4 truncations of the gradient-ascent update reproduce most of its forgetting at matched norm.

Per-method expectations, with their reasons, are tabulated in `RESEARCH_JOURNAL.md` §3.

---

## 6. Results — *skeleton, filled from the journal*

- **6.1 Controls.** Do-nothing on every axis (P1); the oracle cell's $E$ equals the signal-to-floor ratio by construction.
- **6.2 Figure 1: the closeness ladder.** $E$ per method × query set, with CIs and the SNR-guarded cells marked (H2, P2–P4).
- **6.3 Table 1: rung-by-rung pass/fail per method**, against the oracle distribution (H1).
- **6.4 Figure 2: mechanism.** Relearning vs effective rank at matched norm; alignment $a$; truncation curves (H3, P5).
- **6.5 Subset regime.** How the ladder changes when the oracle still knows the class.
- **6.6 LLM corroboration.**
- **6.7 What failed.** Predictions that did not hold, and why we now think they did not.

---

## 7. Limitations (known before the results)

- **Scale.** The models are small: a 6.5M-parameter CIFAR DDPM and a 0.5B LLM. The memorisation regime, and hence MIA power, depends on scale (Kadkhodaie et al. 2024; Carlini et al. 2023), so rungs 1–2 may behave differently at Stable Diffusion scale.
- **Two oracle seeds.** The noise floor is estimated from one oracle pair per regime, so its own uncertainty is not propagated into $E$'s CI. More seeds (S1) would tighten it.
- **Our-reading implementations.** Label annealing and variational EWC are reconstructions from the papers, not the authors' code.
- **Losses, not distributions.** $E$ compares per-example losses at fixed noise, which is a projection of the full output distribution. KLOM-style margin distributions over many oracles are stronger.
- **Platform heterogeneity.** The base model and the oracles were trained on different software stacks (Colab vs Kaggle torch builds). One oracle's data order changed after a resume. Both are distribution-preserving, but are disclosed.
- **Matched steps.** Every method gets 2,000 steps at one learning rate. A per-method tuned budget could move methods along the ladder, so our claims concern the methods at matched cost.
