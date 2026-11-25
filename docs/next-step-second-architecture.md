# Second architecture for the unlearning-audit paper — design

Status: proposed next step (designed 2026-11-25); not started, pending compute. Nothing here is built yet.

## Intent and why

The paper's finding is shown on one 6.48M-parameter class-conditional U-Net DDPM on CIFAR-10. The finding:

- the removed label's embedding row is never trained in an exact retrain, so the removed-label retrain reference is ill-posed;
- the null-label reference is calibrated;
- a zero-step re-draw of the row passes a removed-label audit, although the class is one row re-fit away;
- FID_O against one retrain inherits the problem (C5: retrain-to-retrain FID_O 2.1–4.3× the sampling floor; the unmodified model is not separated on 2 of 3 classes).

Two independent reviews (5/10, then 6/10) named scale and single-architecture as the main gap. The aim is a result with wider reach; there is no deadline. Retraining on Kaggle (quota refreshes 2026-11-28, about 30 wall-hours a week on 2×T4) is acceptable.

We chose **approach C**:

- **(A, main)** rerun the study on a small **DiT**, the transformer backbone family readers recognise;
- **(B, secondary, if quota remains)** our U-Net with **EDM-style label input** (one-hot → Linear, label dropout as the null), a genuinely different conditioning path.

## Goals

1. Test whether each part of the finding holds on DiT, under rules registered before any DiT data:
   - (a) the removed-label row/parameters are untrained in every retrain;
   - (b) the removed-label reference is heterogeneous and fails leave-one-out calibration;
   - (c) the null-label reference is calibrated;
   - (d) zero-step re-draws pass the removed-label audit at a rate similar to held-out retrains;
   - (e) a row re-fit restores the class in re-draws but not in retrains;
   - (f) FID_O against one retrain has a retrain floor that swallows re-draws (as C5).
2. With 4 retrains per class (not 3), each leave-one-out reference has 3 retrains (3 pairs in the floor), matching the main audit's reference count. This answers review points on statistics and on "2 vs 3 references".
3. **(B)** Show that the zero-gradient mechanism and (b)–(d) hold when labels enter as a one-hot vector through a Linear layer, with the null as the zero vector.

## Non-goals

- State-of-the-art sample quality, latent-space DiT or ImageNet.
- Reproducing official DiT FID numbers.
- Tuning or ranking unlearning methods (the approximate-method sweep is not repeated on DiT).
- The graded positive control and the train-the-row test. They stay open; this design does not include them.

## Anti-goals (would make this a failure even if it runs)

- Changing registered thresholds, classes, seeds or statistics after seeing DiT data.
- An original model too weak to generate its classes reliably. The behaviour gate ("forget-class rate ≤ worst retrain + 0.05") and the re-fit test would then be vacuous. Hence the quality gate in the spike.
- Wasting quota on runs that crash late or cannot resume. Every training checkpoints and resumes, and the Kaggle runner is smoke-tested before the long runs.
- Losing infrastructure again. The Kaggle driver and runner live in the repo, not the scratchpad, which a reboot wiped on 2026-11-22.

## Constraints

- **Compute.** Kaggle 2×T4 is about 30 wall-h/week, roughly 60 GPU-h/week with one training per GPU. Quota is ≈ 1 h until 2026-11-28. Local work runs on a laptop (Apple MPS).
- **Laptop MPS memory.** Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`, `LOW=0.4`, `torch.mps.empty_cache()` per model, and an Inception chunk of 50. Keep the charger in and run under `caffeinate -i -s`.
- **Existing U-Net cost.** One 40k-step retrain at batch 128 takes ≈ 3.2 T4 GPU-h.
- **Process.** Pre-register before data (PREREGISTRATION.md, with timestamps), unit-test the scoring before data, report failed predictions as findings, and label exploratory work. 

## Approaches considered

| | What | Trade-off |
|---|---|---|
| **A — DiT** | The official DiT model definition (adaLN-Zero, `LabelEmbedder` = `nn.Embedding(num_classes+1)`, table init N(0, 0.02²)), in pixel space on CIFAR-10 | Most recognisable; best for outreach. Tests a new backbone but the same kind of label lookup. 2–4× the U-Net's cost per training. |
| **B — EDM-style conditioning** | Our U-Net with labels as one-hot·√10 → Linear (EDM SongUNet style), label dropout = zero vector | Cheap (U-Net cost); tests a different conditioning path. Little outreach value on its own. |
| **C — A, then B if quota allows** | **Chosen** | Headline generality from A plus mechanism breadth from B, ordered so that B never delays A. |

## Decided parts of the how

### Step 0 — DiT spike (laptop, after C5; throwaway, exploratory, not registered)

**Purpose:** choose the DiT size and training length before spending Kaggle quota.

**Candidates:**
- DiT-S/2: depth 12, width 384, 6 heads, patch 2 → 256 tokens, ≈ 33M params;
- DiT-S/4: patch 4 → 64 tokens.

Both run in pixel space: `in_channels=3`, `input_size=32`, `learn_sigma=False`.

**Measure:**
- T4-equivalent time per step, from a short Kaggle timing run once quota is back, or estimated from MPS;
- sample quality after a short training: the gated rate at which the model generates its own class, using the existing CIFAR classifier and degenerate-sample gate.

**Decision rule** (written down before the spike runs):
- pick the cheapest candidate whose original can reach a **gated own-class generation rate ≥ 0.85, averaged over the 10 classes**, within the training length we can afford;
- if neither can within the budget, stop and re-plan.

The U-Net original is ≈ 0.98, which is the reference point.

### A — DiT study (Kaggle training, laptop audits)

**Objective and recipe.** ε-prediction MSE, the same as the U-Net (DiT's learned-sigma VLB term is off: stated deviation). Linear β schedule with 1,000 steps, CFG dropout 0.1, and AdamW with DiT's defaults (lr 1e-4, weight decay 0). EMA 0.9999, batch 128 unless the spike says otherwise, steps chosen by the spike, and the same for every run. AMP on CUDA.

**Classes:** automobile (1) and truck (9). These have the highest null-label power on the U-Net (SNR 3.7, 2.4), and the choice is fixed now, before any DiT data.

**Models (9 trainings):**
- 1 original on full CIFAR-10;
- 4 exact retrains per class (seeds 0–3), trained without the class (the retain set as in the U-Net retrains).

**Re-draw construction:** row k of `y_embedder.embedding_table` is replaced by a draw from N(0, 0.02² I), the DiT init distribution, with the same generator seeds 100–119 as C3-A.

**Studies, pre-registered as a new study (C6) with rules copied from the U-Net studies.** Thresholds are unchanged; any change for 4 retrains is written before data.

| Part | Rule source | What it tests on DiT |
|---|---|---|
| C6-0 | label_diag sanity | removed row's relative movement ≈ 0 in all 8 retrains |
| C6-1 | C2-F/G | removed-label pair heterogeneity; leave-one-out calibration fails in ≥ 1 rotation per class |
| C6-2 | C2-0/A, C2-C/D, C3-B-B | null-label trace present (SNR ≥ 1.5), calibrated ([0.8, 1.2] in every rotation), class-level, specific; revised homogeneity rule |
| C6-3 | C3-A1–A2 | re-draws pass the removed-label audit (≥ 1 of 20); spread; plus the post-hoc held-out-retrain pass rate, now registered |
| C6-4 | C4-X | cross-class retrains score null E < 0.5 (2 classes × 4 retrains = 8 per class) |
| C6-5 | C3-K K1–K3 | row re-fit: re-draws ≥ 0.9, retrains ≤ 0.5, original's unconditional rate > retrains' (registered this time) |
| C6-6 | C5 O1–O4 | FID_O with retrain floor (if C5's design works on the U-Net) |

**Leave-one-out with 4 retrains:** the held-out retrain is scored against the other 3, and the floor is the mean of 3 pairs. Pass-rate comparisons use 3 references on both sides.

### B — EDM-style conditioning (only after A's trainings are queued and quota remains)

- Our U-Net, unchanged except the label path:
  - `map_label = Linear(10, emb_dim, bias=True)`, applied to one-hot·√10 (EDM SongUNet style);
  - CFG null = zero vector (label dropout 0.1);
  - Adam(W), weight decay 0.
- The untrained parameter is **column k** of `map_label.weight`; the bias is trained.
- A re-draw re-initialises column k from the layer's init distribution.
- Classes automobile and truck; 1 original + 3 retrains per class (7 trainings × 3.2 GPU-h). A 4th retrain per class is added if quota allows, decided before data.
- Registered as C7, with the C6 rules.

### Code changes (where the existing code is touched)

**Model factory.** `train.py` builds models through `build_unet(cfg)` today. It gains a factory on `cfg.model.type`: `ddpm_unet_small` (current), `dit` and `ddpm_unet_onehot`. Checkpoints record their type.

**Label-parameter interface.** Ten files read or write `model.class_emb.weight[k]` directly:

- `src/analysis/`: `c3_consequence.py`, `c3_knowledge.py`, `c4_controls.py`, `c4_fd.py`, `c5_fido.py`, `label_diag.py`;
- `src/diffusion/`: `rescore_behaviour.py`, `timestep_decomp.py`, `unlearn.py`, `unet.py`.

Every model gets one interface: `null_class`, `get_label_param(k)`, `set_label_param(k, v)` and `draw_label_param(k, generator)`. The audits call the interface. U-Net results must reproduce bit for bit after the refactor: re-run one existing score (e.g. `label_diag` on class 8) and compare against the JSON.

**DiT model.** The official DiT `models.py` is vendored, with attribution, into `src/diffusion/dit.py`, adapted only for `in_channels=3`, `learn_sigma=False` and the interface above.

**Kaggle.** The driver and runner move into the repo (`scripts/kaggle/`). A run takes a list of (model type, class, seed) jobs, one per GPU, checkpoints every 2,000 steps and resumes from the latest checkpoint. A short smoke run on Kaggle precedes the long runs.

**Audits** run on the laptop after training, using the existing scoring code through the interface.

## Order and rough budget

1. Now → 2026-11-26 01:00: C5 finishes on the laptop (unchanged).
2. 2026-11-26:
   - DiT spike on the laptop;
   - refactor to the label interface, plus the bit-for-bit U-Net check;
   - vendor DiT;
   - write and timestamp C6 (and C7) registrations;
   - move the Kaggle driver into the repo.
3. 2026-11-28 onwards, Kaggle:
   - **A:** 9 DiT trainings. At an assumed 5–12 GPU-h each (set by the spike) that is 45–110 GPU-h, ≈ 1–2 weeks of quota.
   - **B:** 7 U-Net trainings ≈ 23 GPU-h, ≈ 0.5 week, after A is fully queued.
4. As checkpoints arrive: laptop audits, C5/C6 scoring, write-up.
5. Paper revision; a fresh independent review.

## Calls made after the builder check (2026-11-25)
1. **Spike and gate.**
   - The spike only picks the size and gives an extrapolated estimate. It may use ≤ 3 GPU-h of Kaggle for a timing and short-quality run.
   - Spike models are discarded.
   - The binding gate is on the full-length **original**, trained first on Kaggle: mean gated own-class rate ≥ 0.85. If it misses, no retrain is launched and the plan is revisited.
2. **"Similar pass rate" (C6-3).** No equivalence claim, because n = 8 held-out rotations is too small.
   - Registered rule: A1 (≥ 1 of 20 re-draws pass per class).
   - Registered report: the re-draw vs held-out-retrain pass rates with Wilson CIs and a two-sided Fisher exact test. The paper may say only "not distinguishable at this n" unless the test says otherwise.
3. **Re-fit (C6-5).** Adam lr is scaled to the init scale, 0.05 × 0.02 = 0.001: the same ≈ 5 %-of-init-scale step per update as C3-K. Steps (400), batch (64) and data (1,000 disjoint class images) are as in C3-K. Models: the first 3 passing re-draws per class in seed order, and all 4 retrains. K1–K3 are registered.
4. **4 retrains, non-leave-one-out statistics.**
   - Heterogeneity and homogeneity ratios (C2-F, C3-B-B) are computed on each of the 4 three-retrain subsets; the registered value is the median over subsets, which keeps the U-Net thresholds comparable.
   - SNR uses the mean over all 6 pairs.
   - Leave-one-out: 3 references, as above.
5. **Cross-class (C6-4).** Each class's 4 cross-class retrains are the other class's retrains, all from a near class; each is scored against all 4 class-k retrains. This is narrower than U-Net C4-X (9 retrains, near and far) and is stated as such. A third class depends on the budget.
6. **FID_O on DiT (C6-6).** Included only if C5's O2 held (the retrain floor exceeds the sampling floor). Its sampling (≈ 140 runs) goes on Kaggle within the A budget, not on the laptop.

## Left to the builder

- Exact DiT training length and batch (from the spike, within the decision rule above).
- Kaggle job packing (which jobs share a session).
- Whether audits are run per class as soon as that class's retrains finish.
- Figure design for the DiT results (extend existing figures or add a panel).
