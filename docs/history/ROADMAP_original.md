> **Historical document.** The original ten-week plan. The project later narrowed to the diffusion finding in the top-level [README](../../README.md); unticked items here were superseded, not forgotten. Kept unedited apart from link paths.

# Roadmap and progress tracker

Ten weeks, one project at a time, minimum-viable version first. Tick boxes as you go; GitHub renders the counts in the file view. Every GPU run also gets an entry in [`EXPERIMENT_LOG.md`](../../EXPERIMENT_LOG.md).

**Start date:** _fill in_ · **Target venue:** ICLR 2027 or NeurIPS 2027 workshop on unlearning / safety (4 pages); IndoML / CODS-COMAD as the Indian venue; TMLR as stretch.

## Milestone 0 — Pre-registration v0.2 (done 2026-10-18)
- [x] Independent review ([`REVIEW.md`](REVIEW_2026-10-18.md))
- [x] H3 restricted to full-parameter tracks with norm control; H2 promoted; amendments logged in [`PREREGISTRATION.md`](../../PREREGISTRATION.md)
- [x] Scaffold committed (configs, requirements, src layout, log template)

## Milestone 1 — Weeks 1–2: reading and ownership (no GPU)
- [ ] Foundations block of [`READING.md`](READING.md) read, with a half-page note per paper in `paper/notes/`
- [ ] Project block of [`READING.md`](READING.md) read, same notes
- [ ] Derive the influence-function formula and the one-step unlearning estimate by hand; write down why damping is needed (`paper/notes/derivation_influence.md`)
- [ ] Write one paragraph in your own words on why a rank-1 update explains near-chance MIA, intact probes and fast relearning at once
- [ ] Rewrite the three hypotheses in your own words in a new amendment (same tests; your wording)
- [ ] Semantic Scholar sweep for DDPM class-unlearning audits and rank-vs-relearning; record in `REVIEW.md` §6
- [ ] Install OpenUnlearning; confirm Qwen2.5-0.5B support; log the decision

## Milestone 2 — Weeks 3–4: base models and control
- [ ] 30-minute DDPM pilot to fix architecture and step count; FID of the pilot logged
- [ ] D1 base class-conditional DDPM trained (3 h schedule); training curve and FID table
- [ ] D2 one oracle retrain on the retain set (same schedule, same seed)
- [ ] L1 LoRA fine-tune on the TOFU-style set via OpenUnlearning; L2 one oracle
- [x] `src/audit`: MIA (loss/SecMI/offline-LiRA/CMIA), attribution (TRAK-style projected gradients), relearn, probe, spectrum + norm — written 2026-10-20, not yet run on GPU
- [x] `src/llm/audit.py`: LLM audit runner (eval, MIA, attribution, probe, relearn, spectrum) wired into `llm_sweep` — written, not yet run on GPU
- [ ] **Expected-to-fail control passes:** "do nothing" checkpoint shows MIA AUC well above 0.5, full probe accuracy, zero relearning steps, zero attribution drop, zero delta (logged)
- [x] `src/diffusion`, `src/llm`, `src/analysis` written 2026-10-20, not yet run on GPU
- [ ] `notebooks/00_smoke_test` passes on Kaggle (tiny UNet + 2 methods + audit)
- [ ] Smallest end-to-end audit on a real 2k-image subset (after D1/D2)

## Milestone 3 — Weeks 5–7: minimum-viable sweeps (~17 GPU-h), DDPM first
- [ ] D3 gradient ascent + label annealing × 2 seeds × 5% forget size (matched steps)
- [ ] D4 MIA + attribution on 2k-image subsets; attribution-drop bar chart with CIs (H2)
- [ ] D5 relearning curves, probes, Δθ spectra and norms; rank-vs-relearning scatter with norm shown (H3, trend only)
- [ ] L3 GA, GradDiff, NPO × 2 seeds on OpenUnlearning; L4 on a 200-example evaluation set
- [ ] ROC curves per method (H1)
- [ ] Every run has a complete `EXPERIMENT_LOG.md` entry

## Current state and remaining tasks (updated 2026-11-24)

**Where the project is.** The diffusion track is complete through study C3. The finding changed from the original plan (see `paper/DRAFT.md`):
- In every one of 12 exact retrains, the removed label's embedding row is untrained, so that reference is ill-posed.
- A null-label reference is calibrated (C2, C3-B).
- A zero-step re-draw of the row passes the removed-label audit on 30–60 % of draws (C3-A, 12/12 predictions held).
- Re-learning only that row restores the class to ≥ 95 %, versus ≤ 25 % for retrains (C3-K, exploratory).

Kaggle quota: 0.97 GPU-h left until the 2026-11-28 00:00 UTC refresh.

**Write-up (no GPU)**
- [x] Paper figures from the existing `results/figures/*.json` (done 2026-11-24; `paper/figures/fig1–fig5`, PDF + PNG):
  - (1) the label-row norms for the retrain and the original;
  - (2) removed-label vs null-label E with leave-one-oracle-out CIs, per class;
  - (3) C3-A pass rates of re-draws vs methods;
  - (4) C3-K re-fit rates;
  - (5) per-timestep pair ratios.
- [x] Figure script: `python -m src.analysis.paper_figures` rebuilds every figure by a single command (done 2026-11-24).
- [x] Workshop version: `paper/workshop/main.tex` → `main.pdf`, 4 pages of main text + references + 2-page appendix (C1 negative result, registrations, method details). Build with `cd paper/workshop && tectonic main.tex`. Uses a generic layout until a venue is chosen. `paper/DRAFT.md` stays as the long version.
- [x] Citation check (2026-11-24, arXiv API plus full texts): no wrong IDs, authors or years. Fixed the descriptions of Selective Amnesia/EraseDiff (no retrain baseline), Ebrahimpour-Boroojeny (target built from the original model), Hayes, Lanyon, Panda UAI (a classifier paper) and Land & Bartolo. Added SFD at ICLR 2025, Kim, Cha & Kim 2026, DDPM and CIFAR-10. `paper/workshop/refs.bib` is author-verified.
- [ ] Optional: cite a source for the gradient-difference baseline and for the CIFAR-10 gating classifier (App. D).
- [x] Outside read by an agent not given our conclusions (2026-11-24): it checked every number, scored the paper 5/10 (weak accept once overclaims are fixed), and all its verified issues were fixed in the paper, DRAFT, log and pre-registration (dated notes). See journal §7.17.
- [ ] Outside read by a human collaborator or advisor before submission.
- [ ] Optional, laptop-only, forward passes, no training: the *common-row control*. Give all three retrains the same removed-label row and recompute the pair ratios and LOO E, to establish the mechanism causally. Register a short amendment first. This is the reviewer's top suggestion that needs no new training.
- [ ] **Decision for the author:** pick a venue and format to its template. Candidates found on 2026-11-24 (check the official sites):
  - SaTML 2027 workshops: papers around 2026-12-18 if a relevant workshop runs; the event is 2027-05-03.
  - ICLR 2027 workshops: the list was not yet published; deadlines are typically around February.
  - ICML 2027 workshops: early August 2027, deadlines around May.
  - TMLR: rolling submissions, full-length version.
  - arXiv preprint at any time.

**Optional strengthening (after the 2026-11-28 quota refresh)**
- [ ] Extend C3-A/C3-K to 1–2 more classes, pre-registered before data, to firm up the "30–60 %" range. This needs 3 retrains per class, about 6–7 GPU-h each.
- [ ] A third seed for the approximate methods at a tuned operating point, if a reviewer asks for method-level claims.
- [ ] Re-check the C3-B homogeneity margin (passed by 0.01 on truck) on any new class.

**Housekeeping**
- [x] Make the hand-rolled test runner in `tests/test_audit_cores.py` assert the number of tests it ran (done 2026-11-24: it now fails if any defined test did not run). On 2026-11-24 the last 7 tests were found to be silently skipped; this is fixed, and all 26 now pass.
- [ ] Keep `ROADMAP.md`, `README.md` and the journal inventory (§6) in sync when anything new runs.

**Deferred / future work (not in this paper)**
- [ ] LLM track (Qwen2.5-0.5B, TOFU-style): smoke-tested only, never run at scale.
- [ ] Subset-regime and method-comparison results: appendix at most (`RESEARCH_JOURNAL.md` §7.10).
- [ ] The stretch items below.

The milestones below are the original 10-week plan (2026-10-18), kept for the record.

## Milestone 4 — Weeks 8–10: analysis and write-up
- [ ] H1, H2, H3 each marked confirmed / refuted / under-powered against the pre-registered thresholds
- [ ] Negative-results section drafted
- [ ] 4-page workshop paper in `paper/`
- [ ] One-page summary for Prathosh A.P. (influence result, variational method's position, where the f-INE-style test would go)
- [ ] `python -m src.reproduce --figure 1` works from cached checkpoints and projected gradients
- [ ] Decide whether the full version (all methods × 3 seeds × 3 forget sizes, LiRA/CMIA, full-FT LLM arm) is worth ~45 more GPU-hours

## Stretch (only after Milestone 4)
- [ ] Full-FT LLM arm with 8-bit Adam for H3, if not already run in Milestone 3
- [ ] S1 two more oracle retrains for the f-INE-style seed null at FDR 0.1
- [ ] Retain-availability ("data constraints") and forget-size ablations
- [ ] S2 Stable Diffusion 1.5 / SD-Turbo concept erasure on a benign object concept
- [ ] One-page summary for Chiranjib Bhattacharyya (per-layer component analysis)
