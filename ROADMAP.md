# Roadmap and progress tracker

Ten weeks, one project at a time, minimum-viable version first. Tick boxes as you go; GitHub renders the counts in the file view. Every GPU run also gets an entry in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

**Start date:** _fill in_ · **Target venue:** ICLR 2027 or NeurIPS 2027 workshop on unlearning / safety (4 pages); IndoML / CODS-COMAD as the Indian venue; TMLR as stretch.

## Milestone 0 — Pre-registration v0.2 (done 2026-10-18)
- [x] Independent review ([`REVIEW.md`](REVIEW.md))
- [x] H3 restricted to full-parameter tracks with norm control; H2 promoted; amendments logged in [`PREREGISTRATION.md`](PREREGISTRATION.md)
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
- [x] `src/audit`: attribution (TRAK-style projected gradients), relearn, probe, spectrum + norm — written 2026-10-20, not yet run on GPU; **mia is an interface + stub, not implemented**
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
