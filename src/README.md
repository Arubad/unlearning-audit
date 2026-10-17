# src/ — planned modules

- `models/` — small UNet DDPM (CIFAR-10, 32×32) with EMA; class-conditional, 5–10M params; the LLM track lives in OpenUnlearning (Qwen2.5-0.5B, LoRA r16 and a full-FT arm with 8-bit Adam for H3), this package only wraps its checkpoints; checkpoint I/O (full UNet weights; LoRA adapters or full weights for the LLM).
- `unlearn/` — `gradient_ascent`, `gradient_difference`, `label_annealing` (UAI 2025 style), `variational` (TMLR 2026 style), `npo` (LLM track), `oracle` (retrain on retain set); all share one matched-compute loop.
- `audit/` — the library, every module takes `(ckpt_original, ckpt_unlearned, data)`:
  - `mia.py` — loss-based, LiRA-style and class-geometry (CMIA) AUC; multi-timestep losses for diffusion, cached per checkpoint.
  - `attribution.py` — gradient-similarity and damped influence functions on 4k-dim random-projected per-example gradients (CPU matrix products once cached); λ and LoRA-subspace sensitivity.
  - `relearn.py` — fine-tune on 1–5 % of the forget set; steps to 90 % recovery.
  - `probe.py` — linear probes on frozen intermediate features.
  - `spectrum.py` — per-layer SVD of Δθ, top-5 mass, effective rank, and ‖Δθ‖_F (the norm H3 controls for); unconditional-drift measurement for diffusion. Full-parameter checkpoints only.
- `analysis/` — attribution-drop CIs, influence-vs-relearning correlation (H2 secondary), rank-vs-relearning partial correlation controlling for norm (H3), seed-to-seed null from oracle retrains (f-INE style, stretch), all figures.
- `utils/` — config hashing, seeding (data order, init, noise, generation), W&B wrappers.
- `reproduce.py` — `python -m src.reproduce --figure 1` from cached checkpoints and projected gradients.
