# src/ — module map

| Package | Files | What it does |
|---|---|---|
| `common/` | config, seed, explog, stats, kaggle_env | copied from the DPO repo: YAML configs + cell overrides + hash, seeding, append-only log, bootstrap/Spearman/partial-Spearman, Kaggle paths/dtype |
| `diffusion/` | `unet.py`, `ddpm.py`, `data.py`, `train.py`, `classifier.py`, `unlearn.py`, `sample.py` | 5–10M-param class-conditional UNet with a null class (CFG) and `features()` for probes; linear-schedule DDPM + DDIM-50 sampler + EMA; CIFAR splits (forget/retain/held-out, seeded, index-based); D1 base + D2 retain-only oracle; CIFAR-10 classifier for the forget-class generation rate; the five unlearning methods + do_nothing under matched steps, recording ‖Δθ‖; conditional/unconditional sampling and drift |
| `llm/` | `data.py`, `finetune.py`, `unlearn.py`, `eval.py` | TOFU splits + chat formatting; LoRA and full-FT fine-tune + retain-only oracle (8-bit AdamW when available); GA / GradDiff / NPO / oracle / do_nothing; forget/retain ROUGE-L (own LCS), forget loss, utility |
| `audit/` | `mia.py`, `attribution.py`, `relearn.py`, `probe.py`, `spectrum.py`, `report.py`, `run_audit.py` | the library: MIA (**interface + stub** — implement per the docstring), TRAK-style influence on 4096-dim projected gradients, few-shot relearning + steps-to-90%, logistic-regression probe on frozen mid features, per-layer SVD of Δθ (effective rank, top-5 energy, norm), record assembly + do-nothing control; `run_audit.py` runs every axis for one diffusion cell |
| `analysis/` | `hypotheses.py`, `figures.py` | pre-registered H1/H2/H3 tests applied mechanically (thresholds from PREREGISTRATION.md), the H2↔H3 link, figures |
| root | `run.py`, `smoke.py` | CLI (`train_base`, `train_classifier`, `train_oracle`, `sweep`, `llm_finetune`, `llm_oracle`, `llm_sweep`, `analyze`, `smoke`) |

**What is NOT implemented in this code drop:** the MIA axis (`src/audit/mia.py`) is an interface + stub. Loss-based, SecMI-style, offline-LiRA and class-geometry MIA are specified in the docstring and must be written before H1 can be evaluated; until then the audit reports NaN for MIA and H1 is "inconclusive". Everything else (influence/H2, spectrum/H3, relearn, probe) is complete.

**Numeric cores are unit-tested** (`tests/test_audit_cores.py`, numpy/sklearn only): schedule math, DDIM x0 recovery, AUC, matrix spectrum (rank-1 vs full), TRAK projection + scores, attribution drop, steps-to-90%, probe on separable features, the H3 norm-controlled partial Spearman, CIFAR split disjointness, do-nothing control.
