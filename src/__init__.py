"""unlearning-audit — Did it really forget? Auditing diffusion and LLM unlearning.

Planned modules (see src/README.md):
    models    — small class-conditional DDPM UNet for CIFAR-10; wrappers around OpenUnlearning checkpoints (Qwen2.5-0.5B)
    unlearn   — gradient ascent, gradient difference, label annealing, variational, NPO, oracle retrain
    audit     — mia, attribution, relearn, probe, spectrum (+ update norm): each accepts any (original, unlearned) checkpoint pair
    analysis  — influence-vs-relearning and rank-vs-relearning (norm-controlled) correlations, f-INE-style null (stretch), figures
    reproduce — one-command reproduction of Figure 1
"""

__version__ = "0.0.0-prereg"
