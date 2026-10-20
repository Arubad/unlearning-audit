"""unlearning-audit: does approximate unlearning erase forgotten data or only suppress it?

Diffusion track (class-conditional DDPM on CIFAR-10) and LLM track (Qwen2.5-0.5B on TOFU),
one audit library (MIA, influence, relearning, probe, Delta-theta spectrum) applied to any
original/unlearned checkpoint pair.  Entry point: python -m src.run
"""

__version__ = "0.2.0-code"
