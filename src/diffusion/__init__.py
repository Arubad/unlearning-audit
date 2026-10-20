"""Diffusion track: compact class-conditional DDPM on CIFAR-10, unlearning methods, sampling.

unet.py        5-10M parameter class-conditional UNet with a null class (CFG dropout); features() for probes
ddpm.py        linear schedule, q_sample, eps-prediction loss, DDIM-50 sampler with CFG, EMA
train.py       base model (D1) and the retain-only oracle (D2) with the same recipe
classifier.py  small CIFAR-10 classifier: forget-class generation rate + frozen features for CMIA
unlearn.py     gradient_ascent, gradient_difference, label_annealing, variational_ewc, oracle, do_nothing
sample.py      conditional / unconditional sampling, forget-class rate, unconditional drift
"""
