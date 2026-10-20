"""The audit library: every module takes an (original, unlearned) checkpoint pair plus data.

mia.py          loss-based, SecMI-style, offline-LiRA (oracle as OUT) and class-geometry MIA
attribution.py  TRAK-style influence on random-projected per-example gradients
relearn.py      few-shot relearning curve and steps-to-90%
probe.py        logistic-regression probe on frozen mid features
spectrum.py     per-layer SVD of delta theta: effective rank, top-5 energy, norm
report.py       one audit record per checkpoint pair + the "do nothing" control assertions

The numeric cores (``*_np`` functions) are numpy-only and unit-tested; the
torch wrappers only collect losses, features and gradients.
"""
