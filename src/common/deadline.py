"""Wall-clock budget for hosted runs (Kaggle kills a session at 12 h without saving).

``UA_DEADLINE`` (unix seconds) is set by the runner.  Long loops call ``past(margin)`` and, when it is true, save
their state and raise ``DeadlineReached``; ``src.pipeline`` catches it and exits cleanly so the session's output
(the resumable checkpoints) is kept.  Unset -> no deadline.
"""
from __future__ import annotations

import os
import time


class DeadlineReached(Exception):
    pass


def deadline() -> float:
    v = os.environ.get("UA_DEADLINE")
    return float(v) if v else float("inf")


def past(margin_s: float = 0.0) -> bool:
    return time.time() + margin_s >= deadline()


def check(margin_s: float = 0.0, what: str = "") -> None:
    if past(margin_s):
        raise DeadlineReached(f"deadline reached before {what}" if what else "deadline reached")
