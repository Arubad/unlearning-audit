"""Paths and device handling that differ between Kaggle, Colab and a local machine.

On Kaggle, ``/kaggle/working`` is the only writable, persisted directory (20 GB).
Results and caches go there so they survive the session and appear as the
notebook's output.  Attach a previous notebook's output as a *Dataset* input and
point ``PREV_RESULTS`` at it to reuse caches (pairs parquet, adapters).
"""
from __future__ import annotations

import os
import shutil
from typing import Optional


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def working_dir() -> str:
    if is_kaggle():
        return "/kaggle/working"
    if is_colab():
        return "/content"
    return os.getcwd()


def results_root(default: str = "results") -> str:
    """Where ``results/`` lives: under /kaggle/working on Kaggle, else repo-relative."""
    env = os.environ.get("RESULTS_ROOT")
    if env:
        return env
    if is_kaggle():
        return os.path.join("/kaggle/working", default)
    return default


def hf_cache_dir() -> Optional[str]:
    if is_kaggle():
        d = "/kaggle/working/hf_cache"
        os.makedirs(d, exist_ok=True)
        os.environ.setdefault("HF_HOME", d)
        return d
    return None


def import_previous_results(prev_dir: Optional[str] = None) -> Optional[str]:
    """Copy caches from an attached Kaggle dataset (``/kaggle/input/<name>/results``) into results_root."""
    prev = prev_dir or os.environ.get("PREV_RESULTS")
    if not prev or not os.path.isdir(prev):
        return None
    dst = results_root()
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(prev):
        s, d = os.path.join(prev, name), os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    return dst


def torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preferred_dtype():
    """bf16 where supported (A100/L4), else fp16 on GPU, fp32 on CPU.

    T4 (sm75) has no bf16.  Qwen2.5 in fp16 can overflow in rare cases; the
    0.5B model is small enough to run in fp32, which the configs select.
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_dtype(name: str):
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        return preferred_dtype()
    return {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[name]
