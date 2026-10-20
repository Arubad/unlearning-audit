"""CIFAR-10 splits: train / forget / retain / held-out, all seeded and index-based.

The *forget set* is a seeded subset of the forget class whose size is
``forget_frac`` of the whole training set (1%, 5%, 10% = 500, 2500, 5000 of the
5000 cat images; 10% is the entire class).  The *retain set* is everything else.
The *held-out* set for MIA is the CIFAR-10 test split (never trained on).
Indices are saved so every checkpoint sees identical splits.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


def cifar_split_indices(labels: np.ndarray, forget_class: int, forget_frac: float, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    cls_idx = np.where(labels == forget_class)[0]
    n_forget = int(round(forget_frac * len(labels)))
    n_forget = min(n_forget, len(cls_idx))
    forget = np.sort(rng.choice(cls_idx, size=n_forget, replace=False))
    mask = np.ones(len(labels), dtype=bool)
    mask[forget] = False
    retain = np.where(mask)[0]
    return {"forget": forget.tolist(), "retain": retain.tolist(), "n_forget": int(n_forget), "n_retain": int(len(retain))}


def load_cifar(root: str, train: bool = True):
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
    return ds


def cifar_labels(ds) -> np.ndarray:
    return np.array(ds.targets, dtype=np.int64)


def make_loader(ds, indices: List[int] | None, batch_size: int, shuffle: bool, seed: int, num_workers: int = 2, drop_last: bool = True):
    import torch
    from torch.utils.data import DataLoader, Subset

    sub = Subset(ds, indices) if indices is not None else ds
    g = torch.Generator().manual_seed(seed)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last and len(sub) >= batch_size, generator=g, pin_memory=True)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def save_split(split: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)


def load_split(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
