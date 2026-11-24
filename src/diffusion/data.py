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


HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def _npz_path(root: str, train: bool) -> str:
    return os.path.join(root, f"cifar10_{'train' if train else 'test'}.npz")


def _fetch_hf(root: str, train: bool) -> str:
    """Download the HF parquet mirror of CIFAR-10 and cache it as uint8 arrays (the Toronto server serves ~50 kB/s
    to Colab).  NOTE: the mirror's image order differs from the official python batches (first train labels are
    0,6,0,2,7,... vs 6,9,9,4,1,...), so split indices refer to the uoft-cs/cifar10 order.  Every run of this
    pipeline uses the same source, so indices are consistent; mixing sources is not supported."""
    import io
    import urllib.request

    import pandas as pd
    from PIL import Image

    os.makedirs(root, exist_ok=True)
    split = "train" if train else "test"
    pq = os.path.join(root, f"cifar10_{split}.parquet")
    if not os.path.exists(pq):
        urllib.request.urlretrieve(HF_URL.format(split=split), f"{pq}.{os.getpid()}.tmp")
        os.replace(f"{pq}.{os.getpid()}.tmp", pq)
    df = pd.read_parquet(pq)
    icol = "img" if "img" in df else df.columns[0]
    lcol = "label" if "label" in df else df.columns[1]
    x = np.stack([np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB")) for v in df[icol]]).astype(np.uint8)
    y = df[lcol].to_numpy().astype(np.int64)
    out = _npz_path(root, train)
    with open(f"{out}.{os.getpid()}.tmp", "wb") as f:
        np.savez(f, x=x, y=y)
    os.replace(f"{out}.{os.getpid()}.tmp", out)
    return out


class CIFARArrays:
    """Minimal CIFAR-10 dataset over cached uint8 arrays; same interface the code uses (``targets``, __getitem__)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform):
        self.data, self.targets, self.transform = x, y.tolist(), transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        from PIL import Image

        return self.transform(Image.fromarray(self.data[i])), self.targets[i]


def load_cifar(root: str, train: bool = True):
    import torchvision.transforms as T

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    p = _npz_path(root, train)
    if not os.path.exists(p):
        try:
            p = _fetch_hf(root, train)
        except Exception as e:  # fall back to the official (slow) server
            print(f"HF mirror failed ({e}); falling back to torchvision download")
            import torchvision

            return torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
    z = np.load(p)
    return CIFARArrays(z["x"], z["y"], tf)


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
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(split, f)
    os.replace(tmp, path)  # atomic: concurrent seed processes may write the same (deterministic) split


def load_split(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
