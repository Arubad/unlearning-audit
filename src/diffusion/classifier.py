"""Small CIFAR-10 classifier (ResNet-style, ~1M params, ~92% in a few minutes on a T4).

Used for (a) the forget-class *generation rate* of sampled images and (b) a frozen
feature extractor for the class-geometry MIA when the UNet's own features are
not wanted.  ``features(x)`` returns the penultimate activations.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.kaggle_env import results_root, torch_device
from ..common.seed import derive_seed, seed_everything
from .data import load_cifar, make_loader


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)) if (stride != 1 or cin != cout) else nn.Identity()

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))


class SmallResNet(nn.Module):
    def __init__(self, n_classes: int = 10, w: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.layers = nn.Sequential(Block(w, w), Block(w, 2 * w, 2), Block(2 * w, 2 * w), Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w), Block(4 * w, 8 * w, 2))
        self.fc = nn.Linear(8 * w, n_classes)

    def features(self, x):
        return F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


def classifier_path() -> str:
    d = os.path.join(results_root(), "checkpoints")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "cifar10_classifier.pt")


def train_classifier(cfg: Dict, epochs: int = 15, seed: int = 0) -> Dict:
    seed_everything(derive_seed(seed, "clf"))
    device = torch_device()
    ds = load_cifar(cfg["data"].get("root", "data"), train=True)
    test = load_cifar(cfg["data"].get("root", "data"), train=False)
    loader = make_loader(ds, None, 256, True, seed, 2)
    tloader = make_loader(test, None, 512, False, seed, 2, drop_last=False)
    model = SmallResNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=epochs * len(loader))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if torch.rand(1).item() < 0.5:
                x = x.flip(3)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate_classifier(model, tloader, device)
        print(f"epoch {ep+1}: test acc {acc:.4f}")
    torch.save({"model": model.state_dict(), "test_acc": acc}, classifier_path())
    return {"checkpoint": classifier_path(), "test_acc": acc, "train_gpu_hours": (time.time() - t0) / 3600}


@torch.no_grad()
def evaluate_classifier(model, loader, device) -> float:
    model.eval()
    correct = n = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        n += len(y)
    return correct / max(n, 1)


def load_classifier(device=None) -> SmallResNet:
    model = SmallResNet()
    model.load_state_dict(torch.load(classifier_path(), map_location="cpu")["model"])
    return model.to(device or torch_device()).eval()
