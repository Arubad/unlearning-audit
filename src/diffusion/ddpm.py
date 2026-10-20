"""DDPM forward process, eps-prediction loss, DDIM sampler with classifier-free guidance, EMA.

The schedule arithmetic lives in numpy (``linear_schedule_np``) so the unit tests
can check it without torch; the torch class wraps the same arrays.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np


def linear_schedule_np(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> Dict[str, np.ndarray]:
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return {"betas": betas, "alphas": alphas, "alphas_cumprod": abar, "sqrt_abar": np.sqrt(abar), "sqrt_1m_abar": np.sqrt(1.0 - abar)}


def ddim_step_np(x_t: np.ndarray, eps: np.ndarray, abar_t: float, abar_prev: float, eta: float = 0.0, noise: Optional[np.ndarray] = None) -> np.ndarray:
    """One deterministic (eta=0) DDIM update, numpy reference used by the tests."""
    x0 = (x_t - np.sqrt(1 - abar_t) * eps) / np.sqrt(abar_t)
    sigma = eta * np.sqrt((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev))
    dir_xt = np.sqrt(max(1 - abar_prev - sigma**2, 0.0)) * eps
    x_prev = np.sqrt(abar_prev) * x0 + dir_xt
    if eta > 0 and noise is not None:
        x_prev = x_prev + sigma * noise
    return x_prev


class Diffusion:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device=None):
        import torch

        s = linear_schedule_np(T, beta_start, beta_end)
        self.T = T
        self.device = device or torch.device("cpu")
        self.abar = torch.tensor(s["alphas_cumprod"], dtype=torch.float32, device=self.device)
        self.sqrt_abar = torch.tensor(s["sqrt_abar"], dtype=torch.float32, device=self.device)
        self.sqrt_1m_abar = torch.tensor(s["sqrt_1m_abar"], dtype=torch.float32, device=self.device)

    def q_sample(self, x0, t, noise):
        a = self.sqrt_abar[t][:, None, None, None]
        b = self.sqrt_1m_abar[t][:, None, None, None]
        return a * x0 + b * noise

    def loss(self, model, x0, y, t=None, noise=None, reduction: str = "mean"):
        """eps-prediction MSE.  ``reduction='none'`` gives one loss per example (used by MIA and gradients)."""
        import torch

        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, y)
        per = ((pred - noise) ** 2).flatten(1).mean(1)
        return per if reduction == "none" else per.mean()

    def train_step_labels(self, model, y):
        """Apply CFG label dropout: replace a fraction of labels with the null class."""
        import torch

        if y is None or model.cfg_dropout <= 0:
            return y
        drop = torch.rand(y.shape[0], device=y.device) < model.cfg_dropout
        return torch.where(drop, torch.full_like(y, model.null_class), y)

    def ddim_sample(self, model, n: int, y=None, steps: int = 50, guidance: float = 2.0, eta: float = 0.0, generator=None, image_size: int = 32):
        """DDIM sampling with classifier-free guidance.  ``y=None`` -> unconditional (null class)."""
        import torch

        model.eval()
        x = torch.randn(n, 3, image_size, image_size, device=self.device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        null = torch.full((n,), model.null_class, device=self.device, dtype=torch.long)
        with torch.no_grad():
            for i, t in enumerate(ts):
                tb = torch.full((n,), int(t), device=self.device, dtype=torch.long)
                if y is None or guidance == 1.0 or guidance == 0.0:
                    eps = model(x, tb, y if y is not None else null)
                else:
                    e_c = model(x, tb, y)
                    e_u = model(x, tb, null)
                    eps = e_u + guidance * (e_c - e_u)
                abar_t = self.abar[t]
                abar_prev = self.abar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=self.device)
                x0 = (x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
                x0 = x0.clamp(-1, 1)
                sigma = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev))
                dir_xt = torch.sqrt((1 - abar_prev - sigma**2).clamp(min=0)) * eps
                x = torch.sqrt(abar_prev) * x0 + dir_xt
                if eta > 0:
                    x = x + sigma * torch.randn(x.shape, device=self.device, generator=generator)
        return x.clamp(-1, 1)


class EMA:
    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        import torch

        with torch.no_grad():
            for ps, pm in zip(self.shadow.parameters(), model.parameters()):
                ps.mul_(self.decay).add_(pm.detach(), alpha=1 - self.decay)
            for bs, bm in zip(self.shadow.buffers(), model.buffers()):
                bs.copy_(bm)
