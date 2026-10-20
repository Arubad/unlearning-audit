"""Compact class-conditional UNet for 32x32 images (5-10M parameters).

Architecture: base channels 64, channel multipliers (1, 2, 2), two residual blocks
per level, one self-attention block at 16x16, sinusoidal time embedding, class
embedding with an extra *null* class (index n_classes) used for classifier-free
guidance (the label is dropped with probability ``cfg_dropout`` during training).

``features(x, t, y)`` returns the mid-block activation (pooled) for linear probes.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        attn = torch.softmax(torch.einsum("bhdn,bhdm->bhnm", q, k) / math.sqrt(c // self.heads), dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, n_classes: int = 10, base: int = 64, mults=(1, 2, 2), n_res: int = 2, attn_res=(16,), image_size: int = 32, dropout: float = 0.1, cfg_dropout: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.null_class = n_classes
        self.cfg_dropout = cfg_dropout
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(n_classes + 1, emb_dim)
        self.base = base
        self.inp = nn.Conv2d(3, base, 3, padding=1)
        chans: List[int] = [base]
        self.down = nn.ModuleList()
        c, res = base, image_size
        for i, m in enumerate(mults):
            for _ in range(n_res):
                blk = nn.ModuleList([ResBlock(c, base * m, emb_dim, dropout)])
                c = base * m
                if res in attn_res:
                    blk.append(AttnBlock(c))
                self.down.append(blk)
                chans.append(c)
            if i != len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                chans.append(c)
                res //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, emb_dim, dropout), AttnBlock(c), ResBlock(c, c, emb_dim, dropout)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(n_res + 1):
                blk = nn.ModuleList([ResBlock(c + chans.pop(), base * m, emb_dim, dropout)])
                c = base * m
                if res in attn_res:
                    blk.append(AttnBlock(c))
                self.up.append(blk)
            if i != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
                res *= 2
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def _embed(self, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base))
        if y is None:
            y = torch.full((t.shape[0],), self.null_class, device=t.device, dtype=torch.long)
        return emb + self.class_emb(y)

    def _encode(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, emb)
                for layer in blk[1:]:
                    h = layer(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)
        return h, hs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self._embed(t, y)
        h, hs = self._encode(x, emb)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = blk[0](torch.cat([h, hs.pop()], dim=1), emb)
                for layer in blk[1:]:
                    h = layer(h)
            else:
                for layer in blk:
                    h = layer(h)
        return self.out(h)

    @torch.no_grad()
    def features(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mid-block activation, spatially averaged -> (B, C).  Used by the linear probe."""
        emb = self._embed(t, y)
        h, _ = self._encode(x, emb)
        return h.mean(dim=(2, 3))


def build_unet(cfg: dict) -> UNet:
    m = cfg["model"]
    return UNet(n_classes=int(m.get("n_classes", 10)), base=int(m.get("base_channels", 64)), mults=tuple(m.get("channel_mults", [1, 2, 2])), n_res=int(m.get("res_blocks", 2)), attn_res=tuple(m.get("attention_resolutions", [16])), image_size=int(m.get("image_size", 32)), dropout=float(m.get("dropout", 0.1)), cfg_dropout=float(m.get("cfg_dropout", 0.1)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
