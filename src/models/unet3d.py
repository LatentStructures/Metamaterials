"""3D conditional U-Net (ROADMAP section 4.2).

Config-driven; no hardcoded hyperparameters. Channels [32,64,128,256], three
down/up stages, bottleneck self-attention at 4^3, FiLM-scale conditioning from
the normalized [E, rho, nu] vector (order fixed by the manifest). No
cross-attention, no extra conditioning flows (locked decisions).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer times in [0, T).  Sinusoidal positional encodings.
        half = self.dim // 2
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        w = t.float().unsqueeze(-1) * inv_freq[None]
        return torch.cat([w.sin(), w.cos()], dim=-1)


class ConditioningEmbedding(nn.Module):
    """FiLM conditioning source: cond (B, 3) -> (B, embed_dim)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.mlp(cond)


class FiLMLayer(nn.Module):
    """FiLM scale+shift on a transformed feature map: gamma/beta from the embedding."""

    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.siLU = nn.SiLU()
        self.in_proj = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.film = nn.Linear(embed_dim, channels * 2)
        self.out_proj = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.siLU(h)
        h = self.in_proj(h)
        gamma, beta = self.film(emb).chunk(2, dim=1)
        h = h * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        return self.out_proj(h)


def _attn(q, k, v):
    """Channel-last attention over head-dim; q,k,v: (B*H, S, D)."""
    scale = q.shape[-1] ** -0.5
    att = (q @ k.transpose(-1, -2)) * scale
    att = torch.softmax(att, dim=-1)
    return att @ v


class SelfAttention3D(nn.Module):
    """Self-attention on the (C, D, H, W) spatial dims; no cross-attention."""

    def __init__(self, channels: int, heads: int = 4, norm_groups: int = 32):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.norm = nn.GroupNorm(min(norm_groups, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[0], x.shape[1]
        h = self.norm(x)  # (B, C, D, H, W)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B * self.heads, -1, *qkv.shape[2:])
        q, k, v = qkv.chunk(3, dim=1)
        spatial = q.shape[2:]
        S = math.prod(spatial)
        q = q.reshape(B * self.heads, self.channels // self.heads, S).transpose(1, 2)
        k = k.reshape(B * self.heads, self.channels // self.heads, S).transpose(1, 2)
        v = v.reshape(B * self.heads, self.channels // self.heads, S).transpose(1, 2)
        out = _attn(q, k, v).transpose(1, 2).reshape(B * self.heads, -1, *spatial)
        out = out.reshape(B, C, *spatial)
        return x + self.proj(out)


class ResBlockDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.film1 = FiLMLayer(in_ch, embed_dim)
        self.down = nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1)
        self.film2 = FiLMLayer(out_ch, embed_dim)
        self.shortcut = (nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.film1(x, emb)
        h = self.down(h)
        h = self.film2(h, emb)
        s = self.shortcut(x)
        if s.shape[2:] != h.shape[2:]:
            s = F.interpolate(s, size=h.shape[2:], mode="nearest")
        return (h + s) / math.sqrt(2.0)


class ResBlockUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.film1 = FiLMLayer(out_ch, embed_dim)
        self.film2 = FiLMLayer(out_ch, embed_dim)
        self.shortcut = (nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self.film1(h, emb)
        h = self.film2(h, emb)
        s = self.shortcut(x)
        s = F.interpolate(s, size=h.shape[2:], mode="nearest")
        return (h + s) / math.sqrt(2.0)


class UNet3D(nn.Module):
    """Conditional 3D U-Net: x (B,1,32,32,32), t (B,), cond (B,3) -> predicted eps."""

    def __init__(self,
                 in_channels: int,
                 channels: list[int],
                 out_channels: int,
                 time_embed_dim: int,
                 conditioning_dim: int,
                 conditioning_embed_dim: int,
                 attention_resolutions: list[int] | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.attn_resolutions = set(attention_resolutions or [])
        emb_dim = time_embed_dim
        self.time_embed = SinusoidalTimeEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cond_embed = ConditioningEmbedding(conditioning_dim, conditioning_embed_dim)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(conditioning_embed_dim, emb_dim))

        # contractive stages
        self.conv_in = nn.Conv3d(in_channels, channels[0], 3, padding=1, bias=False)
        self.downs = nn.ModuleList()
        c_in = channels[0]
        # resolution is locked at 32 (ROADMAP); stage i resolves to 32 // 2^i.
        for i, c_out in enumerate(channels[1:], start=1):
            block = ResBlockDown(c_in, c_out, emb_dim)
            self.downs.append(block)
            c_in = c_out
            res = 32 // (2 ** i)
            if res in self.attn_resolutions:
                self.downs.append(SelfAttention3D(c_out))
        # bottleneck self-attention at the lowest resolution (4^3 = 32//8)
        self.bottleneck_attn = SelfAttention3D(c_in)

        # expansive stages (reversed)
        rev = list(reversed(channels))
        self.ups = nn.ModuleList()
        c_in = rev[0]
        for i, c_out in enumerate(rev[1:], start=1):
            res = 32 // (2 ** (len(channels) - i))
            if res in self.attn_resolutions:
                self.ups.append(SelfAttention3D(c_in))
            self.ups.append(ResBlockUp(c_in, c_out, emb_dim))
            c_in = c_out
        self.conv_out = nn.Conv3d(rev[-1], self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te = self.time_mlp(self.time_embed(t))
        ce = self.cond_mlp(self.cond_embed(cond))
        emb = te + ce

        skips = []
        h = self.conv_in(x)
        skips.append(h)
        for mod in self.downs:
            if isinstance(mod, SelfAttention3D):
                h = mod(h)
            else:
                h = mod(h, emb)
                skips.append(h)
        h = self.bottleneck_attn(h)
        for mod in self.ups:
            if isinstance(mod, SelfAttention3D):
                h = mod(h)
            else:
                if skips:
                    h = h + skips.pop()
                h = mod(h, emb)

        return self.conv_out(h)

    @classmethod
    def from_config(cls, unet_cfg: dict) -> "UNet3D":
        channels = [int(c) for c in unet_cfg["channels"]]
        res = [int(r) for r in unet_cfg.get("attention_resolutions", [])]
        return cls(in_channels=int(unet_cfg["in_channels"]),
                   channels=channels,
                   out_channels=int(unet_cfg["in_channels"]),
                   time_embed_dim=int(unet_cfg["time_embed_dim"]),
                   conditioning_dim=int(unet_cfg["conditioning_dim"]),
                   conditioning_embed_dim=int(unet_cfg["conditioning_embed_dim"]),
                   attention_resolutions=res)