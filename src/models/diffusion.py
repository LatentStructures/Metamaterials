"""DDPM core + DDIM sampler (ROADMAP section 4.3).

Standard ε-MSE (L_simple) training with T=1000 linear betas; DDIM sampling at
50-100 inference steps with eta=0 (deterministic). No classifier guidance, no
timestamp-conditional curriculum beyond FiLM -- this is the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NoiseSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor

    @classmethod
    def linear(cls, T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> NoiseSchedule:
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        return cls(T=T, betas=betas, alphas=alphas, alpha_bar=alpha_bar)

    def to(self, device: torch.device) -> NoiseSchedule:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


class GaussianDiffusion:
    def __init__(self, schedule: NoiseSchedule | None = None, T: int = 1000):
        self.schedule = schedule or NoiseSchedule.linear(T=T)
        self.T = self.schedule.T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
                 ) -> torch.Tensor:
        """x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps."""
        noise = torch.randn_like(x0) if noise is None else noise
        ab = self.schedule.alpha_bar.to(x0.device)[t]
        return torch.sqrt(ab[:, None, None, None, None]) * x0 + \
            torch.sqrt(1.0 - ab[:, None, None, None, None]) * noise

    def predict_denoised(self, xt: torch.Tensor, t: torch.Tensor,
                         eps_pred: torch.Tensor, clamp: tuple[float, float] | None = None
                         ) -> torch.Tensor:
        """Invert q_sample: x0 = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t)."""
        ab = self.schedule.alpha_bar.to(xt.device)[t]
        x0 = (xt - torch.sqrt(1.0 - ab)[:, None, None, None, None] * eps_pred) / \
            torch.sqrt(ab)[:, None, None, None, None]
        if clamp is not None:
            x0 = x0.clamp(*clamp)
        return x0

    def p_losses(self, model: nn.Module, x0: torch.Tensor, cond: torch.Tensor,
                 rng: torch.Generator | None = None) -> torch.Tensor:
        """Random timestep eps-MSE (L_simple) -- the only training loss."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device, generator=rng)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, cond)
        return F.mse_loss(pred, noise)

    def sample_ddim(self, model: nn.Module, cond: torch.Tensor,
                    shape: tuple[int, ...], steps: int = 50,
                    eta: float = 0.0, clamp: bool = True,
                    seed: int | None = None) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sampling; steps in the DDPM grid."""
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, T={self.T}], got {steps}")
        device = next(model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        x = torch.randn(shape, device=device, generator=gen)
        ab = torch.cat([self.schedule.alpha_bar, torch.ones(1, device=device)])
        step_list = list(range(self.T - 1, -1, -(self.T // steps)))
        if step_list[-1] != 0:
            step_list.append(0)
        for i in range(len(step_list) - 1):
            ti = step_list[i]
            t_next = step_list[i + 1]
            ab_t = ab[ti]
            ab_prev = ab[t_next]
            eps = model(x, torch.full((x.shape[0],), ti, device=device, dtype=torch.long), cond)
            x0 = (x - torch.sqrt(1.0 - ab_t)[None, None, None, None] * eps) / \
                 torch.sqrt(ab_t)[None, None, None, None]
            if clamp:
                x0 = x0.clamp(-1.0, 1.0)
            sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * \
                torch.sqrt(1.0 - ab_t / ab_prev)
            x = torch.sqrt(ab_prev)[None, None, None, None] * x0 + \
                torch.sqrt(1.0 - ab_prev - sigma ** 2)[None, None, None, None] * eps
            if sigma > 0:
                x = x + sigma * torch.randn_like(x, generator=gen)
        return x


def build_diffusion(cfg: dict, device: torch.device) -> GaussianDiffusion:
    d = GaussianDiffusion(T=int(cfg["T"]))
    d.schedule.to(device)
    return d