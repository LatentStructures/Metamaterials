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
                 rng: torch.Generator | None = None,
                 return_aux: bool = False) -> torch.Tensor | tuple[
                     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random timestep eps-MSE; optionally return (xt, eps, t, noise)."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device, generator=rng)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, cond)
        loss = F.mse_loss(pred, noise)
        return (loss, xt, pred, t, noise) if return_aux else loss

    def sample_ddim(self, model: nn.Module, cond: torch.Tensor,
                    shape: tuple[int, ...], steps: int = 50,
                    eta: float = 0.0, clamp: tuple[float, float] | bool | None = (-1.0, 1.0),
                    seed: int | None = None,
                    x_init: torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sampling; steps in the DDPM grid."""
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, T={self.T}], got {steps}")
        device = next(model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        if x_init is None:
            x = torch.randn(shape, device=device, generator=gen)
        else:
            if tuple(x_init.shape) != tuple(shape):
                raise ValueError(f"x_init shape {tuple(x_init.shape)} != requested {shape}")
            x = x_init.to(device=device).clone()
        # Use exactly `steps` model evaluations and include the final transition
        # from timestep 0 to clean x0 (alpha_bar_previous = 1).
        step_list = torch.linspace(self.T - 1, 0, steps, device="cpu").round().long().tolist()
        if len(set(step_list)) != steps:
            raise ValueError(f"steps={steps} produced duplicate DDIM timesteps for T={self.T}")
        alpha_bar = self.schedule.alpha_bar.to(device)
        for i, ti in enumerate(step_list):
            t_next = step_list[i + 1] if i + 1 < len(step_list) else -1
            ab_t = alpha_bar[ti]
            ab_prev = alpha_bar[t_next] if t_next >= 0 else torch.ones((), device=device)
            eps = model(x, torch.full((x.shape[0],), ti, device=device, dtype=torch.long), cond)
            x0 = (x - torch.sqrt(1.0 - ab_t)[None, None, None, None] * eps) / \
                 torch.sqrt(ab_t)[None, None, None, None]
            clamp_range = (-1.0, 1.0) if clamp is True else clamp
            if clamp_range is not None and clamp_range is not False:
                x0 = x0.clamp(*clamp_range)
            sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * \
                torch.sqrt(1.0 - ab_t / ab_prev)
            x = torch.sqrt(ab_prev)[None, None, None, None] * x0 + \
                torch.sqrt(1.0 - ab_prev - sigma ** 2)[None, None, None, None] * eps
            # For eta > 0, all non-terminal transitions have positive sigma;
            # the terminal sigma is exactly zero. Avoid reading this device
            # scalar back to Python on every step (a CUDA synchronization).
            # Adding zero times the final noise leaves the terminal result
            # unchanged and consumes no randomness used by a later step.
            if eta > 0.0:
                x = x + sigma * torch.randn(x.shape, device=device, dtype=x.dtype,
                                             generator=gen)
        return x


def build_diffusion(cfg: dict, device: torch.device) -> GaussianDiffusion:
    schedule = cfg.get("beta_schedule", "linear")
    if schedule != "linear":
        raise ValueError(f"unsupported beta_schedule {schedule!r}; baseline uses linear")
    noise_schedule = NoiseSchedule.linear(
        T=int(cfg["T"]),
        beta_start=float(cfg.get("beta_start", 1e-4)),
        beta_end=float(cfg.get("beta_end", 0.02)),
    )
    d = GaussianDiffusion(schedule=noise_schedule)
    d.schedule.to(device)
    return d
