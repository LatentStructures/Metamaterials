"""Model + diffusion tests: config-driven shapes, loss, DDIM sampling."""
import pytest
import torch

from src.models.diffusion import GaussianDiffusion, NoiseSchedule
from src.models.unet3d import SelfAttention3D, UNet3D

CONFIG = {
    "in_channels": 1,
    "channels": [32, 64, 128, 256],
    "out_channels": 1,
    "time_embed_dim": 256,
    "conditioning_dim": 3,
    "conditioning_embed_dim": 256,
    "attention_resolutions": [4],
}


def _unet():
    return UNet3D.from_config(CONFIG)


def test_unet_forward_shapes():
    model = _unet()
    x = torch.randn(2, 1, 32, 32, 32)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 3)
    out = model(x, t, c)
    assert out.shape == x.shape


def test_unet_config_has_locked_architecture():
    model = _unet()
    assert model.channels == [32, 64, 128, 256]
    assert 4 in model.attn_resolutions
    p = sum(p.numel() for p in model.parameters())
    assert abs(p / 1e6 - 12.15) < 0.02
    assert sum(isinstance(m, SelfAttention3D) for m in model.modules()) == 3


def test_noise_schedule_edges():
    s = NoiseSchedule.linear(T=1000)
    assert s.betas[0] == 1e-4 and torch.isclose(s.betas[-1], torch.tensor(0.02))
    assert s.alpha_bar[0] < 1.0 and s.alpha_bar[-1] > 1e-6


def test_q_sample_stats():
    d = GaussianDiffusion(T=1000)
    x0 = torch.randn(1000, 1, 4, 4, 4)
    t = torch.full((1000,), 500, dtype=torch.long)
    xt = d.q_sample(x0, t)
    # var(xt) ~ 1 regardless of t (variance-preserving schedule)
    assert 0.5 < xt.var().item() < 2.0


def test_p_losses_returns_scalar_and_decreases():
    torch.manual_seed(0)
    small = {**CONFIG, "channels": [8, 16, 32, 64], "time_embed_dim": 64,
             "conditioning_embed_dim": 64, "attention_resolutions": []}
    model = UNet3D.from_config(small)
    d = GaussianDiffusion(T=1000)
    x = torch.randn(4, 1, 16, 16, 16)
    c = torch.randn(4, 3)
    loss = d.p_losses(model, x, c)
    assert loss.ndim == 0
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = loss.item()
    for _ in range(60):
        opt.zero_grad()
        step_loss = d.p_losses(model, x, c)
        step_loss.backward()
        opt.step()
    assert step_loss.item() < first, "training did not reduce eps-MSE"


def test_p_losses_auxiliary_path_matches_scalar_path():
    torch.manual_seed(31)
    small = {**CONFIG, "channels": [8, 16, 32, 64], "time_embed_dim": 64,
             "conditioning_embed_dim": 64, "attention_resolutions": []}
    model = UNet3D.from_config(small)
    diffusion = GaussianDiffusion(T=1000)
    x = torch.rand(2, 1, 8, 8, 8)
    cond = torch.randn(2, 3)
    torch.manual_seed(7)
    scalar_loss = diffusion.p_losses(model, x, cond)
    torch.manual_seed(7)
    aux_loss, xt, eps, t, noise = diffusion.p_losses(model, x, cond,
                                                      return_aux=True)
    torch.testing.assert_close(aux_loss, scalar_loss, rtol=0.0, atol=0.0)
    assert aux_loss.ndim == 0
    assert xt.shape == x.shape and eps.shape == noise.shape == x.shape
    assert t.shape == (2,) and torch.isfinite(aux_loss)


def test_ddim_rejects_invalid_steps():
    small = {**CONFIG, "channels": [8, 16, 32, 64], "time_embed_dim": 64,
             "conditioning_embed_dim": 64, "attention_resolutions": []}
    d = GaussianDiffusion(T=100)
    model = UNet3D.from_config(small)
    c = torch.randn(1, 3)
    for bad in (0, 101):
        with pytest.raises(ValueError):
            d.sample_ddim(model, c, (1, 1, 8, 8, 8), steps=bad)


def test_ddim_samples_shape_and_reproducible():
    small = {**CONFIG, "channels": [8, 16, 32, 64], "time_embed_dim": 64,
             "conditioning_embed_dim": 64, "attention_resolutions": []}
    model = UNet3D.from_config(small)
    d = GaussianDiffusion(T=1000)
    c = torch.randn(2, 3)
    shape = (2, 1, 16, 16, 16)
    a = d.sample_ddim(model, c, shape, steps=2, seed=7)
    b = d.sample_ddim(model, c, shape, steps=2, seed=7)
    assert a.shape == shape and torch.equal(a, b)
    # noise-conditioned sampling is non-deterministic without a seed
    c_rnd = d.sample_ddim(model, c, shape, steps=2)
    assert c_rnd.shape == shape


def test_ddim_performs_terminal_denoise_to_x0():
    class ZeroEpsilon(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []

        def forward(self, x, t, cond):
            self.seen.extend(t.tolist())
            return torch.zeros_like(x)

    diffusion = GaussianDiffusion(T=8)
    model = ZeroEpsilon()
    cond = torch.zeros(1, 3)
    seed = 13
    gen = torch.Generator().manual_seed(seed)
    initial_noise = torch.randn((1, 1, 2, 2, 2), generator=gen)
    expected = initial_noise / diffusion.schedule.alpha_bar[-1].sqrt()
    result = diffusion.sample_ddim(model, cond, tuple(initial_noise.shape),
                                   steps=1, clamp=False, seed=seed)
    assert model.seen == [7]
    assert torch.allclose(result, expected)


def test_stochastic_ddim_is_reproducible_with_seed():
    class ZeroEpsilon(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, t, cond):
            return torch.zeros_like(x)

    diffusion = GaussianDiffusion(T=8)
    model = ZeroEpsilon()
    cond = torch.zeros(1, 3)
    shape = (1, 1, 2, 2, 2)
    first = diffusion.sample_ddim(model, cond, shape, steps=4, eta=0.5, seed=29)
    second = diffusion.sample_ddim(model, cond, shape, steps=4, eta=0.5, seed=29)
    assert torch.equal(first, second)


def test_ddim_supports_explicit_clamp_range_and_initial_noise():
    class ZeroEpsilon(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, t, cond):
            return torch.zeros_like(x)

    diffusion = GaussianDiffusion(T=8)
    initial = torch.linspace(-2.0, 2.0, 8).reshape(1, 1, 2, 2, 2)
    result = diffusion.sample_ddim(ZeroEpsilon(), torch.zeros(1, 3),
                                   tuple(initial.shape), steps=1,
                                   clamp=(0.0, 1.0), x_init=initial)
    assert result.min() == 0.0 and result.max() == 1.0


def test_ddim_batched_targets_reuse_per_target_initial_noise():
    torch.manual_seed(9)
    small = {**CONFIG, "channels": [8, 16, 32, 64], "time_embed_dim": 64,
             "conditioning_embed_dim": 64, "attention_resolutions": []}
    model = UNet3D.from_config(small).eval()
    diffusion = GaussianDiffusion(T=16)
    cond = torch.randn(2, 3)
    noise_blocks = []
    for seed in (21, 22):
        noise_blocks.append(torch.randn(
            (1, 1, 8, 8, 8), generator=torch.Generator().manual_seed(seed)))
    x_init = torch.cat(noise_blocks)
    batched = diffusion.sample_ddim(model, cond, tuple(x_init.shape), steps=4,
                                    x_init=x_init, clamp=None)
    separate = torch.cat([
        diffusion.sample_ddim(model, cond[i:i + 1], (1, 1, 8, 8, 8), steps=4,
                              x_init=x_init[i:i + 1], clamp=None)
        for i in range(2)
    ])
    torch.testing.assert_close(batched, separate, atol=1e-6, rtol=1e-6)
