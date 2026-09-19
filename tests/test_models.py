"""Model + diffusion tests: config-driven shapes, loss, DDIM sampling."""
import pytest
import torch

from src.models.diffusion import GaussianDiffusion, NoiseSchedule
from src.models.unet3d import UNet3D

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
    assert 10e6 < p < 30e6


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
        l = d.p_losses(model, x, c)
        l.backward()
        opt.step()
    assert l.item() < first, "training did not reduce eps-MSE"


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