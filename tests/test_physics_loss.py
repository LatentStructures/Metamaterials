"""Tests for the baseline Option-B physics proxy (ROADMAP 4.4)."""
import numpy as np
import torch

from src.losses.physics_loss import connectivity_proxy_loss


def _grid(solid_slices: list[tuple[slice, slice, slice]], n: int = 32) -> torch.Tensor:
    v = np.zeros((1, 1, n, n, n), dtype=np.float32)
    for s in solid_slices:
        v[0, 0, s[0], s[1], s[2]] = 1.0
    return torch.as_tensor(v)


def _solid_cube(n: int = 32) -> torch.Tensor:
    lo, hi = 4, n - 4
    return _grid([(slice(lo, hi), slice(lo, hi), slice(lo, hi))], n)


def _two_isolated_blocks(n: int = 32) -> torch.Tensor:
    s1 = (slice(2, 8), slice(2, 8), slice(2, 8))
    s2 = (slice(n - 8, n - 2), slice(n - 8, n - 2), slice(n - 8, n - 2))
    return _grid([s1, s2], n)


def _soft_bridged_blocks(n: int = 32, bridge_p: float = 0.5) -> torch.Tensor:
    v = _two_isolated_blocks(n).numpy()
    v[0, 0, 8 : n - 8, 4:8, 4:8] = bridge_p
    return torch.as_tensor(v)


def test_solid_cube_small_deficit():
    d = connectivity_proxy_loss(_solid_cube()).item()
    assert 0.0 <= d < 0.35


def test_disconnected_blocks_large_deficit():
    d = connectivity_proxy_loss(_two_isolated_blocks()).item()
    assert d > 0.5


def test_empty_grid_deficit_is_one():
    assert connectivity_proxy_loss(torch.zeros(1, 1, 32, 32, 32)).item() == 1.0


def test_differentiable_in_p():
    x = _soft_bridged_blocks().requires_grad_(True)
    d = connectivity_proxy_loss(x)
    d.backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0
    assert 0.0 < d.item() < 1.0  # strictly in the sensitive regime, not saturating


def test_batched_input_scalar_output():
    b = torch.zeros(4, 1, 16, 16, 16)
    b[:, 0, 2:14, 2:14, 2:14] = 1.0
    out = connectivity_proxy_loss(b)
    assert out.dim() == 0
    assert bool(out == out)  # not NaN


def test_epsilon_parameter_respected():
    d_few = connectivity_proxy_loss(_two_isolated_blocks(), iterations=5)
    d_many = connectivity_proxy_loss(_two_isolated_blocks(), iterations=50)
    assert d_many.item() > d_few.item()  # tighter convergence -> closer to 1