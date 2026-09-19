"""Tests for the training-loop augmentation/physics wiring (ROADMAP section 3.4)."""
import numpy as np
import torch

from src.train import _load_physics, _make_augment


def test_disabled_returns_none():
    assert _make_augment(False, 0) is None


def test_augment_index_keyed_reproducible_and_preserves_density():
    v = np.zeros((8, 8, 8), dtype=np.float32)
    v[2:6, 2:6, 2:6] = 1.0
    t1 = _make_augment(True, seed=123)
    a = t1(v, idx=7)
    assert a.shape == v.shape
    assert abs(float(a.sum()) - float(v.sum())) < 1e-9  # exact voxel permutation
    # same seed + same sample index -> same rotation, even across workers
    t2 = _make_augment(True, seed=123)
    assert np.array_equal(t2(v, idx=7), a)


def test_augment_seed_shifts_map_to_different_rotations():
    # A single off-centre voxel has no rotational symmetry: its 24-rotation
    # orbit is 24 distinct grid positions (a rotation fixing it would pin the
    # whole cube). Different seed+index keys must not collapse to one image.
    v = np.zeros((8, 8, 8), dtype=np.float32)
    v[1, 2, 3] = 1.0
    images = {tuple(np.argwhere(_make_augment(True, s)(v, 0))[0])
              for s in range(24)}
    assert len(images) > 1


def test_physics_zero_lambda_is_identity_zero():
    f = _load_physics(0.0, torch.device("cpu"), iterations=60)
    x = torch.ones(1, 1, 4, 4, 4)
    c = torch.zeros(1, 3)
    t = torch.zeros(1, 1)
    out = f(x, c, t)
    assert out.shape == torch.Size([]) and float(out) == 0.0


def test_physics_positive_lambda_returns_finite_loss():
    f = _load_physics(1.0, torch.device("cpu"), iterations=60)
    x = torch.zeros(1, 1, 4, 4, 4)
    x[0, 0, 1:3, 1:3, 1:3] = 1.0
    c = torch.zeros(1, 3)
    t = torch.zeros(1, 1)
    out = f(x, c, t)
    assert out.shape == torch.Size([]) and torch.isfinite(out)
    # bounded by construction (normalized deficit in [0, 1])
    assert 0.0 <= float(out) <= 1.0