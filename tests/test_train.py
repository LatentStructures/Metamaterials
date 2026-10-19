"""Tests for the training-loop augmentation/physics wiring (ROADMAP section 3.4)."""
import numpy as np
import torch

from src.train import (
    EMA,
    _clip_grad,
    _load_physics,
    _make_augment,
    _resolve_physics_lambda,
    _restore_rng_state,
    _rng_state,
)


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


def test_ema_average_parameters_restores_live_model():
    model = torch.nn.Linear(2, 1)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.shadow = {k: torch.full_like(v, -3.0) for k, v in ema.shadow.items()}
    with ema.average_parameters(model):
        assert all(torch.equal(v, torch.full_like(v, -3.0))
                   for v in model.state_dict().values())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in live.items())


def test_rng_checkpoint_round_trip():
    import random

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    state = _rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(1))
    _restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(1))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


class _GpuTensorStub:
    """Stand-in for a checkpoint tensor loaded with map_location='cuda'."""

    def __init__(self, t):
        self._t = t
        self.cpu_called = False

    @property
    def device(self):
        return torch.device("cuda", 0)

    def cpu(self):
        self.cpu_called = True
        return self._t


def test_restore_rng_state_sends_cuda_state_back_to_cpu(monkeypatch):
    # Regression: torch.load(..., map_location=device) moves the stored CUDA RNG
    # state to the GPU, but torch.cuda.set_rng_state_all only accepts CPU
    # ByteTensors, so resume crashed with "RNG state must be a torch.ByteTensor".
    seen = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all",
                        lambda states: seen.update(states=list(states)))
    stub = _GpuTensorStub(torch.zeros(16, dtype=torch.uint8))
    state = _rng_state()
    state["torch_cuda"] = [stub]
    _restore_rng_state(state)
    assert stub.cpu_called
    assert len(seen["states"]) == 1
    only = seen["states"][0]
    assert isinstance(only, torch.Tensor)
    assert only.dtype == torch.uint8 and only.device.type == "cpu"


def test_physics_config_selects_on_run_and_zero_ablation():
    cfg = {"lambda_1": 0.05, "ablation": {"lambda_1": 0.0}}
    assert _resolve_physics_lambda(cfg) == 0.05
    assert _resolve_physics_lambda(cfg, ablation=True) == 0.0


def test_clip_grad_disabled_is_noop():
    m = torch.nn.Linear(4, 4)
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    before = torch.norm(torch.stack([p.grad.norm() for p in m.parameters()]))
    assert _clip_grad(m, None) is None
    assert _clip_grad(m, 0.0) is None
    after = torch.norm(torch.stack([p.grad.norm() for p in m.parameters()]))
    assert float(before) == float(after)  # untouched when disabled


def test_clip_grad_bounds_global_norm_and_returns_preclip():
    m = torch.nn.Linear(4, 4)
    for p in m.parameters():
        p.grad = torch.full_like(p, 3.0)
    total_before = torch.norm(torch.stack([p.grad.norm() for p in m.parameters()]))
    returned = _clip_grad(m, 1.0)
    assert returned is not None and abs(returned - float(total_before)) < 1e-6
    total_after = torch.norm(torch.stack([p.grad.norm() for p in m.parameters()]))
    assert float(total_after) <= 1.0 + 1e-6
