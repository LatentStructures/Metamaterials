"""Tests for the training-loop augmentation/physics wiring (ROADMAP section 3.4)."""
import gc
import sys

import numpy as np
import torch
import yaml

from src.data.dataset import ConditioningStats, write_conditioned_hdf5
from src.train import (
    EMA,
    _apply_rotation_indices,
    _augment_batch,
    _clip_grad,
    _ema_checkpoint_fields,
    _load_physics,
    _make_augment,
    _resolve_physics_lambda,
    _restore_rng_state,
    _rng_state,
)
from src.train import (
    main as train_main,
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


def test_batch_rotation_is_exact_and_epoch_keyed():
    from src.data.augment import _ROTATIONS, rotate_voxels

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        v = torch.zeros(24, 1, 8, 8, 8, device=device)
        v[:, 0, 1, 2, 3] = 1
        got = _apply_rotation_indices(v, torch.arange(24, device=device))
        for idx, rotation in enumerate(_ROTATIONS):
            expected = rotate_voxels(v[0, 0].cpu().numpy(), rotation)
            assert np.array_equal(got[idx, 0].cpu().numpy(), expected)
        assert torch.equal(got.sum(dim=(1, 2, 3, 4)), v.sum(dim=(1, 2, 3, 4)))
        batch = v[:1].repeat(24, 1, 1, 1, 1)
        a = _augment_batch(batch, seed=12, epoch=3, step=5)
        assert torch.equal(a, _augment_batch(batch, seed=12, epoch=3, step=5))
        assert not torch.equal(a, _augment_batch(batch, seed=12, epoch=4, step=5))


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


def test_multi_decay_ema_matches_warmup_reference_and_checkpoint_format():
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.9, decays=[0.5, 0.9])
    refs = {d: {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            for d in (0.5, 0.9)}
    for n in range(1, 16):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * n)
        ema.update(model)
        for decay, shadow in refs.items():
            effective = min(decay, (1.0 + n) / (10.0 + n))
            for key, value in model.state_dict().items():
                shadow[key].mul_(effective).add_(value.float(), alpha=1.0 - effective)
    for decay, shadow in refs.items():
        for key, value in shadow.items():
            torch.testing.assert_close(ema.shadows[ema._key(decay)][key], value)
    fields = _ema_checkpoint_fields(ema)
    assert set(fields["ema_shadows"]) == {"0.5", "0.9"}
    assert fields["ema_updates"] == 15


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


def _tiny_h5(tmp_path):
    """Small in-memory dataset (8 random 8³ cells; the U-Net is size-agnostic)."""
    rng = np.random.default_rng(7)
    n = 8
    voxels = rng.integers(0, 2, size=(n, 8, 8, 8)).astype(np.uint8)
    cond_raw = np.column_stack([rng.uniform(0.1, 0.6, n),
                                rng.uniform(0.1, 0.6, n), np.full(n, 0.3)])
    stats = ConditioningStats(mean=cond_raw.mean(axis=0),
                              std=cond_raw.std(axis=0) + 1e-8)
    cond_norm = ((cond_raw - stats.mean) / stats.std).astype(np.float32)
    path = tmp_path / "ds.h5"
    write_conditioned_hdf5(path, voxels, cond_norm, stats=stats)
    return path


def _run_tiny_train(tmp_path, monkeypatch, extra_args=()):
    """Run main() end-to-end on a tiny config; return logged scalar histories."""
    train_cfg = {
        "seed": 0,
        "data": {"hdf5": str(_tiny_h5(tmp_path)), "augmentation": False,
                 "conditioning_order": ["E", "relative_density", "nu"]},
        "optimizer": {"lr": 1e-4, "grad_clip": 1.0},
        "batch_size": 4,
        "epochs": 1,
        "checkpoint_every": 1000,
        "physics_loss": {"option": "B", "lambda_1": 0.05, "iterations": 2,
                         "warmup_fraction": 0.2,
                         "ablation": {"lambda_1": 0.0, "tag": "abl_physics"}},
        "logging": {"backend": "tensorboard", "project": "test",
                    "sample_interval": 1000},
    }
    model_cfg = {
        "unet3d": {"in_channels": 1, "channels": [4, 8, 16, 32],
                   "out_channels": 1, "time_embed_dim": 16,
                   "conditioning_dim": 3, "conditioning_embed_dim": 16,
                   "attention_heads": 2, "attention_resolutions": [4]},
        # T >= the 50-step end-of-run DDIM render (sample_ddim validates steps <= T).
        "diffusion": {"T": 1000, "beta_schedule": "linear"},
        "sampling": {"schedule": "ddim", "steps": 50, "eta": 0.0},
    }
    tc = tmp_path / "train_tiny.yaml"
    mc = tmp_path / "model_tiny.yaml"
    tc.write_text(yaml.safe_dump(train_cfg))
    mc.write_text(yaml.safe_dump(model_cfg))
    run_dir = tmp_path / "run"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU-only test
    monkeypatch.setattr(sys, "argv", [
        "src.train", "--train-config", str(tc), "--model-config", str(mc),
        "--run-dir", str(run_dir), "--overfit", "4", "--max-iterations", "51",
        *extra_args])
    train_main()
    gc.collect()  # flush the SummaryWriter the finished main() still holds
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = next(run_dir.glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(events))
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)}
            for t in ("train/loss", "train/loss_diffusion", "train/loss_physics")}


def test_logs_loss_components_summing_to_total(tmp_path, monkeypatch):
    # Regression for the train/loss logging gap: train/loss used
    # to conflate the ε-MSE term with λ·physics, so neither component could be
    # inspected post-hoc. The two tags must exist and sum to train/loss
    # (the physics tag is the weighted, warm-up-adjusted contribution).
    logs = _run_tiny_train(tmp_path, monkeypatch)
    assert set(logs) == {"train/loss", "train/loss_diffusion", "train/loss_physics"}
    assert set(logs["train/loss"]) == {1, 50}
    for step, total in logs["train/loss"].items():
        diff = logs["train/loss_diffusion"][step]
        phys = logs["train/loss_physics"][step]
        assert abs(diff + phys - total) <= 1e-5 * (1.0 + abs(total))
    # λ₁ warm-up: no contribution at step 1; post-warm-up the term is finite
    # and non-negative (the normalized connectivity deficit is in [0, 1]).
    assert logs["train/loss_physics"][1] == 0.0
    assert logs["train/loss_physics"][50] >= 0.0


def test_ablation_logs_zero_physics_component(tmp_path, monkeypatch):
    # The λ₁=0 control must report an exactly-zero physics tag and a total
    # identical to the diffusion term (the λ₁=0 control
    # reports zero physics loss).
    logs = _run_tiny_train(tmp_path, monkeypatch, extra_args=("--physics-ablation",))
    for step, total in logs["train/loss"].items():
        assert logs["train/loss_physics"][step] == 0.0
        assert logs["train/loss_diffusion"][step] == total
