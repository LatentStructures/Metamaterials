"""Dataset loader/writer tests: schema, normalization, reload round-trip."""
import numpy as np
import pytest
import torch

from src.data.dataset import (
    ConditioningStats,
    MetamaterialDataset,
    read_conditioning_stats,
    write_conditioned_hdf5,
)


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory):
    rng = np.random.default_rng(1)
    n = 12
    voxels = rng.integers(0, 2, size=(n, 8, 8, 8)).astype(np.uint8)
    cond_raw = np.column_stack([
        np.full(n, 1.0), rng.uniform(0.1, 0.6, n), np.full(n, 0.3)])
    stiffness = rng.normal(size=(n, 6, 6)).astype(np.float32)
    stats = ConditioningStats(mean=cond_raw.mean(axis=0), std=cond_raw.std(axis=0) + 1e-8)
    path = tmp_path_factory.mktemp("data") / "ds.h5"
    cond_norm = (cond_raw - stats.mean) / stats.std
    write_conditioned_hdf5(path, voxels, cond_norm, stiffness=stiffness, stats=stats)
    return path


def test_shapes_and_dtype(h5_path):
    ds = MetamaterialDataset(h5_path, return_stiffness=True)
    assert len(ds) == 12
    x, c, C = ds[0]
    assert x.shape == torch.Size([1, 8, 8, 8]) and x.dtype == torch.float32
    assert c.shape == torch.Size([3])
    assert C.shape == torch.Size([6, 6])


def test_voxel_values_are_binary_01(h5_path):
    ds = MetamaterialDataset(h5_path)
    x, _ = ds[0]
    assert set(torch.unique(x).tolist()) <= {0.0, 1.0}


def test_normalization_stats_round_trip(h5_path):
    ds = MetamaterialDataset(h5_path)
    st = ds.conditioning_mean_std()
    st2 = read_conditioning_stats(h5_path)
    assert np.allclose(st.mean, st2.mean) and np.allclose(st.std, st2.std)
    # cond values written were already normalized to ~N(0,1)
    x, c = ds[0]
    assert float(c.abs().max()) < 10.0


def test_normalize_denormalize_roundtrip():
    s = ConditioningStats(mean=np.zeros(3, dtype=np.float32),
                          std=np.ones(3, dtype=np.float32))
    v = np.array([1.0, 0.5, -2.0], dtype=np.float32)
    assert np.allclose(s.denormalize(s.normalize(v)), v)