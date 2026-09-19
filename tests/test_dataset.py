"""Dataset loader/writer tests: schema, normalization, reload round-trip."""
import h5py
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
    _, c = ds[0]
    assert float(c.abs().max()) < 10.0


def test_normalize_denormalize_roundtrip():
    s = ConditioningStats(mean=np.zeros(3, dtype=np.float32),
                          std=np.ones(3, dtype=np.float32))
    v = np.array([1.0, 0.5, -2.0], dtype=np.float32)
    assert np.allclose(s.denormalize(s.normalize(v)), v)


def test_conditioning_order_round_trip(tmp_path):
    """Writer/reader must expose raw [E, rho, nu] in the locked order."""
    rng = np.random.default_rng(2)
    n = 6
    path = tmp_path / "rt.h5"
    voxels = rng.integers(0, 2, size=(n, 8, 8, 8)).astype(np.uint8)
    cond_raw = np.column_stack([
        np.full(n, 2.0), rng.uniform(0.15, 0.5, n), np.full(n, 0.4)])
    stats = ConditioningStats(mean=cond_raw.mean(axis=0) * 0.3,
                              std=cond_raw.std(axis=0) + 1e-8)
    cond_norm = (cond_raw - stats.mean) / stats.std
    write_conditioned_hdf5(path, voxels, cond_norm, stiffness=None, stats=stats)

    with h5py.File(path, "r") as f:
        stored = np.asarray(f["/conditioning_vector"])
        order = [str(s, "utf-8") for s in f["/conditioning_order"]]
    assert order == ["E", "relative_density", "nu"]
    assert stored.shape == (n, 3)
    recovered = stored * stats.std + stats.mean
    assert np.allclose(recovered, cond_raw, atol=1e-6)


def test_writer_rejects_non_cubic_voxels(tmp_path):
    rng = np.random.default_rng(3)
    paths = [tmp_path / f"bad{i}.h5" for i in range(2)]
    voxels = rng.integers(0, 2, size=(4, 8, 8, 9)).astype(np.uint8)
    stats = ConditioningStats(mean=np.zeros(3, dtype=np.float32),
                              std=np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        write_conditioned_hdf5(paths[0], voxels, rng.random((4, 3)).astype(np.float32),
                               stiffness=None, stats=stats)
    bad_cond = rng.random((4, 4)).astype(np.float32)
    voxels_cube = rng.integers(0, 2, size=(4, 8, 8, 8)).astype(np.uint8)
    with pytest.raises(ValueError):
        write_conditioned_hdf5(paths[1], voxels_cube, bad_cond,
                               stiffness=None, stats=stats)


def test_reader_rejects_wrong_conditioning_width(tmp_path):
    path = tmp_path / "x.h5"
    rng = np.random.default_rng(4)
    voxels = rng.integers(0, 2, size=(4, 8, 8, 8)).astype(np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("/voxels", data=voxels, chunks=True,
                         compression="gzip", compression_opts=4)
        f.create_dataset("/conditioning_vector", data=rng.random((4, 4)))
        f["/conditioning_vector"].attrs["name"] = "test"
    with pytest.raises(ValueError):
        read_conditioning_stats(path)