"""Dataset-generation tests: stratified splits and checkpoint/resume semantics.

Exercises ``generate_dataset.generate`` end-to-end with the mock homogenizer
(no dolfinx), including the resume-after-FEA-failure path, which previously
reprocessed already-completed cells and never retried the failed ones.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import src.data.generate_dataset as gen


class _FailGuard:
    """Wraps a homogenizer, raising on the first ``fail_first`` calls."""

    def __init__(self, inner, fail_first: int = 0):
        self.inner = inner
        self.fail_left = int(fail_first)
        self.calls = 0

    @property
    def times(self) -> list[float]:
        return self.inner.times

    def __call__(self, voxels, mesh, pairings, res):
        self.calls += 1
        if self.fail_left > 0:
            self.fail_left -= 1
            raise RuntimeError("injected FEA failure")
        return self.inner(voxels, mesh, pairings, res)


def _write_tiny_config(tmp_path: pytest.TempPathFactory) -> str:
    cfg = {
        "seed": 0,
        "resolution": 8,
        "num_cells": 5,
        "families": ["cubic_strut"],
        "base_material": {"E": 1.0, "nu": 0.3},
        "relative_density_range": [0.1, 0.6],
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "hdf5": str(tmp_path / "dataset_{split}.h5"),
        "manifest": str(tmp_path / "manifest.parquet"),
        "checkpoint_every": 7,
        "homogenization": {
            "method": "mock",
            "validation_tolerance": 0.05,
            "void_scale": 1.0e-6,
        },
        "fea_retries": 1,
    }
    path = tmp_path / "tiny.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return str(path)


# --------------------------------------------------------------------------- #
# Stratified splits
# --------------------------------------------------------------------------- #

def test_stratify_splits_leaves_train_nonempty_for_small_buckets():
    samples = [
        {"family": "cubic_strut", "achieved_density": 0.1 + 0.09 * i}
        for i in range(5)
    ]
    splits = gen._stratify_splits(
        samples, {"train": 0.8, "val": 0.1, "test": 0.1}, (0.1, 0.6),
        np.random.default_rng(0))
    lengths = {k: len(v) for k, v in splits.items()}
    assert lengths["train"] >= 1, lengths
    assert sum(lengths.values()) == 5
    # every sample lands in exactly one split
    flattened = [i for k in splits for i in splits[k]]
    assert len(set(flattened)) == 5


def test_stratify_splits_bucket_allocation():
    samples = [{"family": "gyroid", "achieved_density": 0.2} for _ in range(120)]
    splits = gen._stratify_splits(
        samples, {"train": 0.8, "val": 0.1, "test": 0.1}, (0.1, 0.6),
        np.random.default_rng(0))
    assert len(splits["train"]) == 96
    assert len(splits["val"]) == 12
    assert len(splits["test"]) == 12


def test_stratify_keeps_identical_grids_in_one_split():
    unique = [np.full((2, 2, 2), i, dtype=np.uint8) for i in range(9)]
    samples = [{"family": "gyroid", "achieved_density": 0.3, "voxels": grid}
               for grid in unique]
    samples.insert(3, {"family": "gyroid", "achieved_density": 0.3,
                       "voxels": unique[0].copy()})
    splits = gen._stratify_splits(
        samples, {"train": 0.8, "val": 0.1, "test": 0.1}, (0.1, 0.6),
        np.random.default_rng(0))
    assignments = {i: split for split, indexes in splits.items() for i in indexes}
    assert assignments[0] == assignments[3]
    assert gen._check_cross_split_leakage(samples, splits) == 0


def test_unique_test_conditions_requires_distinct_vectors_and_is_seeded():
    records = [{"cond": np.asarray([float(i), 0.25, 0.3], dtype=np.float32)}
               for i in range(4)]
    first = gen._unique_test_conditions(records, [0, 1, 2, 3], 3, seed=9)
    second = gen._unique_test_conditions(records, [0, 1, 2, 3], 3, seed=9)
    assert np.array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 3
    with pytest.raises(RuntimeError, match="distinct conditioning vectors"):
        gen._unique_test_conditions(records, [0, 0, 1], 3, seed=9)


def test_grouped_split_keeps_row_fractions_while_preventing_leakage():
    grids = [np.full((2, 2, 2), i, dtype=np.uint8) for i in range(20)]
    samples = [{"family": "cubic_strut", "achieved_density": 0.3,
                "voxels": grid} for grid in grids]
    samples.extend([
        {"family": "cubic_strut", "achieved_density": 0.3,
         "voxels": grids[0].copy()} for _ in range(3)
    ])
    samples.extend([
        {"family": "cubic_strut", "achieved_density": 0.3,
         "voxels": grids[1].copy()} for _ in range(2)
    ])
    splits = gen._stratify_splits(
        samples, {"train": 0.8, "val": 0.1, "test": 0.1}, (0.1, 0.6),
        np.random.default_rng(4))
    assert gen._check_cross_split_leakage(samples, splits) == 0
    sizes = {k: len(v) / len(samples) for k, v in splits.items()}
    assert 0.65 <= sizes["train"] <= 0.9
    assert 0.05 <= sizes["val"] <= 0.2
    assert 0.05 <= sizes["test"] <= 0.2


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #

def test_generate_resume_skips_done_and_retries_failed(monkeypatch, tmp_path):
    cfg_path = _write_tiny_config(tmp_path)
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)

    run1 = _FailGuard(gen._MockHomogenizer(), fail_first=2)
    monkeypatch.setattr(gen, "_load_homogenizer",
                        lambda *a, **k: (run1, None))
    gen.generate(cfg_path, mock_fea=True)

    ckpt = tmp_path / "data" / "checkpoints" / "ckpt_tiny_s0.parquet"
    assert ckpt.exists()
    assert len(pd.read_parquet(ckpt)) == 3, "two injected failures must be absent"

    run2 = _FailGuard(gen._MockHomogenizer(), fail_first=0)
    monkeypatch.setattr(gen, "_load_homogenizer",
                        lambda *a, **k: (run2, None))
    gen.generate(cfg_path, mock_fea=True)

    assert run2.calls == 2, "resume must redo only the two failed cells"

    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    assert len(manifest) == 5
    assert manifest["cell_id"].is_unique
    assert set(manifest["cell_id"]) == {0, 1, 2, 3, 4}

    # ROADMAP 3.4: HDF5 carries /metadata plus the locked conditioning order,
    # and the manifest carries the achieved E/nu plus an absolute voxel path.
    assert {"E_eff", "nu_eff", "voxel_path"}.issubset(manifest.columns)
    assert str(manifest.iloc[0]["voxel_path"]).startswith(str(tmp_path))
    with h5py.File(tmp_path / "dataset_train.h5", "r") as f:
        assert "/metadata" in f and "/conditioning_order" in f
        meta = f["/metadata"]
        for col in ("family", "target_density", "achieved_density",
                    "E_eff", "nu_eff", "valid"):
            assert col in meta.dtype.names, meta.dtype.names
        order = [str(s, "utf-8") for s in f["/conditioning_order"][...]]
        assert order == ["E", "relative_density", "nu"]


def test_check_cross_split_leakage():
    v1 = np.zeros((4, 4, 4), dtype=np.uint8)
    v2 = np.ones((4, 4, 4), dtype=np.uint8)
    records = [{"voxels": v1}, {"voxels": v2}, {"voxels": v1}]
    # Case 1: distinct grids in train vs test -> 0 leakages
    splits_clean = {"train": [0], "test": [1]}
    assert gen._check_cross_split_leakage(records, splits_clean) == 0

    # Case 2: identical grid in train and test -> 1 leakage
    splits_leaked = {"train": [0], "test": [2]}
    assert gen._check_cross_split_leakage(records, splits_leaked) == 1
