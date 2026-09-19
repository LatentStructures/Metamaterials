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
    with open(path, "w", encoding="utf-8") as fh:
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