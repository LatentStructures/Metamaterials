"""Tests for the visual evidence pack (reference-vs-generated display).

The pack is the display side of the acceptance loop: it pairs every generated
grid with the dataset reference cell its target was drawn from, features the
top-ranked candidate (same ranking as the STL export), and indexes every
displayed claim in ``evidence_manifest.json``. Nothing is re-simulated.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import evidence as ev
from src.data.dataset import (
    ConditioningStats,
    MetamaterialDataset,
    write_conditioned_hdf5,
)

STATS = ConditioningStats(
    mean=np.array([1.00, 0.30, 0.25], dtype=np.float32),
    std=np.array([0.50, 0.10, 0.05], dtype=np.float32),
)

RAW = np.array([
    [0.20, 0.40, 0.30],
    [0.45, 0.52, 0.26],
], dtype=np.float32)

ROWS = [
    {"file": "c00.npy", "target_E": 0.20, "target_rho": 0.40, "target_nu": 0.30,
     "achieved_rho": 0.395, "E_eff": 0.207, "nu_eff": 0.301,
     "valid": True, "reason": "ok"},
    {"file": "c01.npy", "target_E": 0.45, "target_rho": 0.52, "target_nu": 0.26,
     "achieved_rho": 0.500, "E_eff": 0.300, "nu_eff": 0.270,
     "valid": True, "reason": "ok"},
]


def _sphere(n: int = 16, r: float = 6.0) -> np.ndarray:
    g = np.arange(n)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    c = 0.5 * (n - 1)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    return (d <= r).astype(np.uint8)


def _dataset(tmp_path):
    path = tmp_path / "split.h5"
    voxels = np.stack([_sphere(r=6.0), _sphere(r=5.0)])
    cond = STATS.normalize(RAW).astype(np.float32)
    write_conditioned_hdf5(path, voxels, cond, stats=STATS)
    return path


def _pipeline_dirs(tmp_path, rows=ROWS):
    samples = tmp_path / "samples"
    evald = tmp_path / "eval"
    samples.mkdir()
    evald.mkdir()
    for r in rows:
        np.save(samples / r["file"], _sphere())
    pd.DataFrame(rows).to_csv(evald / "per_sample.csv", index=False)
    return samples, evald


def test_nearest_reference_matches_the_source_cell(tmp_path):
    ds = MetamaterialDataset(_dataset(tmp_path))
    idx, dist = ev.nearest_reference(RAW, ds)
    assert list(idx) == [0, 1]
    assert np.all(dist < 1e-6)


def test_nearest_reference_off_manifold_is_far(tmp_path):
    """An off-manifold target must surface as a large distance, so the panel
    can label it 'nearest cell' instead of pretending an exact match exists."""
    ds = MetamaterialDataset(_dataset(tmp_path))
    far = np.array([[0.90, 0.95, -0.50]], dtype=np.float32)
    _idx, dist = ev.nearest_reference(far, ds)
    assert dist[0] > ev.DEFAULT_MAX_MATCH_DIST


def test_pair_rgb_projects_and_colorizes():
    ref, gen = _sphere(n=8, r=3.0), _sphere(n=8, r=2.0)
    img = ev._pair_rgb(ref, gen)
    assert img.shape == (8, 2 * 8 + 2, 3)
    left, right = img[:, :8], img[:, 10:]
    # astype(bool) is load-bearing: max() of a uint8 0/1 grid stays integer,
    # and an integer array indexes values instead of masking (it would gather
    # along axis 0 and silently return a 4D result).
    ref_mask = ref.max(axis=0).astype(bool)
    gen_mask = gen.max(axis=0).astype(bool)
    assert np.all(left[ref_mask] == ev.REF_RGB)
    assert np.all(right[gen_mask] == ev.GEN_RGB)


def test_pair_rgb_marks_a_missing_generated_grid():
    img = ev._pair_rgb(_sphere(n=8, r=3.0), None)
    assert np.all(img[:, 10:] == ev.MISSING_RGB)


def test_run_evidence_writes_full_pack(tmp_path):
    ds_path = _dataset(tmp_path)
    samples, evald = _pipeline_dirs(tmp_path)
    export = tmp_path / "export"
    export.mkdir()
    pd.DataFrame([
        {"file": "c00.npy", "exported": True, "stl": "outputs/export/c00.stl",
         "reason": "ok", "vertices": 100, "faces": 200, "is_watertight": True,
         "euler_characteristic": -8},
        {"file": "c01.npy", "exported": False, "reason": "missing"},
    ]).to_csv(export / "export_manifest.csv", index=False)

    out = tmp_path / "out"
    man = ev.run_evidence(samples, evald, ds_path, out, export_dir=export, cols=2)

    for name in ("reference_vs_generated.png", "selected_pair.png",
                 "selected_candidate_3d.png", "export_montage.png",
                 "evidence_manifest.json"):
        f = out / name
        assert f.is_file() and f.stat().st_size > 0, name
    assert man["selected"]["file"] == "c00.npy"      # smaller E relative error
    assert man["selected"]["stl"]["is_watertight"] is True
    assert man["reference_match"]["n_exact"] == 2
    assert man["missing_grids"] == []
    on_disk = json.loads((out / "evidence_manifest.json").read_text())
    assert on_disk["selected"]["file"] == "c00.npy"
    assert on_disk["validity"]["valid"] == 2
    assert on_disk["images"]["export_montage"] == "export_montage.png"


def test_run_evidence_selected_override_and_no_export_dir(tmp_path):
    ds_path = _dataset(tmp_path)
    rows = [dict(ROWS[0]), {**ROWS[1], "valid": False, "reason": "connectivity"}]
    samples, evald = _pipeline_dirs(tmp_path, rows)
    out = tmp_path / "out"
    # stem-only --selected resolution; invalid row featureable on purpose
    man = ev.run_evidence(samples, evald, ds_path, out, export_dir=None,
                          selected="c01", cols=2)
    assert man["selected"]["file"] == "c01.npy"
    assert man["selected"]["valid"] is False
    assert man["images"]["export_montage"] is None


def test_run_evidence_requires_the_fea_evaluation(tmp_path):
    """Evidence without FEA verification would be misleading: refuse to run."""
    ds_path = _dataset(tmp_path)
    empty_eval = tmp_path / "eval_empty"
    empty_eval.mkdir()
    with pytest.raises(FileNotFoundError):
        ev.run_evidence(tmp_path / "samples", empty_eval, ds_path, tmp_path / "out")


def test_run_evidence_refuses_to_guess_when_nothing_valid(tmp_path):
    ds_path = _dataset(tmp_path)
    rows = [{**ROWS[0], "valid": False, "reason": "connectivity"},
            {**ROWS[1], "valid": False, "reason": "min_wall_thickness"}]
    samples, evald = _pipeline_dirs(tmp_path, rows)
    with pytest.raises(SystemExit):
        ev.run_evidence(samples, evald, ds_path, tmp_path / "out")
