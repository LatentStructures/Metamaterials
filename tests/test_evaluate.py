"""Tests for the FEA-verified evaluation + synthesizability filter (4.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.evaluate as ev


def _solid_cube(n: int = 8) -> np.ndarray:
    vol = np.ones((n, n, n), dtype=np.uint8)
    vol[:1] = 0
    vol[-1:] = 0
    vol[:, :1] = 0
    vol[:, -1:] = 0
    vol[:, :, :1] = 0
    vol[:, :, -1:] = 0
    return vol


# --------------------------------------------------------------------------- #
# Synthesizability filter
# --------------------------------------------------------------------------- #

def test_connectivity_single_component():
    assert ev.compute_components(_solid_cube()) == 1
    assert ev.filter_connectivity(_solid_cube()) is True


def test_connectivity_floating_voxel_rejected():
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[0] = 1
    vol[-1] = 1
    vol[:, 0] = 1
    vol[:, -1] = 1
    vol[:, :, 0] = 1
    vol[:, :, -1] = 1                  # hollow box shell (connected)
    vol[3, 3, 3] = 1                   # floating interior voxel
    assert ev.compute_components(vol) == 2
    assert ev.filter_connectivity(vol) is False


def test_empty_grid_has_zero_components():
    assert ev.compute_components(np.zeros((4, 4, 4), dtype=np.uint8)) == 0
    assert ev.filter_connectivity(np.zeros((4, 4, 4), dtype=np.uint8)) is False


def test_corner_touching_voxels_are_one_component():
    """26-connectivity: cubes meeting only at a corner count as connected.

    This is the convention shared with the Option-B physics proxy, so the soft
    and hard validity signals agree on corner-touching structures.
    """
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[0, 0, 0] = 1                      # corner cube
    vol[1, 1, 1] = 1                      # touches it only at one corner
    assert ev.compute_components(vol) == 1
    assert ev.filter_connectivity(vol) is True
    # two cubes separated by a full-voxel gap are definitely disconnected
    vol2 = np.zeros((4, 4, 4), dtype=np.uint8)
    vol2[0, :, :] = 1
    vol2[3, :, :] = 1
    assert ev.compute_components(vol2) == 2


def test_min_wall_thickness_thick_passes():
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, 2:6] = 1                 # 4-voxel-thick slab
    assert ev.thin_solid_voxels(vol, 2) < 1e-12


def test_min_wall_thickness_thin_fails():
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, 3] = 1                   # 1-voxel-thin slab
    thin = ev.thin_solid_voxels(vol, 2)
    assert thin > 0.5


def test_thin_fraction_monotonic_in_wall():
    thin = ev.thin_solid_voxels(np.pad(np.ones((4, 4, 1), dtype=np.uint8),
                                       ((2, 2), (2, 2), (0, 0))), 2)
    assert thin > 0.5                 # a 1-voxel membrane is almost fully thin


def test_overhang_solid_cube_fully_supported():
    assert ev.overhang_fraction(np.ones((6, 6, 6), dtype=np.uint8)) == 0.0


def test_overhang_suspended_slab_detected():
    vol = np.zeros((6, 6, 6), dtype=np.uint8)
    vol[:, :, 4:] = 1                 # suspended slab base layer lacks support
    assert ev.overhang_fraction(vol) > 0.4


def test_overhang_floor_voxels_supported():
    vol = np.zeros((6, 6, 6), dtype=np.uint8)
    vol[2:4, 2:4] = 1                 # pillar from the platform
    assert ev.overhang_fraction(vol) <= 0.5


def test_evaluate_validity_on_solid():
    ok, reason = ev.evaluate_validity(_solid_cube(), {})
    assert ok and reason == "ok"


def test_evaluate_validity_reports_reason():
    ok, reason = ev.evaluate_validity(np.zeros((8, 8, 8), dtype=np.uint8), {})
    assert not ok and reason == "connectivity"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_reconstruction_metrics_exact():
    pred = np.array([[1.0, 0.3, 0.3], [2.0, 0.4, 0.35]])
    tgt = pred.copy()
    m = ev.reconstruction_metrics(pred, tgt, ev.COND_NAMES)
    for k in range(3):
        assert m[ev.COND_NAMES[k]]["mean_abs_rel_err"] == 0.0
        assert m[ev.COND_NAMES[k]]["r2"] == 1.0


def test_reconstruction_metrics_known_values():
    pred = np.array([[1.0, 0.5, 0.3]], dtype=float)
    tgt = np.array([[2.0, 0.5, 0.3]], dtype=float)
    m = ev.reconstruction_metrics(pred, tgt, ev.COND_NAMES)
    assert m["E"]["mean_abs_rel_err"] == 0.5
    assert m["relative_density"]["mean_abs_rel_err"] == 0.0
    assert m["E"]["bias"] == -1.0


def test_reconstruction_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ev.reconstruction_metrics(np.zeros((3, 2)), np.zeros((3, 3)), ev.COND_NAMES)


# --------------------------------------------------------------------------- #
# End-to-end run with a stubbed FEA provider
# --------------------------------------------------------------------------- #

def test_run_evaluate_end_to_end(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    out = tmp_path / "out"
    grids, targets = [], []
    for i in range(4):
        vol = _solid_cube(8)
        grids.append(vol)
        np.save(samples / f"s{i:02d}.npy", vol)
        rho = float(vol.mean())
        targets.append({"file": f"s{i:02d}.npy",
                        "target_E": rho, "target_rho": rho, "target_nu": 0.3})
    pd.DataFrame(targets).to_csv(samples / "samples_manifest.csv", index=False)

    def fake_fea(voxels, res):
        return _iso_c6(float(voxels.mean()))

    cfg = {"base_material": {"E": 1.0, "nu": 0.3},
           "void_scale": 1.0e-6, "resolution": 8}
    report = ev.run_evaluate(cfg, samples, out, no_filter=False, fea_fn=fake_fea)

    assert report["total_samples"] == 4
    assert report["fea_failures"] == 0
    assert report["filter_used"] == "own_core"
    assert report["validity_rate"] == 1.0
    rec = report["reconstruction"]
    assert "E" in rec and "relative_density" in rec and "nu" in rec
    assert abs(rec["E"]["mean_abs_rel_err"]) < 1e-9

    assert (out / "report.json").exists()
    assert (out / "per_sample.csv").exists()
    assert (out / "parity.png").exists()
    assert (out / "relerr.png").exists()


def _iso_c6(rho: float, E_base: float = 1.0, nu: float = 0.3) -> np.ndarray:
    E = rho * E_base
    C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    C12 = C11 * nu / (1 - nu)
    C44 = E / (2.0 * (1.0 + nu))                 # shear modulus mu
    C6 = np.zeros((6, 6))
    C6[:3, :3] = C12
    C6[0, 0] = C6[1, 1] = C6[2, 2] = C11
    C6[3, 3] = C6[4, 4] = C6[5, 5] = C44
    return C6


def test_run_evaluate_no_filter_uses_all(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    out = tmp_path / "out"
    vol = np.zeros((8, 8, 8), dtype=np.uint8)   # invalid: empty grid
    np.save(samples / "s00.npy", vol)
    pd.DataFrame([{"file": "s00.npy", "target_E": 0.3,
                   "target_rho": 0.3, "target_nu": 0.3}]
                 ).to_csv(samples / "samples_manifest.csv", index=False)

    def fake_fea(voxels, res):
        return _iso_c6(0.3)

    cfg = {"base_material": {"E": 1.0, "nu": 0.3},
           "void_scale": 1.0e-6, "resolution": 8}
    report = ev.run_evaluate(cfg, samples, out, no_filter=True, fea_fn=fake_fea)
    assert report["filter_used"] == "none"
    assert report["validity_rate"] == 1.0     # no_filter: all samples counted


def test_run_evaluate_fea_failure_excluded(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    out = tmp_path / "out"
    np.save(samples / "s00.npy", _solid_cube(4))
    pd.DataFrame([{"file": "s00.npy", "target_E": 0.3,
                   "target_rho": 0.3, "target_nu": 0.3}]
                 ).to_csv(samples / "samples_manifest.csv", index=False)

    def boom(voxels, res):
        raise RuntimeError("no solver")

    cfg = {"base_material": {"E": 1.0, "nu": 0.3},
           "void_scale": 1.0e-6, "resolution": 4}
    report = ev.run_evaluate(cfg, samples, out, no_filter=True, fea_fn=boom)
    assert report["fea_failures"] == 1
    assert report["total_samples"] == 1
    assert "reconstruction" not in report or report["reconstruction"] == {}