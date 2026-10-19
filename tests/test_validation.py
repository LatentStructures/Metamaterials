"""Tests for the synthesizability filter (ROADMAP 4.6, step 4)."""
from __future__ import annotations

import numpy as np

import src.validation as v


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
    assert v.compute_components(_solid_cube()) == 1
    assert v.filter_connectivity(_solid_cube()) is True


def test_connectivity_floating_voxel_rejected():
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[0] = 1
    vol[-1] = 1
    vol[:, 0] = 1
    vol[:, -1] = 1
    vol[:, :, 0] = 1
    vol[:, :, -1] = 1                  # hollow box shell (connected)
    vol[3, 3, 3] = 1                   # floating interior voxel
    assert v.compute_components(vol) == 2
    assert v.filter_connectivity(vol) is False


def test_empty_grid_has_zero_components():
    assert v.compute_components(np.zeros((4, 4, 4), dtype=np.uint8)) == 0
    assert v.filter_connectivity(np.zeros((4, 4, 4), dtype=np.uint8)) is False


def test_corner_touching_voxels_are_one_component():
    """26-connectivity: cubes meeting only at a corner count as connected.

    This is the convention shared with the Option-B physics proxy, so the soft
    and hard validity signals agree on corner-touching structures.
    """
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[0, 0, 0] = 1                      # corner cube
    vol[1, 1, 1] = 1                      # touches it only at one corner
    assert v.compute_components(vol) == 1
    assert v.filter_connectivity(vol) is True
    # two cubes separated by a full-voxel gap are definitely disconnected
    vol2 = np.zeros((4, 4, 4), dtype=np.uint8)
    vol2[0, :, :] = 1
    vol2[3, :, :] = 1
    assert v.compute_components(vol2) == 2


def test_min_wall_thickness_thick_passes():
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, 2:6] = 1                 # 4-voxel-thick slab
    assert v.thin_solid_voxels(vol, 2) < 1e-12


def test_min_wall_thickness_thin_fails():
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, 3] = 1                   # 1-voxel-thin slab
    thin = v.thin_solid_voxels(vol, 2)
    assert thin > 0.5


def test_thin_fraction_monotonic_in_wall():
    thin = v.thin_solid_voxels(np.pad(np.ones((4, 4, 1), dtype=np.uint8),
                                      ((2, 2), (2, 2), (0, 0))), 2)
    assert thin > 0.5                 # a 1-voxel membrane is almost fully thin


def test_overhang_solid_cube_fully_supported():
    assert v.overhang_fraction(np.ones((6, 6, 6), dtype=np.uint8)) == 0.0


def test_overhang_suspended_slab_detected():
    vol = np.zeros((6, 6, 6), dtype=np.uint8)
    vol[:, :, 4:] = 1                 # suspended slab base layer lacks support
    assert v.overhang_fraction(vol) > 0.4


def test_overhang_floor_voxels_supported():
    vol = np.zeros((6, 6, 6), dtype=np.uint8)
    vol[2:4, 2:4] = 1                 # pillar from the platform
    assert v.overhang_fraction(vol) <= 0.5


def test_evaluate_validity_on_solid():
    ok, reason = v.evaluate_validity(_solid_cube(), {})
    assert ok and reason == "ok"


def test_evaluate_validity_reports_reason():
    ok, reason = v.evaluate_validity(np.zeros((8, 8, 8), dtype=np.uint8), {})
    assert not ok and reason == "connectivity"
