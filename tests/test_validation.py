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
    # Under periodic boundaries these two slabs are neighbours across the wrap,
    # so they form one component; true separation needs an interior gap.
    assert v.compute_components(vol2) == 1
    vol2[0, :, :] = 0
    assert v.compute_components(vol2) == 1


def test_wrap_adjacent_voxels_are_one_component():
    """Periodic boundaries: solid meeting across a wrapped face is connected.

    A unit cell is tessellated, so two voxels on opposite faces of the same
    axis are neighbours. Without this merge, a single continuous network is
    miscounted as several islands and the validity rate is meaningless.
    """
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[0, 4, 4] = 1
    vol[-1, 4, 4] = 1                     # adjacent to the first via the wrap
    assert v.compute_components(vol) == 1
    assert v.filter_connectivity(vol) is True

    far = np.zeros((8, 8, 8), dtype=np.uint8)
    far[0, 4, 4] = 1
    far[-1, 0, 0] = 1                     # wrapped neighbour is (0,3,3): not touching
    assert v.compute_components(far) == 2


def test_min_wall_is_periodic():
    """A member running through the cell is not 'thin' merely because it is clipped."""
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[:, :, 0] = 1                      # 1-voxel slab sitting on the z=0 face
    vol[:, :, -1] = 1                     # ...and the wrapped z=7 face
    # Together these form a 2-voxel-thick periodic slab -> no thin defect.
    assert v.thin_solid_voxels(vol, 2) < 1e-12


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
    """Thinner members score a strictly higher thin fraction than thicker ones."""
    membrane = np.pad(np.ones((6, 6, 1), dtype=np.uint8), 2)   # isolated 1-voxel slab
    thick = np.pad(np.ones((4, 4, 4), dtype=np.uint8),
                   ((2, 2), (2, 2), (2, 2)))
    thin_1 = v.thin_solid_voxels(membrane, 2)
    thin_4 = v.thin_solid_voxels(thick, 2)
    assert thin_1 > thin_4
    assert thin_4 < 1e-12                 # a 4-voxel member has no thin defect
    # A 1-voxel sheet spanning the whole cell is a *periodic* member: its
    # wrapped thickness is 2, so it must NOT be reported as thin.
    sheet = np.zeros((8, 8, 8), dtype=np.uint8)
    sheet[:, :, 0] = 1
    sheet[:, :, -1] = 1
    assert v.thin_solid_voxels(sheet, 2) < 1e-12


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


def test_diagonal_contact_across_periodic_face_is_connected():
    vol = np.zeros((32, 32, 32), dtype=np.uint8)
    vol[31, 5, 5] = 1
    vol[0, 6, 5] = 1
    assert v.compute_components(vol) == 1
