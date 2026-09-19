"""Geometry package tests: density targeting, mesh positivity, PBC pairing."""
import numpy as np
import pytest

from src.geometry import (
    parameter_bounds,
    periodic_pairing,
    sample_for_density,
    voxels_to_surface,
    voxels_to_tetra,
)

FAMILIES = ("cubic_strut", "octet_truss", "gyroid")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("res", [16, 32])
def test_density_targeting_within_tolerance(family, res):
    tol = 0.015 if res == 32 else 0.03  # 16^3 grids quantize in coarser plateaus
    for target in (0.2, 0.5):
        grid, achieved = sample_for_density(family, target, res)
        assert grid.dtype == np.uint8 and grid.shape == (res, res, res)
        assert abs(achieved - target) <= tol, (family, res, target, achieved)
        assert 0.0 < achieved < 1.0


def test_all_families_have_bounds():
    for family in FAMILIES:
        lo, hi = parameter_bounds(family)
        assert lo < hi and lo >= 0.0


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("res", [16, 32])
def test_struct_tetra_mesh_positive_and_flagged(family, res):
    grid, _ = sample_for_density(family, 0.3, res)
    mesh = voxels_to_tetra(grid)
    pts = mesh.points
    cells = mesh.cells_dict["tetra"]
    assert len(cells) == res**3 * 6, len(cells)
    v01 = pts[cells[:, 1]] - pts[cells[:, 0]]
    v02 = pts[cells[:, 2]] - pts[cells[:, 0]]
    v03 = pts[cells[:, 3]] - pts[cells[:, 0]]
    vols = np.einsum("ij,ij->i", v01, np.cross(v02, v03)) / 6.0
    assert (vols > 0).all(), "mesh contains degenerate or inverted tets"
    void = mesh.cell_data["void"][0]
    solid = mesh.cell_data["solid"][0]
    assert (void | solid == 1).all()
    assert (void == 0).sum() == 6 * int(grid.sum()), "solid tet count != 6 * solid voxels"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("res", [16, 32])
def test_periodic_pairing_is_exact(family, res):
    grid, _ = sample_for_density(family, 0.3, res)
    mesh = voxels_to_tetra(grid)
    pairing = periodic_pairing(mesh, res)
    assert set(pairing) == {0, 1, 2}
    eps = 1e-6
    for axis, (lo, hi) in pairing.items():
        assert len(lo) == len(hi) == (res + 1) ** 2
        assert np.allclose(mesh.points[lo][:, axis], 0.0, atol=eps)
        assert np.allclose(mesh.points[hi][:, axis], 1.0, atol=eps)
        # partners carry identical in-plane coordinates (periodicity identity).
        in_plane = np.delete(np.arange(3), axis)
        assert np.allclose(mesh.points[lo][:, in_plane], mesh.points[hi][:, in_plane])


def test_surface_mesher_runs():
    grid, _ = sample_for_density("gyroid", 0.3, 16)
    surf = voxels_to_surface(grid)
    assert len(surf.cells_dict["triangle"]) > 0