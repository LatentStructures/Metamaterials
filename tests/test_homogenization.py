"""Homogenizer tests: closed-form validation, tensor sanity, property round-trips.

The FEA solver caches per-resolution problem data and carries module-level
material state, so every test pins ``set_base_material`` explicitly.
"""
import numpy as np
import pytest

from src.fea import (
    effective_properties,
    set_base_material,
    set_void_scale,
    voigt_isotropic,
)
from src.fea.benchmarks import (
    laminate_grid,
    laminate_stiffness,
    rods_closure,
    rods_grid,
    solid_cube_grid,
    voigt_reuss_bounds,
)
from src.fea.homogenization import (
    Homogenizer,
    homogenize_periodic_cell,
    validate_vs_analytical,
)
from src.geometry import periodic_pairing, voxels_to_tetra

RES = 4


@pytest.fixture(autouse=True)
def _reset_material():
    set_base_material(1.0, 0.3)
    set_void_scale(1.0e-6)
    yield


def test_validate_vs_analytical_passes_at_low_res():
    summary = validate_vs_analytical(tolerance=0.05, res=RES)
    assert summary["valid"] is True
    for name, info in summary["benchmarks"].items():
        assert info["ok"], name
        assert info["max_rel_err"] < 1.0e-5, name


def test_solid_cube_is_exact_isotropic():
    grid = solid_cube_grid(RES)
    mesh = voxels_to_tetra(grid)
    C6 = homogenize_periodic_cell(grid, mesh, None, RES)
    ep = effective_properties(C6)
    assert np.isclose(ep.E, 1.0, rtol=1e-6)
    assert np.isclose(ep.nu, 0.3, rtol=1e-6)


def test_tensor_sanity_symmetry_and_pd():
    grid, _ = rods_grid(RES, 0.5)
    mesh = voxels_to_tetra(grid)
    C6 = homogenize_periodic_cell(grid, mesh, None, RES)
    assert np.allclose(C6, C6.T, atol=1e-8)
    assert np.all(np.isfinite(C6))
    eigen = np.linalg.eigvalsh(C6)
    assert np.all(eigen > 1e-9)


def test_pairings_path_agrees_with_none():
    grid, _ = laminate_grid(RES, 0.4)
    mesh = voxels_to_tetra(grid)
    pairings = periodic_pairing(mesh, RES)
    C6a = homogenize_periodic_cell(grid, mesh, None, RES)
    C6b = homogenize_periodic_cell(grid, mesh, pairings, RES)
    assert np.allclose(C6a, C6b, atol=1e-10)


def test_laminate_matches_closed_form():
    grid, f_ach = laminate_grid(RES, 0.4)
    mesh = voxels_to_tetra(grid)
    C6 = homogenize_periodic_cell(grid, mesh, None, RES)
    E_s, nu_s = 1.0, 0.3
    ref = laminate_stiffness(f_ach, E_s, nu_s, 1.0e-6 * E_s, nu_s)
    assert np.allclose(C6, ref, rtol=1e-6, atol=1e-9)


def test_rods_uniaxial_combos_exact():
    grid, rho = rods_grid(RES, 0.5)
    mesh = voxels_to_tetra(grid)
    C6 = homogenize_periodic_cell(grid, mesh, None, RES)
    closures = rods_closure(C6, rho, 1.0, 0.3, 1.0e-6, 0.3)
    for value, target in closures.values():
        assert np.isclose(value, target, rtol=0.0, atol=1e-6)


def test_rods_closure_rejects_unequal_poisson():
    with pytest.raises(ValueError):
        rods_closure(np.eye(6), 0.5, 1.0, 0.3, 1.0e-6, 0.25)


def test_base_material_scales_stiffness():
    grid = solid_cube_grid(RES)
    mesh = voxels_to_tetra(grid)
    C6_1 = homogenize_periodic_cell(grid, mesh, None, RES)
    set_base_material(2.0, 0.3)
    C6_2 = homogenize_periodic_cell(grid, mesh, None, RES)
    assert np.allclose(C6_2, 2.0 * C6_1, rtol=1e-6)


def test_void_scale_controls_void_phase():
    set_base_material(1.0, 0.3)
    grid = np.zeros((RES, RES, RES), dtype=np.uint8)  # fully void
    grid[1, 1, 1] = 1
    mesh = voxels_to_tetra(grid)
    set_void_scale(1.0e-3)
    C6 = homogenize_periodic_cell(grid, mesh, None, RES)
    # with essentially all-void material the stiffness collapses to ~scale
    assert C6.max() < 5.0e-3


def test_property_round_trip_via_voigt():
    E, nu = 1.3, 0.25
    C6 = voigt_isotropic(E, nu)
    ep = effective_properties(C6)
    assert np.isclose(E, ep.E, rtol=1e-9)
    assert np.isclose(nu, ep.nu, rtol=1e-9)


def test_effective_properties_rejects_bad_matrices():
    with pytest.raises(ValueError):
        effective_properties(np.eye(6) * (-1.0))
    with pytest.raises(ValueError):
        effective_properties(-np.ones((6, 6)))


def test_voigt_reuss_bounds_bracket():
    rng = np.random.default_rng(3)
    C_s = voigt_isotropic(2.0, 0.32)
    C_v = voigt_isotropic(1.0e-3, 0.3)
    rho = 0.6
    C_V, C_R = voigt_reuss_bounds(rho, C_s, C_v)
    for i in range(6):
        assert C_R[i, i] < C_V[i, i]
    grid, rho = rods_grid(RES, 0.5)
    mesh = voxels_to_tetra(grid)
    C6 = homogenize_periodic_cell(grid, mesh, None, RES)
    C_V, C_R = voigt_reuss_bounds(rho, voigt_isotropic(1.0, 0.3),
                                  voigt_isotropic(1.0e-6, 0.3))
    for i in range(6):
        assert C_R[i, i] <= C6[i, i] + 1e-6 <= C_V[i, i] + 1e-6


def test_homogenizer_wrapper_times():
    h = Homogenizer(RES)
    grid, _ = rods_grid(RES, 0.5)
    mesh = voxels_to_tetra(grid)
    C6 = h(grid, mesh, None, RES)
    assert C6.shape == (6, 6)
    assert len(h.times) == 1 and h.times[0] > 0