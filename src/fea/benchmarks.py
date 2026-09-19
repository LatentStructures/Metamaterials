"""Closed-form homogenization benchmarks and bounds (ROADMAP 3.3).

The FEA homogenizer is validated against three analytically exact unit cells
before any dataset generation is allowed to run:

1. ``solid_cube`` -- fully solid isotropic cube; the macroscopically homogeneous
   solution is exact on the mesh, so ``C = C_iso(E_s, nu_s)``.
2. ``laminate_x`` -- periodic two-phase laminate with layers perpendicular to x
   (volume fraction ``f_s``).  A 6x6 linear system from traction continuity
   (sigma_11, sigma_12, sigma_13) + kinematic averages reproduces the exact
   effective tensor exactly (Voigt-type mixing for the in-plane shear C44_23,
   Reuss-type compliance for the axial modulus).
3. ``aligned_rods_x`` -- solid rods running along x.  The periodic-array
   response under a *macro* ``e11`` load is NOT the rule of mixtures (the macro
   lateral strains are pinned to zero, so ``C_1111 = C_11bar + C_12bar**2 / ...``
   exceeds the Voigt value whenever ``nu > 0``).  The exact statement is for the
   uniaxial affine field ``eps_bar = diag(1, -nu, -nu)``: with both phases
   sharing the same Poisson ratio the fluctuation vanishes and ``sigma_bar = 0``
   in every transverse direction.  The tensile combination becomes exact:
   ``C11 - nu*(C12 + C13) = f_s E_s + f_v E_v`` together with the five
   transverse/homogenisation combinations ``Cj1 - nu*(Cj2 + Cj3) = 0``.

``voigt_reuss_bounds`` provides the two-phase Voigt and Reuss bounds on the
stiffness diagonal; it is used by the validation tests as a bracketing check on
the computed effective tensors.

All Voigt matrices use the engineering strain convention matching
``src/data/augment.py``: strain vector ``[e11, e22, e33, g23, g13, g12]``.
"""
from __future__ import annotations

import numpy as np

# unit macroscopic strain load cases as engineering-strain vectors
# (e11, e22, e33, g23, g13, g12)
_UNIT_STRAINS = np.eye(6)


def _iso(E: float, nu: float) -> tuple[float, float]:
    """(lam, mu) Lame parameters for an isotropic material."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def laminate_stiffness(f_s: float, E_s: float, nu_s: float,
                       E_v: float, nu_v: float) -> np.ndarray:
    """Exact 6x6 effective stiffness of a two-phase laminate layered along x.

    Solves the traction-continuity / kinematic-average system defined in the
    module docstring.  Returns the engineering-Voigt stiffness matrix.
    """
    if not (0.0 < f_s < 1.0):
        raise ValueError(f"laminate volume fraction must be in (0, 1), got {f_s}")
    f_v = 1.0 - f_s
    lam = [_iso(E_s, nu_s)[0], _iso(E_v, nu_v)[0]]
    mu = [_iso(E_s, nu_s)[1], _iso(E_v, nu_v)[1]]

    # unknowns: [eps_11^s, eps_11^v, g_12^s, g_12^v, g_13^s, g_13^v]
    A = np.zeros((6, 6))
    # traction continuity: sigma_11, sigma_12, sigma_13
    A[0, 0] = lam[0] + 2.0 * mu[0]
    A[0, 1] = -(lam[1] + 2.0 * mu[1])
    A[1, 2] = mu[0]
    A[1, 3] = -mu[1]
    A[2, 4] = mu[0]
    A[2, 5] = -mu[1]
    # kinematic averages
    A[3, 0] = f_s
    A[3, 1] = f_v
    A[4, 2] = f_s
    A[4, 3] = f_v
    A[5, 4] = f_s
    A[5, 5] = f_v

    C6 = np.zeros((6, 6))
    for k, (e11, e22, e33, g23, g13, g12) in enumerate(_UNIT_STRAINS):
        t = e22 + e33
        b = np.array([
            lam[1] * t - lam[0] * t,
            0.0, 0.0,
            e11, g12, g13,
        ])
        eps11s, eps11v, g12s, _g12v, g13s, _g13v = np.linalg.solve(A, b)
        s11 = (lam[0] + 2.0 * mu[0]) * eps11s + lam[0] * t
        s22 = (f_s * ((lam[0] + 2.0 * mu[0]) * e22 + lam[0] * (eps11s + e33))
               + f_v * ((lam[1] + 2.0 * mu[1]) * e22 + lam[1] * (eps11v + e33)))
        s33 = (f_s * ((lam[0] + 2.0 * mu[0]) * e33 + lam[0] * (eps11s + e22))
               + f_v * ((lam[1] + 2.0 * mu[1]) * e33 + lam[1] * (eps11v + e22)))
        s23 = (f_s * mu[0] + f_v * mu[1]) * g23
        s13 = mu[0] * g13s
        s12 = mu[0] * g12s
        C6[k] = [s11, s22, s33, s23, s13, s12]
    return C6


def rods_closure(C6: np.ndarray, rho: float, E_s: float, nu_s: float,
                 E_v: float, nu_v: float) -> dict[str, tuple[float, float]]:
    """Exact linear-combination closures of the aligned-rods periodic array.

    Evaluates the six ``sigma_bar`` components of the exact uniaxial field
    ``eps_bar = diag(1, -nu, -nu)`` against the computed ``C6``; equal Poisson
    ratios of the two phases make the fluctuation identically zero, so each
    combination is exact up to round-off.

    Returns ``{name: (computed, target)}`` (target ``0`` for the transverse
    entries).  Raises ``ValueError`` if the Poisson ratios differ.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rod volume fraction must be in (0, 1), got {rho}")
    if not np.isclose(nu_s, nu_v):
        raise ValueError(
            "aligned-rod exact closure requires equal Poisson ratios "
            f"(nu_s={nu_s}, nu_v={nu_v}) -- do not use this benchmark otherwise"
        )
    C = np.asarray(C6, dtype=float)
    ebar = rho * E_s + (1.0 - rho) * E_v  # rule of mixtures, tensile field
    names = ["s11", "s22", "s33", "s23", "s13", "s12"]
    out: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(names):
        val = C[j, 0] - nu_s * (C[j, 1] + C[j, 2])
        target = ebar if j == 0 else 0.0
        out[name] = (val, target)
    return out


def voigt_reuss_bounds(rho: float, C_s: np.ndarray,
                       C_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase Voigt (upper) and Reuss (lower) bounds on the stiffness diagonal.

    ``C_V = rho C_s + (1-rho) C_v``, ``C_R = (rho S_s + (1-rho) S_v)^-1``.
    For any two-phase composite ``diag(C_R) <= diag(C) <= diag(C_V)`` elementwise.
    """
    C_s = np.asarray(C_s, dtype=float)
    C_v = np.asarray(C_v, dtype=float)
    C_V = rho * C_s + (1.0 - rho) * C_v
    S_R = rho * np.linalg.inv(C_s) + (1.0 - rho) * np.linalg.inv(C_v)
    C_R = np.linalg.inv(S_R)
    return C_V, C_R


# --------------------------------------------------------------------------- #
# Benchmark voxel-grid builders (pure numpy; meshing happens in the FEA layer)
# --------------------------------------------------------------------------- #

def solid_cube_grid(res: int) -> np.ndarray:
    """Fully solid ``(res, res, res)`` occupancy grid."""
    return np.ones((res, res, res), dtype=np.uint8)


def laminate_grid(res: int, f_s: float) -> tuple[np.ndarray, float]:
    """Voxel laminate along x; returns (grid, achieved solid fraction)."""
    grid = np.zeros((res, res, res), dtype=np.uint8)
    solid = int(np.floor(f_s * res))
    grid[:solid, :, :] = 1
    f_ach = solid / res
    return grid, f_ach


def rods_grid(res: int, rho: float) -> tuple[np.ndarray, float]:
    """Solid rods along x (unit resolvable columns); returns (grid, achieved rho).

    Columns are scattered pseudo-randomly (seeded) rather than packed
    contiguously -- a contiguous ``cols[:ncols]`` block degenerates to a slab of
    the first ``y`` rows, which turns the benchmark into a laminate and breaks
    the axial (rule-of-mixtures) closure.
    """
    ncols = round(rho * res * res)
    ncols = max(1, min(ncols, res * res - 1))
    cols = np.zeros(res * res, dtype=np.uint8)
    cols.flat[np.random.default_rng(0).permutation(res * res)[:ncols]] = 1
    cols = cols.reshape(res, res)
    grid = np.broadcast_to(cols, (res, res, res)).copy().astype(np.uint8)
    rho_ach = ncols / (res * res)
    return grid, rho_ach