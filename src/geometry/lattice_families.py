"""Parametric unit-cell generators for the three baseline topology families
(ROADMAP section 3.1): cubic strut lattice, octet-truss, gyroid. Only these three.

Each family maps a single continuous parameter to a binary voxel grid on the
unit torus:

- ``cubic_strut``: radius ``r`` of the three mutually perpendicular cell-spanning
  strut rings along the coordinate axes.
- ``octet_truss``: radius ``r`` of the struts joining nearest neighbours of the
  FCC lattice (the actual nodal structure of the octet truss), fully periodic.
- ``gyroid``: sheet level-set threshold ``c`` of the standard TPMS implicit
  function; the solid is ``|sin(2pi x)cos(2pi y) + sin(2pi y)cos(2pi z) + sin(2pi z)cos(2pi x)| <= c``
  (sheet gyroid -- inversion-symmetric, so periodic face pairing is exact).

``sample_for_density`` inverts the (monotone) density--parameter map by
bisection so the dataset generator can target the section 3.1 density band
0.1-0.6.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from itertools import product

import numpy as np

from src.geometry.voxelize import implicit_grid, relative_density, strut_grid

BAND = (0.1, 0.6)  # target relative-density band (ROADMAP section 3.1)

FAMILIES = ("cubic_strut", "octet_truss", "gyroid")


# --------------------------------------------------------------------------- #
# Cubic strut lattice
# --------------------------------------------------------------------------- #
def cubic_strut_grid(radius: float, resolution: int) -> np.ndarray:
    r = float(radius)
    if not 0.0 < r < 0.5:
        raise ValueError(f"cubic_strut radius must be in (0, 0.5), got {r}")
    return strut_grid([], radius=r, resolution=resolution, axis_rings=[0, 1, 2])


# --------------------------------------------------------------------------- #
# Octet truss
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _fcc_nearest_neighbour_struts() -> list[tuple[np.ndarray, np.ndarray]]:
    """In-cell chords of all nearest-neighbour struts of the FCC lattice.

    FCC points in the reference cell (units of 0.5): all (i/2, j/2, k/2) with
    i+j+k even. A strut joins any two such points a lattice vector apart, i.e.
    at squared distance 0.5 (the 12 nearest-neighbour directions of the cell).
    """
    units = (0.0, 0.5, 1.0)
    pts = [
        np.array(p, dtype=float)
        for p in product(units, repeat=3)
        if (round(p[0] * 2) + round(p[1] * 2) + round(p[2] * 2)) % 2 == 0
    ]
    struts: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = pts[j] - pts[i]
            if np.dot(d, d) == 0.5:
                struts.append((pts[i], pts[j]))
    return struts


def octet_truss_grid(radius: float, resolution: int) -> np.ndarray:
    r = float(radius)
    if not 0.0 < r < 0.5:
        raise ValueError(f"octet_truss radius must be in (0, 0.5), got {r}")
    return strut_grid(_fcc_nearest_neighbour_struts(), radius=r, resolution=resolution)


# --------------------------------------------------------------------------- #
# Gyroid -- sheet form (|f| <= c), symmetric for exact PBC face pairing
# --------------------------------------------------------------------------- #
def _gyroid_implicit() -> Callable[[np.ndarray], np.ndarray]:
    """Absolute gyroid field ``|sin(2pi x) cos(2pi y) + sin(2pi y) cos(2pi z) + sin(2pi z) cos(2pi x)|``.

    The *sheet* gyroid (ROADMAP section 3.1) occupies the region ``|f| <= c``;
    it is inversion-symmetric, which guarantees exact periodic boundary pairing
    for any resolution.
    """
    def func(pts: np.ndarray) -> np.ndarray:
        p = 2.0 * np.pi * pts
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        return np.abs(np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x))

    return func


def gyroid_grid(threshold: float, resolution: int) -> np.ndarray:
    c = float(threshold)
    if not 0.0 <= c <= 1.5:
        raise ValueError(f"gyroid sheet threshold must be in [0, 1.5], got {c}")
    return implicit_grid(_gyroid_implicit(), threshold=c, resolution=resolution,
                         solid_where="<=")


# --------------------------------------------------------------------------- #
# Family dispatch + density targeting
# --------------------------------------------------------------------------- #
def generate_voxels(family: str, parameter: float, resolution: int) -> np.ndarray:
    """Rasterize one unit cell of ``family`` at ``parameter`` onto ``resolution``^3."""
    if family == "cubic_strut":
        return cubic_strut_grid(parameter, resolution)
    if family == "octet_truss":
        return octet_truss_grid(parameter, resolution)
    if family == "gyroid":
        return gyroid_grid(parameter, resolution)
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def parameter_bounds(family: str) -> tuple[float, float]:
    if family == "cubic_strut":
        return (0.005, 0.35)
    if family == "octet_truss":
        return (0.002, 0.20)
    if family == "gyroid":
        return (0.0, 1.5)
    raise ValueError(f"unknown family {family!r}")


def _density_at(family: str, parameter: float, resolution: int) -> float:
    return relative_density(generate_voxels(family, parameter, resolution))


def sample_for_density(
    family: str,
    target_density: float,
    resolution: int,
    tolerance: float = 1.5e-2,
    max_iter: int = 48,
) -> tuple[np.ndarray, float]:
    """Return (voxel grid, achieved_density) closest to ``target_density``.

    Assumes density is monotone in the parameter (it may decrease with
    increasing parameter).  Bisection probes the endpoints first to determine
    the direction.

    The binary grid quantizes density into coarse steps, so the achieved value
    lands on a plateau near the target.  Callers must record the *achieved*
    density (``relative_density(grid)``), not the target, in the manifest.
    """
    if not BAND[0] - tolerance <= target_density <= BAND[1] + tolerance:
        raise ValueError(f"target density {target_density} outside baseline band {BAND}")
    lo, hi = parameter_bounds(family)
    d_lo, d_hi = _density_at(family, lo, resolution), _density_at(family, hi, resolution)
    if not (min(d_lo, d_hi) <= target_density <= max(d_lo, d_hi)):
        raise RuntimeError(
            f"{family} density [{d_lo:.3f}, {d_hi:.3f}] cannot reach {target_density}"
        )
    if abs(d_lo - target_density) <= tolerance:
        return generate_voxels(family, lo, resolution), d_lo
    if abs(d_hi - target_density) <= tolerance:
        return generate_voxels(family, hi, resolution), d_hi

    increasing = d_hi > d_lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d_mid = _density_at(family, mid, resolution)
        if abs(d_mid - target_density) <= tolerance:
            lo = hi = mid
            d_lo = d_hi = d_mid
            break
        if (d_mid < target_density) == increasing:
            lo, d_lo = mid, d_mid
        else:
            hi, d_hi = mid, d_mid
    # Pick the bracketing parameter with the achieved density closest to target.
    if abs(d_hi - target_density) < abs(d_lo - target_density):
        final_p = hi
    else:
        final_p = lo
    grid = generate_voxels(family, final_p, resolution)
    return grid, float(relative_density(grid))