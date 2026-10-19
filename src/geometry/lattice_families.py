"""Parametric unit-cell generators for the three baseline topology families
(ROADMAP section 3.1): cubic strut lattice, octet-truss, gyroid. Only these three.

Each family maps the target density and a second continuous shape parameter
to a binary voxel grid on the unit torus:

- ``cubic_strut``: solved strut radius plus a cubic-symmetric joint-radius ratio.
- ``octet_truss``: solved FCC strut radius plus its joint-radius ratio.
- ``gyroid``: sheet half-thickness plus a signed level-set offset of the standard
  periodic gyroid field.

``sample_for_density`` inverts the (monotone) density--parameter map by
bisection so the dataset generator can target the section 3.1 density band
0.1-0.6.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from src.geometry.voxelize import (
    grid_coords,
    implicit_grid,
    relative_density,
    strut_grid,
)

BAND = (0.1, 0.6)  # target relative-density band (ROADMAP section 3.1)

FAMILIES = ("cubic_strut", "octet_truss", "gyroid")


# --------------------------------------------------------------------------- #
# Cubic strut lattice
# --------------------------------------------------------------------------- #
def _add_periodic_nodes(grid: np.ndarray, points: list[np.ndarray],
                        radius: float) -> np.ndarray:
    """Union equal-radius periodic spherical joints into a strut network."""
    if radius <= 0:
        return grid
    res = grid.shape[0]
    g = grid_coords(res)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    node_mask = np.zeros(len(pts), dtype=bool)
    for point in points:
        for shift in product((-1, 0, 1), repeat=3):
            delta = pts - (point + np.asarray(shift))
            node_mask |= np.einsum("ij,ij->i", delta, delta) <= radius**2
    return (grid.ravel().astype(bool) | node_mask).reshape(grid.shape).astype(np.uint8)


def cubic_strut_grid(radius: float, resolution: int,
                     node_factor: float = 1.0) -> np.ndarray:
    r = float(radius)
    if not 0.0 < r < 0.5:
        raise ValueError(f"cubic_strut radius must be in (0, 0.5), got {r}")
    if node_factor < 1.0:
        raise ValueError("node_factor must be >= 1")
    base = strut_grid([], radius=r, resolution=resolution, axis_rings=[0, 1, 2])
    return _add_periodic_nodes(base, [np.zeros(3)], r * node_factor)


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


def octet_truss_grid(radius: float, resolution: int,
                     node_factor: float = 1.0) -> np.ndarray:
    r = float(radius)
    if not 0.0 < r < 0.5:
        raise ValueError(f"octet_truss radius must be in (0, 0.5), got {r}")
    if node_factor < 1.0:
        raise ValueError("node_factor must be >= 1")
    base = strut_grid(_fcc_nearest_neighbour_struts(), radius=r, resolution=resolution)
    nodes = [np.array(p, dtype=float) for p in product((0.0, 0.5), repeat=3)
             if sum(round(v * 2) for v in p) % 2 == 0]
    return _add_periodic_nodes(base, nodes, r * node_factor)


# --------------------------------------------------------------------------- #
# Gyroid -- periodic sheet between two level surfaces
# --------------------------------------------------------------------------- #
def gyroid_grid(threshold: float, resolution: int,
                level_offset: float = 0.0) -> np.ndarray:
    c = float(threshold)
    if not 0.0 <= c <= 1.5:
        raise ValueError(f"gyroid sheet threshold must be in [0, 1.5], got {c}")
    def field(pts: np.ndarray) -> np.ndarray:
        p = 2.0 * np.pi * pts
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
                + np.sin(z) * np.cos(x))
    # A shifted pair of parallel gyroid level surfaces retains the same periodic
    # implicit field while adding a continuous sheet-placement parameter.
    def band(pts: np.ndarray) -> np.ndarray:
        return np.abs(field(pts) - level_offset)
    return implicit_grid(band, threshold=c, resolution=resolution, solid_where="<=")


# --------------------------------------------------------------------------- #
# Family dispatch + density targeting
# --------------------------------------------------------------------------- #
def generate_voxels(family: str, parameter: float, resolution: int,
                    shape_parameter: float | None = None) -> np.ndarray:
    """Rasterize one unit cell of ``family`` at ``parameter`` onto ``resolution``^3."""
    if shape_parameter is None:
        shape_parameter = 0.0 if family == "gyroid" else 1.0
    if family == "cubic_strut":
        return cubic_strut_grid(parameter, resolution, shape_parameter)
    if family == "octet_truss":
        return octet_truss_grid(parameter, resolution, shape_parameter)
    if family == "gyroid":
        return gyroid_grid(parameter, resolution, shape_parameter)
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def parameter_bounds(family: str) -> tuple[float, float]:
    if family == "cubic_strut":
        return (0.005, 0.35)
    if family == "octet_truss":
        return (0.002, 0.20)
    if family == "gyroid":
        return (0.0, 1.5)
    raise ValueError(f"unknown family {family!r}")


def _density_at(family: str, parameter: float, resolution: int,
                shape_parameter: float) -> float:
    return relative_density(generate_voxels(family, parameter, resolution, shape_parameter))


def sample_for_density(
    family: str,
    target_density: float,
    resolution: int,
    tolerance: float = 1.5e-2,
    max_iter: int = 48,
    shape_parameter: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return (voxel grid, achieved_density) closest to ``target_density``.

    Assumes density is monotone in the parameter (it may decrease with
    increasing parameter).  Bisection probes the endpoints first to determine
    the direction.

    The binary grid quantizes density into coarse steps, so the achieved value
    lands on a plateau near the target.  Callers must record the *achieved*
    density (``relative_density(grid)``), not the target, in the manifest.
    """
    if shape_parameter is None:
        shape_parameter = 0.0 if family == "gyroid" else 1.0
    if not BAND[0] - tolerance <= target_density <= BAND[1] + tolerance:
        raise ValueError(f"target density {target_density} outside baseline band {BAND}")
    lo, hi = parameter_bounds(family)
    d_lo = _density_at(family, lo, resolution, shape_parameter)
    d_hi = _density_at(family, hi, resolution, shape_parameter)
    if not (min(d_lo, d_hi) <= target_density <= max(d_lo, d_hi)):
        raise RuntimeError(
            f"{family} density [{d_lo:.3f}, {d_hi:.3f}] cannot reach {target_density}"
        )
    if abs(d_lo - target_density) <= tolerance:
        return generate_voxels(family, lo, resolution, shape_parameter), d_lo
    if abs(d_hi - target_density) <= tolerance:
        return generate_voxels(family, hi, resolution, shape_parameter), d_hi

    increasing = d_hi > d_lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d_mid = _density_at(family, mid, resolution, shape_parameter)
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
    grid = generate_voxels(family, final_p, resolution, shape_parameter)
    return grid, float(relative_density(grid))
