"""Voxelization of unit-cell generators onto a fixed binary grid (ROADMAP section 3.2).

All geometry lives on the flat unit torus [0, 1)^3 and is rasterized with
periodic boundary handling (ROADMAP section 3.2 item 2): struts that cross a
cell face continue on the opposite face, exactly as the FEA periodic-boundary
solver assumes.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Periodic distance helpers
# --------------------------------------------------------------------------- #
def periodic_segment_dist_sq(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distance from points ``x`` to a strut segment on the torus.

    ``a``, ``b`` are in-cell endpoints (in [0, 1]^3). The torus distance is the
    minimum Euclidean distance to *any* lattice copy of the segment; copies
    shifted by at most one cell per axis always contain the nearest image
    because every strut is shorter than the cell in every coordinate.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    best = np.full(x.shape[0], np.inf)
    v = b - a
    vv = float(v @ v)
    for shift in itertools.product((-1, 0, 1), repeat=3):
        s = np.array(shift, dtype=float)
        w = x - (a + s)
        t = np.clip((w @ v) / vv, 0.0, 1.0)
        proj = w - t[:, None] * v
        best = np.minimum(best, np.einsum("ij,ij->i", proj, proj))
    return best


def periodic_axis_dist_sq(x: np.ndarray, axis: int) -> np.ndarray:
    """Squared distance to a cell-spanning strut ring along ``axis``.

    The cubic-lattice struts run the full length of the cell, i.e. they are
    rings on the torus; the distance is purely the periodic distance in the two
    transverse coordinates.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    other = [j for j in range(3) if j != axis]
    d = np.abs(x[:, other])
    return np.sum(np.square(np.minimum(d, 1.0 - d)), axis=1)


# --------------------------------------------------------------------------- #
# Generic rasterizers
# --------------------------------------------------------------------------- #
def grid_coords(resolution: int) -> np.ndarray:
    """Voxel center coordinates on [0, 1)^3 (cell-centered sampling)."""
    return (np.arange(resolution, dtype=float) + 0.5) / resolution


def strut_grid(
    segments: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    resolution: int,
    axis_rings: list[int] | None = None,
) -> np.ndarray:
    """Binary occupancy grid from a list of periodic strut segments.

    ``axis_rings`` optionally adds the three full-cell axis struts (cubic
    lattice), which are cheaper to evaluate as rings than as segments.
    """
    g = grid_coords(resolution)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    dist_sq = np.full(pts.shape[0], np.inf)
    for a, b in segments:
        dist_sq = np.minimum(dist_sq, periodic_segment_dist_sq(pts, a, b))
    for k in axis_rings or []:
        dist_sq = np.minimum(dist_sq, periodic_axis_dist_sq(pts, k))

    grid = (dist_sq < radius * radius).reshape((resolution,) * 3)
    return grid.astype(np.uint8)


def implicit_grid(func: Callable[[np.ndarray], np.ndarray], threshold: float,
                  resolution: int, solid_where: str = ">=") -> np.ndarray:
    """Binary occupancy grid from an implicit function on [0, 1)^3.

    ``solid_where`` selects the solid side of the level set (e.g. gyroid).
    """
    g = grid_coords(resolution)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij", sparse=False)
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vals = np.asarray(func(pts), dtype=float).reshape((resolution,) * 3)
    if solid_where == ">=":
        grid = vals >= threshold
    elif solid_where == "<=":
        grid = vals <= threshold
    else:
        raise ValueError(f"solid_where must be '>=' or '<=', got {solid_where!r}")
    return grid.astype(np.uint8)


def relative_density(grid: np.ndarray) -> float:
    """Solid fraction of a binary voxel grid (the conditioning target)."""
    return float(np.mean(grid, dtype=float))