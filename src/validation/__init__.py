"""Synthesizability and physical validation (ROADMAP section 2 / section 4.6).

Owns the hard, binary checks that decide whether a generated unit cell is
synthesizable: a single 26-connected solid component, no 1-voxel-thin walls, and
a coarse additive-manufacturing overhang estimate. ``src.evaluate`` applies these
to every generated sample after FEA re-verification (ROADMAP 4.6, step 4).

The 3x3x3 adjacency convention here matches the Option-B physics proxy in
``src/losses/physics_loss.py``, so the soft training signal and this hard filter
agree on corner-touching structures.

Optional override: define ``filter_generated(vol: np.ndarray) -> bool`` in this
module to replace the built-in heuristic during evaluation; ``src.evaluate``
uses it when present.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "compute_components",
    "evaluate_validity",
    "filter_connectivity",
    "overhang_fraction",
    "thin_solid_voxels",
]


def compute_components(vol: np.ndarray) -> int:
    """Number of 26-connected solid components (0 for an empty grid).

    The 3x3x3 structuring element matches the adjacency convention used by the
    Option-B physics proxy (``src/losses/physics_loss.py``), so the soft
    connectivity signal and the hard validity filter agree on corner-touching
    structures.
    """
    solid = np.asarray(vol, dtype=bool)
    if not solid.any():
        return 0
    _, n = ndimage.label(solid, structure=np.ones((3, 3, 3), dtype=bool))
    return int(n)


def filter_connectivity(vol: np.ndarray) -> bool:
    """True iff the solid phase is a single connected component."""
    return compute_components(vol) == 1


def thin_solid_voxels(vol: np.ndarray, min_wall: int = 2) -> float:
    """Fraction of solid voxels removed by an opening with a ``min_wall`` cube.

    A single opening strips features thinner than ``min_wall`` voxels, so a
    nonzero fraction flags 1-voxel-thin walls.
    """
    solid = np.asarray(vol, dtype=bool)
    if not solid.any():
        return 0.0
    structure = np.ones((min_wall,) * 3, dtype=bool)
    opened = ndimage.binary_opening(solid, structure=structure)
    return float(solid.sum() - opened.sum()) / float(solid.sum())


def overhang_fraction(vol: np.ndarray) -> float:
    """Approximate fraction of solid voxels lacking downward support.

    A voxel at z>0 is "supported" if the 3x3 block directly beneath it
    contains any solid.  The estimate ignores the build platform and is meant
    as a coarse manufacturability heuristic only.
    """
    solid = np.asarray(vol, dtype=bool)
    if not solid.any():
        return 0.0
    above = solid[:, :, 1:]
    lower = solid[:, :, :-1]
    support = ndimage.binary_dilation(
        lower, structure=np.ones((3, 3, 1), dtype=bool))
    unsupported = above & (~support)
    return float(unsupported.sum()) / float((above | lower).sum())


def evaluate_validity(vol: np.ndarray, cfg: dict) -> tuple[bool, str]:
    """Full synthesizability check; returns (valid, reason)."""
    if compute_components(vol) != 1:
        return False, "connectivity"
    min_wall = int(cfg.get("min_wall_voxels", 2))
    max_thin = float(cfg.get("max_thin_fraction", 0.05))
    if thin_solid_voxels(vol, min_wall) > max_thin:
        return False, "min_wall_thickness"
    max_overhang = float(cfg.get("max_overhang_fraction", 1.0))
    if overhang_fraction(vol) > max_overhang:
        return False, "overhang"
    return True, "ok"
