"""Synthesizability and physical validation (ROADMAP section 2 / section 4.6).

Owns the hard, binary checks that decide whether a generated unit cell is
synthesizable: a single connected solid component, no 1-voxel-thin walls, and
a coarse additive-manufacturing overhang estimate. ``src.evaluate`` applies these
to every generated sample after FEA re-verification (ROADMAP 4.6, step 4).

**Periodic (wrap-around) boundary:** every structure in this project is a
*periodic unit cell* -- it is tessellated in all three directions, so solid that
exits one face re-enters the opposite one. Connectivity and wall thickness are
therefore evaluated under fully periodic boundaries. Without wrapping, a single
continuous network appears as several islands merely because the 32³ box cuts
it, which is why the uncorrected check scored the real gyroid cells at 9% while
they are in fact one connected phase that touches all six faces.

The 3x3x3 adjacency convention here matches the Option-B physics proxy in
``src/losses/physics_loss.py``, so the soft training signal and this hard filter
agree on corner-touching structures. This is a voxel-topology convention: an
edge- or corner-only contact does not provide a finite cross-section and is not
evidence of a robust load path. Passing this check is not a mechanical-strength
or manufacturability guarantee; wall checks and FEA are separate gates.

Optional override: define ``filter_generated(vol: np.ndarray) -> bool`` in this
module to replace the built-in heuristic during evaluation; ``src.evaluate``
uses it when present.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy import ndimage

__all__ = [
    "compute_components",
    "evaluate_validity",
    "filter_connectivity",
    "overhang_fraction",
    "safe_sample_name",
    "thin_solid_voxels",
]


def safe_sample_name(name: str, source: str) -> str:
    """Reject manifest entries that escape the samples directory (path traversal).

    Shared by ``src.evaluate`` and ``src.export.export_stl``, which both read
    sample filenames out of untrusted manifests.
    """
    p = Path(name)
    if p.is_absolute() or len(p.parts) != 1 or p.name in ("", ".", ".."):
        raise ValueError(
            f"{source} entry {name!r} must be a bare filename inside the samples directory"
        )
    return str(p)

# Face-neighbour offsets, used to union components that meet across a wrap.
_AXIS_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def _count_periodic_components(vol: np.ndarray) -> int:
    """Count 26-connected solid components, merging those joined across a wrap."""
    solid = np.asarray(vol, dtype=bool)
    if not solid.any():
        return 0
    labels, n = ndimage.label(solid, structure=_AXIS_STRUCTURE)
    parent = np.arange(n + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        shifted = np.roll(labels, offset, axis=(0, 1, 2))
        pairs = np.stack((labels[solid & (shifted > 0)],
                          shifted[solid & (shifted > 0)]), axis=1)
        if not pairs.size:
            continue
        for a, b in np.unique(pairs, axis=0):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
    return len({find(i) for i in range(1, n + 1)})

def compute_components(vol: np.ndarray) -> int:
    """Number of 26-connected solid components under periodic boundaries.

    The 3x3x3 structuring element matches the adjacency convention used by the
    Option-B physics proxy (``src/losses/physics_loss.py``), so the soft
    connectivity signal and the hard validity filter agree on corner-touching
    structures. This is a discrete topological convention; it does not certify
    finite contact area or load-bearing robustness. Components that meet across
    a wrapped face are merged, because the cell is tessellated rather than a
    closed standalone body.
    """
    solid = np.asarray(vol, dtype=bool)
    if not solid.any():
        return 0
    return _count_periodic_components(solid)


def filter_connectivity(vol: np.ndarray) -> bool:
    """True iff the solid phase is a single connected periodic component."""
    return compute_components(vol) == 1


def thin_solid_voxels(vol: np.ndarray, min_wall: int = 2) -> float:
    """Fraction of solid voxels removed by a periodic opening with a cube.

    A single opening strips features thinner than ``min_wall`` voxels, so a
    nonzero fraction flags 1-voxel-thin walls. The opening is evaluated on a
    padded-and-wrapped copy so members that only appear thin because the cell is
    clipped at a face are not counted as defects.
    """
    solid = np.asarray(vol, dtype=bool)
    if not solid.any():
        return 0.0
    # Pad by at least the wall size: wrapping must not fold a thin member onto
    # itself, or a 1-voxel membrane would appear thick.
    pad = max(1, int(min_wall))
    wrapped = np.pad(solid, pad, mode="wrap")
    structure = np.ones((min_wall,) * 3, dtype=bool)
    opened = ndimage.binary_opening(wrapped, structure=structure)
    sl = tuple(slice(pad, -pad) for _ in range(3))
    removed = solid.sum() - opened[sl].sum()
    return float(max(0, removed)) / float(solid.sum())


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
