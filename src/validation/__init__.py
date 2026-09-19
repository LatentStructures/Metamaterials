"""Synthesizability and physical validation package (ROADMAP section 2 / section 4.6).

Exposes synthesizability filters and checks for generated unit-cell microstructures.
An external pluggable filter can optionally define ``filter_generated(vol: np.ndarray) -> bool``
here to override the built-in heuristics in ``src.evaluate``.
"""
from __future__ import annotations

from src.evaluate import (
    compute_components,
    evaluate_validity,
    filter_connectivity,
    overhang_fraction,
    thin_solid_voxels,
)

__all__ = [
    "compute_components",
    "evaluate_validity",
    "filter_connectivity",
    "overhang_fraction",
    "thin_solid_voxels",
]
