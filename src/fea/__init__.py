"""Phase 1 FEA: periodic-boundary homogenization + effective-property extraction.

- ``homogenization``: FEniCSx strain-driven homogenization of voxel unit cells
  (ROADMAP section 3.3) + the closed-form analytical validation preflight.
- ``property_extraction``: cubic/Hill reduction of C to (E, nu, G, K).
- ``benchmarks``: closed-form cells (solid cube, laminate, aligned rods) and
  two-phase Voigt-Reuss bounds used as exact validation targets.
"""
from src.fea import benchmarks
from src.fea.homogenization import (
    Homogenizer,
    base_material,
    homogenize_periodic_cell,
    set_base_material,
    set_void_scale,
    validate_vs_analytical,
    void_scale,
)
from src.fea.property_extraction import (
    EffectiveProperties,
    cubic_averages,
    effective_properties,
    voigt_isotropic,
)
from src.voigt import VOIGT_PAIRS

__all__ = [
    "VOIGT_PAIRS",
    "EffectiveProperties",
    "Homogenizer",
    "base_material",
    "benchmarks",
    "cubic_averages",
    "effective_properties",
    "homogenize_periodic_cell",
    "set_base_material",
    "set_void_scale",
    "validate_vs_analytical",
    "void_scale",
    "voigt_isotropic",
]