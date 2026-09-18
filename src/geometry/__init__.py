"""Phase 1 geometry: parametric unit-cell generators, periodic voxelization,
and the voxel -> surface -> tetra mesh pipeline (ROADMAP section 3.1-3.2)."""
from src.geometry.lattice_families import (
    FAMILIES,
    BAND,
    generate_voxels,
    parameter_bounds,
    sample_for_density,
)
from src.geometry.mesh_utils import (
    periodic_pairing,
    voxels_to_struct_tetra,
    voxels_to_surface,
    voxels_to_tetra,
)
from src.geometry.voxelize import relative_density

__all__ = [
    "FAMILIES",
    "BAND",
    "generate_voxels",
    "sample_for_density",
    "parameter_bounds",
    "relative_density",
    "voxels_to_surface",
    "voxels_to_struct_tetra",
    "voxels_to_tetra",
]