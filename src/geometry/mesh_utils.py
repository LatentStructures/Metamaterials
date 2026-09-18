"""Voxel -> mesh utilities (ROADMAP section 3.2 items 3-5).

``voxels_to_struct_tetra`` is the primary FEA input: a structured voxel mesh on
the FULL (res+1)^3 node lattice with one 6-tet patch per voxel (solid AND void).
Void patches carry a ``void`` cell flag so the homogenizer assigns them ~zero
stiffness -- exactly the voxel-based homogenization discretization of Andreassen
& Andreasen (2014), which ROADMAP section 3.2 item 5 explicitly permits.

Because the node lattice is complete, the two faces of each axis carry a *full*
(res+1)^2 node set, so the PBC pairing is exact by lattice index, independent of
the sampled geometry. The FEA PBC constraint maps each +face node to its -face
partner via ``periodic_pairing``.

``voxels_to_surface`` computes a smooth marching-cubes surface for rendering /
inspection only, NOT for FEA (periodic cells are not closed standalone bodies).
"""
from __future__ import annotations

import meshio
import numpy as np

# 6-tet tessellation of a unit cube as (corner-id, corner-id, ...) with corners
# 0..7 = bit pattern (x + 2y + 4z); every tet has positive orientation.
_TET_TABLE = np.array(
    [
        [0, 1, 3, 7],
        [0, 2, 3, 7],
        [0, 1, 5, 7],
        [0, 5, 6, 7],
        [0, 2, 6, 7],
        [0, 4, 6, 7],
    ],
    dtype=np.int64,
)


# --------------------------------------------------------------------------- #
# Structured voxel tetrahedral mesh (primary FEA input)
# --------------------------------------------------------------------------- #
def voxels_to_struct_tetra(voxels: np.ndarray) -> "meshio.Mesh":
    """Structured tet mesh of the full (res+1)^3 voxel lattice in [0, 1]^3.

    Every voxel (solid or void) becomes 6 conforming tets; void tets are flagged
    in ``mesh.cell_data['void']`` for the homogenizer to give ~zero stiffness.
    Returns ``meshio.Mesh`` with cells ``tetra`` and per-cell int8 ``void``.
    """
    voxels = np.asarray(voxels)
    res = voxels.shape[0]
    solid = voxels > 0
    n = res + 1

    # Full node lattice; node index along axis dims is i + j*n + k*n^2.
    pts = np.stack(np.unravel_index(np.arange(n**3), (n, n, n)), axis=1)
    pts = (pts / res).astype(np.float64)
    nid = np.arange(n**3, dtype=np.int64).reshape(n, n, n)

    # Per-voxel corner ids via broadcasting on the lattice grid.
    ix = np.arange(res, dtype=np.int64)
    corners = np.stack(np.meshgrid(ix, ix, ix, indexing="ij"), axis=-1).reshape(-1, 3)
    all_nodes = np.zeros((res**3, 8), dtype=np.int64)
    for b in range(8):
        dx, dy, dz = ((b // 4) % 2, (b // 2) % 2, b % 2)
        idx = corners + np.array([dx, dy, dz])
        all_nodes[:, b] = nid[idx[:, 0], idx[:, 1], idx[:, 2]]
    cells = all_nodes[:, _TET_TABLE].reshape(-1, 4)  # (V, 6, 4) -> (6V, 4)

    # Orientation fix (all tets positive).
    v = pts
    signed = np.einsum(
        "ij,ij->i",
        v[cells[:, 1]] - v[cells[:, 0]],
        np.cross(v[cells[:, 2]] - v[cells[:, 0]], v[cells[:, 3]] - v[cells[:, 0]]),
    )
    flip = signed < 0
    if flip.any():
        cells[flip, 1], cells[flip, 2] = cells[flip, 2].copy(), cells[flip, 1].copy()

    void_flag = np.zeros(len(cells), dtype=np.int8)
    # void tets correspond to void voxels (every voxel -> 6 tets in order).
    vox_is_solid = solid.reshape(-1)  # res^3 voxels in same order as corners
    void_flag[np.repeat(~vox_is_solid, 6)] = 1

    mesh = meshio.Mesh(points=pts, cells=[("tetra", cells)])
    mesh.cell_data["void"] = [void_flag]
    mesh.cell_data["solid"] = [np.logical_not(void_flag).astype(np.int8)]
    return mesh


# --------------------------------------------------------------------------- #
# Periodic boundary pairing
# --------------------------------------------------------------------------- #
def periodic_pairing(tet_mesh: "meshio.Mesh", resolution: int):
    """Exact PBC pairing tables for the structured voxel mesh.

    Returns dict ``{axis: (ids_lo, ids_hi)}``: local node ids on the ``axis=0``
    and ``axis=1`` faces, paired 1:1 by lattice index (i.e. identical in-plane
    coordinates). Because the mesh uses the full (res+1)^3 lattice, every +face
    node has exactly one -face partner -- no projection, no mismatch.
    """
    pts = tet_mesh.points
    n = resolution + 1
    pairing: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ax in range(3):
        on_lo = np.flatnonzero(np.isclose(pts[:, ax], 0.0))
        on_hi = np.flatnonzero(np.isclose(pts[:, ax], 1.0))
        # Pair by index within the lattice: node (0, j, k) <-> node (n-1, j, k).
        key = lambda p: tuple(round(float(c), 9) for c in p[[j for j in range(3) if j != ax]])
        klo = {key(pts[i]): i for i in on_lo}
        pairs_lo = []
        pairs_hi = []
        for i in on_hi:
            k = key(pts[i])
            j = klo[k]
            if j is None:
                raise RuntimeError(
                    f"PBC pairing on axis {ax}: lattice misalignment ({k} not on "
                    "-face); this must never happen with the full-lattice mesh."
                )
            pairs_lo.append(int(j))
            pairs_hi.append(int(i))
        pairing[ax] = (np.asarray(sorted(pairs_lo)), np.asarray(sorted(pairs_hi)))
    return pairing


# --------------------------------------------------------------------------- #
# Smoothed surface (rendering / inspection only)
# --------------------------------------------------------------------------- #
def voxels_to_surface(voxels: np.ndarray, smooth: bool = True) -> "meshio.Mesh":
    """Smooth marching-cubes surface mesh of the solid phase (not for FEA).

    Periodic cells are never closed standalone bodies (they intersect the cell
    faces); this surface is for inspection/rendering/logging only.
    """
    from scipy import ndimage
    from skimage.measure import marching_cubes

    voxels = np.asarray(voxels)
    res = voxels.shape[0]
    binv = (voxels > 0).astype(np.int8)
    if binv.all() or not binv.any():
        raise RuntimeError("refusing to surface a fully solid / fully void voxel grid")
    if smooth:
        dist_in = ndimage.distance_transform_edt(binv)
        dist_out = ndimage.distance_transform_edt(1 - binv)
        field = np.where(binv > 0, dist_in, -dist_out)
        level = 0.0
    else:
        field = binv.astype(float)
        level = 0.5
    verts, faces, _, _ = marching_cubes(field, level=level, spacing=(1.0, 1.0, 1.0))
    verts = verts / res
    return meshio.Mesh(points=verts.astype(np.float64),
                       cells=[("triangle", faces.astype(np.int64))])


def voxels_to_tetra(voxels: np.ndarray) -> "meshio.Mesh":
    """Primary voxel -> tetra mesh entry point (structured, PBC-exact)."""
    return voxels_to_struct_tetra(voxels)