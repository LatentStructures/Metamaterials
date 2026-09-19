"""Cubic symmetry augmentation (ROADMAP section 3.4).

Each unit cell admits up to 24 *proper* rotations -- the rotation group of the
cube (no mirror reflections are ever used, per the locked decision). Rotating a
voxel grid alone is not enough: the effective stiffness tensor must be rotated
to match, or labels silently corrupt. The rotation is implemented on the
fourth-order tensor so that strain energy is exactly invariant (verified in
tests).
"""
from __future__ import annotations

import itertools

import numpy as np

from src.voigt import VOIGT_PAIRS


def proper_rotations() -> list[np.ndarray]:
    """All 24 proper rotations of the cube (signed permutation matrices, det=+1)."""
    rots: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            R = np.zeros((3, 3))
            for out_col, in_row in enumerate(perm):
                R[in_row, out_col] = signs[out_col]
            if np.linalg.det(R) > 0:
                rots.append(R)
    return rots


_ROTATIONS = proper_rotations()


def rotate_voxels(voxels: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a proper rotation R (3x3 signed permutation) to a voxel grid.

    A signed permutation maps grid axes to grid axes with possible flips, so no
    interpolation is needed -- exact, invertible voxel remapping.
    """
    if voxels.ndim != 3 or voxels.shape[0] != voxels.shape[1] or voxels.shape[1] != voxels.shape[2]:
        raise ValueError(f"voxels must be a cubic uint8 grid, got {voxels.shape}")
    axes = np.zeros(3, dtype=int)
    flips = np.zeros(3, dtype=bool)
    for out_axis in range(3):
        for in_axis in range(3):
            v = R[out_axis, in_axis]
            if abs(v) == 1.0:
                axes[out_axis] = in_axis
                flips[out_axis] = v < 0
    out = np.transpose(voxels, axes=axes)
    for axis, flip in enumerate(flips):
        if flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def stiffness_to_full(C6: np.ndarray) -> np.ndarray:
    """Expand a Voigt 6x6 stiffness (engineering shear strain) to a 3x3x3x3 tensor.

    With the engineering-strain convention the mapping is identity on Voigt
    entries (the 2x in gamma cancels the symmetric-pair double count), see the
    tests for the energy-invariance check.
    """
    C6 = np.asarray(C6, dtype=float)
    C4 = np.zeros((3, 3, 3, 3))
    for i, (a, b) in enumerate(VOIGT_PAIRS):
        for j, (c, d) in enumerate(VOIGT_PAIRS):
            val = C6[i, j]
            for (p1, p2) in ((a, b), (b, a)):
                for (q1, q2) in ((c, d), (d, c)):
                    C4[p1, p2, q1, q2] = val
    return C4


def full_to_stiffness(C4: np.ndarray) -> np.ndarray:
    """Contract a 3x3x3x3 stiffness tensor back to Voigt 6x6 (engineering shear)."""
    C4 = np.asarray(C4, dtype=float)
    C6 = np.zeros((6, 6))
    for i, (a, b) in enumerate(VOIGT_PAIRS):
        for j, (c, d) in enumerate(VOIGT_PAIRS):
            C6[i, j] = C4[a, b, c, d]
    return C6


def rotate_stiffness(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate an effective stiffness tensor (Voigt 6x6) under a proper rotation R.

    C'_{ijkl} = R_ia R_jb R_kc R_ld C_abcd on the fourth-order tensor, then reduced
    back to Voigt. For cubic-symmetric entries this is exact and label-preserving.
    """
    C4 = stiffness_to_full(C6)
    C4r = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C4)
    return full_to_stiffness(C4r)


def random_rotation(rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw one of the 24 proper rotations uniformly."""
    rng = rng or np.random.default_rng()
    return _ROTATIONS[int(rng.integers(0, len(_ROTATIONS)))]


def _voigt_to_tensor(e_voigt: np.ndarray) -> np.ndarray:
    """Voigt engineering-strain vector (e11,e22,e33,g23,g13,g12) -> symmetric tensor."""
    e = np.asarray(e_voigt, dtype=float)
    t = np.zeros((3, 3))
    for i, (a, b) in enumerate(VOIGT_PAIRS):
        if a == b:
            t[a, a] = e[i]
        else:
            t[a, b] = t[b, a] = 0.5 * e[i]
    return t


def _tensor_to_voigt(t: np.ndarray) -> np.ndarray:
    """Symmetric strain tensor -> Voigt engineering-strain vector."""
    t = np.asarray(t, dtype=float)
    v = np.zeros(6)
    for i, (a, b) in enumerate(VOIGT_PAIRS):
        v[i] = t[a, a] if a == b else 2.0 * t[a, b]
    return v
