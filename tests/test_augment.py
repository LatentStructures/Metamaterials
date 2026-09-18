"""Augmentation tests: 24 proper rotations; physical exactness of the tensor.

The whole point of this module is that rotating a sample is *physically exact*:
the strain energy of the rotated stiffness must equal that of the original for
the rotated strain. Any silent tensor bug shows up here.
"""
import numpy as np

from src.data.augment import (
    proper_rotations,
    rotate_voxels,
    rotate_stiffness,
    _voigt_to_tensor,
    _tensor_to_voigt,
)


def test_exactly_24_proper_rotations():
    rots = proper_rotations()
    assert len(rots) == 24
    for R in rots:
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_of_rotation_is_identity():
    grid = np.arange(4**3).reshape(4, 4, 4)
    for R in proper_rotations():
        back = rotate_voxels(rotate_voxels(grid, R), R.T)
        assert np.array_equal(back, grid)


def test_stiffness_rotation_preserves_energy():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    C6 = 0.5 * (A + A.T) + 4.0 * np.eye(6)  # SPD, anisotropic
    for R in proper_rotations():
        for _ in range(5):
            e = rng.normal(size=6)
            et = _voigt_to_tensor(e)
            etr = (R @ et @ R.T)
            er = _tensor_to_voigt(etr)
            C6r = rotate_stiffness(C6, R)
            e0 = e @ C6 @ e
            e1 = er @ C6r @ er
            assert abs(e0 - e1) <= 1e-8 * abs(e0), (e0, e1)
        # rotating the fully-cubic tensor corresponding to isotropic elastic
        # moduli must return itself exactly (cubic symmetry -> invariance).
        E, nu = 1.5, 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C_iso = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                C_iso[i, j] = lam if i != j else lam + 2 * mu
        for i in range(3, 6):
            C_iso[i, i] = mu
        assert np.allclose(rotate_stiffness(C_iso, R), C_iso)