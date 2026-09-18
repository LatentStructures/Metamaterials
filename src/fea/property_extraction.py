"""Effective-property extraction from a Voigt 6x6 stiffness (ROADMAP 3.3).

Converts the full effective stiffness tensor of a unit cell into the two scalar
baseline conditioning targets -- effective Young's modulus ``E`` and Poisson's
ratio ``nu`` (shear modulus ``G`` is computed only as an auxiliary metric and is
never part of the conditioning vector, per ROADMAP Phase 1 Section 3.3 / Phase 2
Section 4.1).

Families are (weakly) anisotropic, so the tensor is reduced through the cubic
symmetry averages of its 6x6 Voigt representation:

    C11 = mean(diag 0..2),   C12 = mean(off-diagonals of the 3x3 block),
    C44 = mean(diag 3..5),

and Voigt (Hill) averaging is used for the isotropic engineering constants:

    GV = (C11 - C12 + 3 C44) / 5
    GR = 5 C44 (C11 - C12) / (4 C44 + 3 (C11 - C12))
    G  = (GV + GR) / 2,   K = (C11 + 2 C12) / 3
    E  = 9 K G / (3 K + G),   nu = (3 K - 2 G) / (2 (3 K + G))

All quantitative checks -- positive-definiteness, physical bounds on E and nu --
are enforced here so callers get a hard failure instead of silently unphysical
labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Voigt index order 11, 22, 33, 23, 13, 12 (engineering shear strain).
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class EffectiveProperties:
    """Isotropic engineering constants reduced from a (near) cubic unit cell."""

    E: float
    nu: float
    G: float
    K: float
    C11: float
    C12: float
    C44: float


def voigt_isotropic(E: float, nu: float) -> np.ndarray:
    """Full 6x6 isotropic stiffness (engineering Voigt) for (E, nu)."""
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError(f"E must be positive and finite, got {E}")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must lie in (-1, 0.5), got {nu}")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C6 = np.zeros((6, 6))
    C6[:3, :3] = lam
    for i in range(3):
        C6[i, i] = lam + 2.0 * mu
    for i in range(3, 6):
        C6[i, i] = mu
    return C6


def cubic_averages(C6: np.ndarray) -> tuple[float, float, float]:
    """Reduce a 6x6 (engineering Voigt) stiffness to cubic constants (C11, C12, C44)."""
    C6 = np.asarray(C6, dtype=float)
    C11 = float(np.mean([C6[i, i] for i in range(3)]))
    off = [C6[i, j] for i in range(3) for j in range(3) if i != j]
    C12 = float(np.mean(off))
    C44 = float(np.mean([C6[i, i] for i in range(3, 6)]))
    return C11, C12, C44


def effective_properties(C6: np.ndarray) -> EffectiveProperties:
    """Effective (E, nu, G, K) from a valid 6x6 stiffness; raises on non-physical input."""
    C6 = np.asarray(C6, dtype=float)
    if C6.shape != (6, 6):
        raise ValueError(f"expected a (6, 6) stiffness, got {C6.shape}")
    if not np.all(np.isfinite(C6)):
        raise ValueError("stiffness contains NaN/inf and is unusable as a label")
    sym_err = np.abs(C6 - C6.T).max()
    if sym_err > 1e-6 * max(1.0, np.abs(C6).max()):
        raise ValueError(f"stiffness is not symmetric (max off-sym {sym_err:.3e})")

    eig = np.linalg.eigvalsh(np.asarray(C6))
    if eig.min() <= 1e-12 * max(1.0, np.abs(eig).max()):
        raise ValueError("stiffness is not positive-definite; unphysical unit cell")

    C11, C12, C44 = cubic_averages(C6)
    K = (C11 + 2.0 * C12) / 3.0
    GV = (C11 - C12 + 3.0 * C44) / 5.0
    denom_vr = 4.0 * C44 + 3.0 * (C11 - C12)
    if denom_vr <= 0.0:
        raise ValueError(f"no valid Hill shear lower bound (got {denom_vr:.3e})")
    GR = 5.0 * C44 * (C11 - C12) / denom_vr
    G = 0.5 * (GV + GR)

    denom = 3.0 * K + G
    if not (K > 0.0 and G > 0.0 and denom != 0.0):
        raise ValueError(f"non-physical modulus (K={K:.3e}, G={G:.3e})")
    E = 9.0 * K * G / denom
    nu = (3.0 * K - 2.0 * G) / (2.0 * denom)
    if not (E > 0.0 and -1.0 < nu < 0.5):
        raise ValueError(f"non-physical isotropic constants (E={E:.3e}, nu={nu:.3e})")

    return EffectiveProperties(E=float(E), nu=float(nu), G=float(G), K=float(K),
                               C11=C11, C12=C12, C44=C44)