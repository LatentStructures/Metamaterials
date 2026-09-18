"""Baseline Option-B physics proxy: differentiable soft-connectivity loss.

Interface (used by ``src/train.py``)::

    connectivity_proxy_loss(x, cond=None, t=None) -> scalar Tensor

returning ``L_physics`` to be weighted by ``lambda_1(t)`` and added to
``L_diffusion``.  The exact functional form is the binding spec in
``src/losses/README.md`` and MUST stay in sync with this file.

Operates on the soft, pre-threshold occupancy ``p = clamp(x, 0, 1)`` (in
training, ``x`` is the model's predicted ``x0``).  A binary connected-component
count is not differentiable, so ``L_physics`` is a *normalized* soft
algebraic-connectivity deficit of the weighted 26-neighbourhood graph::

    A_ij = p_i p_j,   D_ii = sum_j A_ij,   L = I - D^{-1/2} A D^{-1/2}
    lambda_2_hat = 1 - Rayleigh(M)        (deflated power iteration on
                                          M = D^{-1/2} A D^{-1/2} = I - L)
    L_physics = (1 - min(lambda_2_hat / lambda_ref, 1))^2

``lambda_ref`` is the Fiedler value of the *fully solid* grid at the same
resolution -- without this normalization ``lambda_2`` of any large grid is tiny
(it scales like the lattice spectral gap ~ n**-2), so the raw gap cannot tell a
solid cube from a detached cube on 32^3; the ratio can.  The penalty is ~0 for a
well-connected structure and ~1 for a detached/floating one.  A perfectly empty
grid is maximally disconnected (``L_physics = 1``).
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn

__all__ = ["connectivity_proxy_loss"]

_REFERENCE_ITERATIONS = 200


def _unit_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize a (B, 1, D, H, W) tensor to unit L2 norm per sample."""
    return x / x.reshape(x.shape[0], -1).norm(dim=1).view(
        x.shape[0], 1, 1, 1, 1).clamp_min(1e-8)


def _deflate(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Orthogonalize ``v`` against the trivial eigenvector ``u`` (per sample)."""
    return v - u * (u * v).sum(dim=(1, 2, 3, 4), keepdim=True)


def _soft_laplacian_action(p: torch.Tensor, v: torch.Tensor,
                           D: torch.Tensor | None = None) -> torch.Tensor:
    """Return (L_norm v) for the weighted 26-neighbour Laplacian of ``p``.

    Computed via 3x3x3 correlation without ever materialising the N x N matrix:
    ``(A w)_i = p_i (conv(p * w) - p_i w_i)`` and ``L v = v - D^{-1/2} A D^{-1/2} v``.
    """
    ones = torch.ones(1, 1, 3, 3, 3, device=p.device, dtype=p.dtype)
    dinv = (D + 1e-8).rsqrt() if D is not None else torch.ones_like(p)
    w = dinv * v
    pw = p * w
    Aw = p * (nn.functional.conv3d(pw, ones, padding=1) - pw)
    return v - dinv * Aw


def _fiedler_estimate(p: torch.Tensor, iterations: int = 60) -> torch.Tensor:
    """Per-sample Fiedler estimate ``lambda_2(L)`` of the soft graph (B,).

    Runs deflated power iteration on ``M = I - L = D^{-1/2} A D^{-1/2}``: the
    trivial top eigenvector ``D^{1/2}`` (eigenvalue 1) is deflated, so the
    iterate converges to the next eigenvalue of ``M``, which is ``1 - lambda_2(L)``.
    Each *disconnected component* of the graph contributes its own eigenvalue-1
    direction of ``M``, so multi-component (floating-cluster) samples converge
    to ``lambda_2(L) = 0`` quickly.  Empty samples return 0.
    """
    B = p.shape[0]
    s = nn.functional.conv3d(p, torch.ones(1, 1, 3, 3, 3, device=p.device, dtype=p.dtype),
                             padding=1)
    D = (p * s - p * p).clamp(min=0.0)

    dsum = D.reshape(B, -1).sum(dim=1)
    good = dsum > 1e-12
    if not bool(good.any()):
        return torch.zeros(B, device=p.device, dtype=p.dtype)

    mask = good.to(p.dtype).view(B, 1, 1, 1, 1)
    Dg = D * mask
    dinv = (Dg + 1e-8).rsqrt()
    u = Dg * dinv
    u = u / (u.reshape(B, -1).norm(dim=1).view(B, 1, 1, 1, 1) + 1e-8)

    ones = torch.ones(1, 1, 3, 3, 3, device=p.device, dtype=p.dtype)

    def mv(s):
        """M s = D^{-1/2} A D^{-1/2} s for the weighted 26-neighbour adjacency."""
        w = dinv * s
        pw = p * w
        aw = p * (nn.functional.conv3d(pw, ones, padding=1) - pw)
        return dinv * aw

    v = torch.randn_like(p)
    v = _unit_norm(_deflate(v, u))
    for _ in range(max(1, iterations)):
        Mv = mv(v)
        v = _unit_norm(_deflate(Mv, u))

    Mv = mv(v)
    rq = (v * Mv).sum(dim=(1, 2, 3, 4))  # v normalized
    lam2 = 1.0 - rq
    return torch.where(good, lam2, torch.zeros_like(lam2))


@lru_cache(maxsize=8)
def _reference_fiedler(spatial_shape: tuple[int, int, int]) -> float:
    """Fiedler value of the fully solid grid at ``spatial_shape`` (detached scalar).

    Cached per resolution: this is the connectivity ceiling against which the
    loss is normalized, so well-connected samples score ~0.
    """
    p = torch.ones(1, 1, *spatial_shape, dtype=torch.float32)
    with torch.no_grad():
        lam = _fiedler_estimate(p, iterations=_REFERENCE_ITERATIONS)
    return float(lam[0])


def connectivity_proxy_loss(
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    iterations: int = 200,
    normalize: bool = True,
) -> torch.Tensor:
    """Normalized soft-connectivity deficit of occupancy ``x`` (scalar).

    ``cond`` and ``t`` are accepted for compatibility with the training-loop
    call signature but do not influence Option B (the baseline proxy is
    unconditional -- it only protects the validity rate).
    """
    if x.dim() == 4:  # (B, D, H, W)
        x = x.unsqueeze(1)
    spatial = tuple(int(d) for d in x.shape[2:])
    p = x.clamp(0.0, 1.0)
    if p.shape[0] == 0:
        raise ValueError("connectivity_proxy_loss received an empty batch")

    lam2 = _fiedler_estimate(p, iterations=iterations)
    lam_ref = _reference_fiedler(spatial) if normalize else 1.0
    if lam_ref <= 0.0:
        lam_ref = 1.0  # degenerate resolution guard; raw squared gap below
    ratio = (lam2 / lam_ref).clamp(min=0.0, max=1.0)
    deficit = (1.0 - ratio).square() if normalize else (1.0 - lam2).square()
    return deficit.mean()