# Physics-informed loss: exact functional form (ROADMAP Phase 2, Section 4.4)

**Status: implemented (baseline, Option B).** `src/losses/physics_loss.py`
provides the differentiable connectivity proxy described here; this file is the
binding description of the exact form and MUST stay in sync with that module.
`src/train.py` applies it (weighted by `physics_loss.lambda_1`) on the soft,
pre-threshold denoised occupancy. True PDE-residual injection is deferred to
the research track.

## Total loss

```text
L_total = L_diffusion + lambda_1(t) * L_physics_proxy
```

- `L_diffusion` is the standard ε-MSE objective from `src/models/diffusion.py`
  (Ho et al., 2020), computed on the predicted noise at the sampled timestep.
- `L_physics_proxy` is evaluated on the **soft, pre-threshold denoised occupancy**
  `p = clamp(x0_pred, 0, 1)`, shape `(B, 1, 32, 32, 32)`.
- `lambda_1(t)` is warmed up linearly from `0` to the configured value over the
  first `warmup_fraction` (default `0.2`) of total training steps, so the model
  learns basic geometry before physics constraints act. Grid resolution and all
  weights come from config only (`configs/train_baseline.yaml`,
  `physics_loss.*`); nothing is hardcoded.

## Option B: soft connectivity proxy (baseline)

The synthesizability filter (Phase 2, Section 4.6) rejects structures with
multiple disconnected solid components. The proxy below is the differentiable
relaxation of that binary connected-component check, applied to `p` instead of
the thresholded grid.

Surface an empty/isolated voxel, and treating zero values as a disconnected
cluster, define the weighted 26-neighbourhood graph over grid nodes:

```text
A_ij = p_i * p_j          for all pairs (i, j) within the 26-neighbour cube
D_ii = sum_j A_ij         (degree; A is symmetric, D diagonal)
L    = I - D^{-1/2} A D^{-1/2}      (normalized graph Laplacian)
```

The **Fiedler value** `lambda_2(L)` is the second-smallest eigenvalue; it
approximates the soft algebraic connectivity. A structure with a floating
cluster detached from the bulk has `lambda_2 -> 0`; a single well-connected
solid has `lambda_2` comfortably away from zero.

`lambda_2` of any large grid scales like the lattice spectral gap (~n**-2), so
an absolute value cannot tell a solid cube from a detached cube on 32^3.  The
proxy is therefore *normalized* by the Fiedler value `lambda_ref` of the fully
solid grid at the same resolution:

```text
L_conn = (1 - min(lambda_2_hat / lambda_ref, 1))^2
```

`lambda_2_hat` is a differentiable estimate obtained by `t` iterations of
deflated power iteration on the walk matrix `M = I - L = D^{-1/2} A D^{-1/2}`
(Lanczos-free, autograd-safe): the trivial top eigenvector `D^{1/2}`
(eigenvalue 1) is deflated, so the Rayleigh quotient converges to
`1 - lambda_2(L)`:

```text
v0 = random unit vector, deflated against u = D^{1/2} / ||D^{1/2}||
for k in 1..t:  v = M v ; deflate against u ; normalize
lambda_2_hat = 1 - (v, M v)        (v unit)
```

Each disconnected component of the graph contributes its own eigenvalue-1
direction of `M`, so multi-component (floating-cluster) samples converge to
`lambda_2_hat = 0` quickly -- they score `L_conn ~ 1`.  `lambda_ref` is cached
per resolution (computed with `t_ref = 200` iterations under `no_grad`); the
default `t = 200`, configured the same way as `lambda_ref`.  For a fully empty
grid take `L_conn = 1` (maximally disconnected); a connected-but-floating
single component also scores near 0 (it passes the binary validity filter).
This term protects the validity rate; it does not steer properties toward the
conditioning targets — that weakness is exactly what Option A (a frozen
surrogate critic, contingency only) fixes.

Reference implementation already exists for the *hard* version in
`src/evaluate.compute_components` (binary 26-connected labeling); the two share
the adjacency convention.

## Option A: frozen surrogate critic (contingency, out of baseline scope)

Documented for future reference only. Train a small 3D CNN regressor mapping a
voxel grid to `[E, rho, nu]` on the Phase 1 dataset (a day of work), freeze it,
and during diffusion training backprop through `denoised x0 -> surrogate`:

```text
L_surrogate = w_E * ||E_sur(p) - E_tgt||^2 + w_rho * ||rho_sur(p) - rho_tgt||^2
              + w_nu * ||nu_sur(p) - nu_tgt||^2
```

Grid it under `physics_loss.option: A` if it is ever implemented.

## Ablation contract (Phase 2, Section 4.4)

Train the pair `{lambda_1: 0}` vs `{lambda_1: 0.05}` (config keys under
`physics_loss.ablation`) and log both to W&B under `tag: abl_physics`; the
default `0.05` value is only a starting point, to be tuned during the ablation.