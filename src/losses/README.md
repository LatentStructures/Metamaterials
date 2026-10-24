# Physics-informed loss: exact functional form (ROADMAP Phase 2, Section 4.4)

**Status: implemented (baseline, Option B).** `src/losses/physics_loss.py`
provides the differentiable connectivity proxy described here; this file is the
binding description of the exact form and MUST stay in sync with that module.
`src/train.py` applies it (weighted by `physics_loss.lambda_1`) on the soft,
pre-threshold denoised occupancy. True PDE-residual injection is deferred to
the research track.

Training runs log `train/loss_diffusion` (the ε-MSE term) and
`train/loss_physics` (the warm-up-weighted Option-B contribution) separately;
their sum equals `train/loss` up to floating-point rounding. The physics tag is
zero for a λ₁=0 ablation. Existing run event files predate this instrumentation
and cannot be decomposed retroactively.

## Total loss

```text
L_total = L_diffusion + lambda_1(t) * L_physics_proxy
```

- `L_diffusion` is the standard ε-MSE objective from `src/models/diffusion.py`
  (Ho et al., 2020), computed on the predicted noise at the sampled timestep.
- `L_physics_proxy` is evaluated on the **soft, pre-threshold denoised occupancy**
  `p = clamp(x0_pred, 0, 1)`, shape `(B, 1, 32, 32, 32)`.
- `lambda_1(t)` is warmed up linearly from `0` to the configured value over the
  first `warmup_fraction` of total training steps (set to `0.2` in
  `configs/train_baseline.yaml`; the code falls back to `0.0` — no warmup — if
  the key is absent), so the model
  learns basic geometry before physics constraints act. Grid resolution and all
  weights come from config only (`configs/train_baseline.yaml`,
  `physics_loss.*`); nothing is hardcoded.

## Option B: soft connectivity proxy (baseline)

The synthesizability filter (Phase 2, Section 4.6) rejects structures with
multiple disconnected solid components. The proxy below is a differentiable
soft-graph regularizer inspired by that binary connected-component check,
applied to `p` instead of the thresholded grid. It is not mathematically
equivalent to the hard filter and does not guarantee that the thresholded
sample is valid.

Treating empty and isolated voxels as disconnected clusters, define the
weighted 26-neighbourhood graph over the grid nodes with **periodic unit-cell
boundaries**. Every voxel sees wrapped neighbours at opposite faces, matching
the periodic hard filter and homogenization setup:

```text
A_ij = p_i * p_j          for periodic pairs within the 26-neighbour cube
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
per resolution (computed with `t_ref = 200` iterations under `no_grad`, where
autograd depth is irrelevant); the live-graph budget defaults to `t = 60`
(power iterations inside the autograd graph: fewer keep the backprop graph
shallow), configurable under `physics_loss.iterations` in the train config.
For a fully empty grid take `L_conn = 1` (maximally disconnected); a
connected-but-floating single component also scores near 0 (it passes the
binary connectivity filter).
The neighbor sum uses a circular halo and `avg_pool3d`, exactly equivalent to
a periodic convolution with a 3×3×3 all-ones kernel. There are no zero-padding
boundary nodes in the graph.

**Relaxation limitation (measured, 2026-10-21).** On the 32³ sanity pattern in
`scripts/check_physics_proxy_cases.py`, two binary-isolated blocks score 1.0,
but setting the background to soft occupancy 0.05 connects them in the weighted
graph and reduces the loss to 0.0229. This is expected for a soft graph: even
small positive values create edges. The 26-neighbor convention also treats
edge/corner-only voxel contacts as connected, although they have no finite
cross-section. The proxy does not enforce minimum wall thickness or robust
periodic load-bearing paths. Keep the thresholded hard filter, wall checks, and
FEA as independent checks; do not report this loss as a validity guarantee.

**Implementation correction (2026-10-21):** an earlier version used zero
padding, which did not match the periodic unit-cell boundary. The earlier
physics-on training run and its ablation therefore measure that zero-padded
proxy. The historical physics-off checkpoint is unchanged by this correction,
though it also predates the epoch-varying augmentation fix. Do not transfer the
old physics-on mechanism claim to the periodic proxy until it has been retrained
and evaluated.

This term was designed on the hypothesis that it protects the validity rate; it
does not steer properties toward the conditioning targets — that weakness is
exactly what Option A (a frozen surrogate critic, contingency only) fixes.

**Measured outcome for the legacy zero-padded proxy (λ₁ = 0 ablation, 30k
iters, 50 held-out targets, same seed): the term did not protect validity.** The
control matched or beat that physics-on
run on every metric — validity 94.0% vs 92.0%, E error 2.44% vs 3.63%, ρ 1.30% vs
1.51%, ν 0.56% vs 0.99%, final loss 0.0033 vs 0.0118 — and the physics-on arm
showed the gradient bursts that made `optimizer.grad_clip` necessary (pre-clip
norms up to 4,229 post-warmup, vs 1.31 without the term). The likely mechanism is
that the connectivity proxy rewards connectedness, and thin struts are the
cheapest way to stay connected at low density: paired across the same targets,
37/50 are thinner under physics-on than physics-off (Wilcoxon p = 0.0015). So the
term trades wall thickness for connectivity, which the existing `min_wall_voxels`
filter then rejects. Treat the claim above as the design intent, not a result;
see `PROJECT.md` §8 for the full comparison.

Reference implementation for the *hard* version is in
`src/validation.compute_components` (binary 26-connected labeling); both use
periodic 26-neighbour adjacency, with the proxy operating on soft occupancy.

## Option A: frozen surrogate critic (contingency, out of baseline scope)

Documented for future reference only. Train a small 3D CNN regressor mapping a
voxel grid to `[E, rho, nu]` on the Phase 1 dataset (a day of work), freeze it,
and during diffusion training backprop through `denoised x0 -> surrogate`:

```text
L_surrogate = w_E * ||E_sur(p) - E_tgt||^2 + w_rho * ||rho_sur(p) - rho_tgt||^2
              + w_nu * ||nu_sur(p) - nu_tgt||^2
```

Gate it under `physics_loss.option: A` if it is ever implemented.

## Ablation contract (Phase 2, Section 4.4)

Train the pair `{lambda_1: 0}` vs `{lambda_1: 0.05}` (config keys under
`physics_loss.ablation`) and log both under the `abl_physics` tag (W&B tags,
or the run's TensorBoard project); the default `0.05` value is only a starting
point, to be tuned during the ablation.
