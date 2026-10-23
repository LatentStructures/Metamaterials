# FINDINGS — evidence appendix and historical notes

> **This is not the current status document.** Current results and limitations are in
> [`PROJECT.md`](PROJECT.md) §8; the binding scope and remaining work are in [`ROADMAP.md`](ROADMAP.md).
> This file keeps the detailed evidence and the dated implementation notes that are too long for those
> documents.

**Scope (2026-10-22).** The 2026-10-21 integrity audits of the data and the historical training runs
remain valid for those artifacts. A later code review found that the Option-B proxy used zero-padded
graph neighbours while the hard validity filter is periodic; the proxy is now periodic. Historical
physics-on results and the causal analysis in §1 therefore describe the **legacy zero-padded proxy
only**. The corrected-proxy matched runs are reported in `PROJECT.md` §8; they reproduce the outcome
(the λ₁=0 control leads) but not the legacy gradient mechanism.

---

## 1. Legacy zero-padded proxy: why it was retired

For the legacy implementation only, three questions were tested: whether λ₁ was mis-scaled, whether
the gradient path was pathological, and whether the proxy could even see the failures the filter
rejects.

### 1.1 Gradient forensics — heavy-tailed, not uniformly mis-scaled

Source: `train/grad_norm_preclip` and `train/loss` in `runs/baseline_clipfix_30k` (λ₁ = 0.05) and
`runs/baseline_physics_off` (λ₁ = 0), 601 logged points each, matched 1:1 by step.

| run | median | p75 | p90 | p99 | max | steps > 1.0 (clip bit) |
|---|---|---|---|---|---|---|
| physics-on | 0.1146 | 0.2051 | 0.3222 | **277.86** | **4,229.23** (iter 27,600) | 14 (2.33%) |
| physics-off | 0.1116 | 0.2018 | 0.3245 | 0.8688 | 5.90 (iter 1, warm-up) | 3 (0.50%) |

Per-phase (100 logged points per bucket):

| phase | med on | p99 on | max on | med off | p99 off | median ratio | p99 ratio |
|---|---|---|---|---|---|---|---|
| 0–5k | 0.191 | 1.02 | 5.90 | 0.213 | 0.97 | 0.90 | 1.05× |
| 5–10k | 0.164 | 20.87 | 1,307 | 0.148 | 0.48 | 1.11 | **43×** |
| 10–15k | 0.132 | 273.96 | 1,463 | 0.113 | 0.79 | 1.16 | **346×** |
| 15–20k | 0.097 | 0.50 | 0.95 | 0.095 | 0.99 | 1.02 | 0.51× |
| 20–25k | 0.062 | 1,030.89 | 1,577 | 0.058 | 1.01 | 1.08 | **1,025×** |
| 25–30k | 0.036 | 349.07 | 4,229 | 0.029 | 0.21 | 1.24 | **1,635×** |
| overall | 0.115 | 277.86 | 4,229 | 0.112 | 0.87 | **1.026** | **320×** |

* On 97.7% of steps the physics term adds essentially nothing to the gradient (median ratio ≈ 1.0;
  median absolute difference +0.0014 on a base of ~0.11).
* The tail is extreme: 5 steps above 1,000, peak 4,229 — bursts ~10⁴–10⁵× the contemporaneous
  diffusion gradient. Seven of the 14 burst steps land after 20k, when the cosine-annealed diffusion
  gradient is smallest.
* Loss does not spike when the gradient spikes: Pearson r(loss, grad) = −0.003 (all steps) / +0.006
  (excluding warm-up); burst-step loss is only 1.22× the running mean after removing the warm-up
  confound.
* `train/loss` includes λ·physics on the physics-on arm: final-100 means 0.01184 vs 0.00331 (ratio
  3.575). Both runs have zero non-finite values over all 601 points.

### 1.2 The clipping-conversion mechanism

`optimizer.grad_clip: 1.0` stopped the divergence, but it shaped *how* the physics term entered.
When a burst arrived, the pre-clip gradient was >99.99% pure physics direction (diffusion
contributed ~0.03–0.06 of norm 1,000–4,229). Rescaling the whole gradient to unit norm therefore did
not attenuate the burst: it converted each burst into a full-magnitude, pure-physics update that
displaced the diffusion step for that iteration. Seven of these landed in the annealed late phase
where surrounding updates were tiny.

This reconciles the observations: typical gradients match between arms, the loss shows no spike
response, and yet physics-off wins — the outcome difference is attributable to ~14 clipped
adversarial-scale updates rather than a persistent gradient offset. It is a legacy-proxy finding;
the corrected proxy's worst pre-clip norms are 37.2 / 23.2, not 4,229.

### 1.3 Proxy discrimination experiment — blind to thin walls

Source: in-memory experiment over `connectivity_proxy_loss` (`src/losses/physics_loss.py`, default
`iterations=60`, `normalize=True`, mean of 7 fixed-seed power-iteration inits per grid). Stimulus:
12 validity-passing cells per family (36 total, density-spanning) from `data/manifest.parquet`, each
in four variants: unmodified control; **fragmented** (two antipodal 2-voxel periodic slab cuts along
one axis — verified 36/36 exactly 2 components, `evaluate_validity` reason `connectivity`);
**thinned-gentle** (connectivity-guarded periodic erosion to thin-fraction ≥ 0.10, just past the 5%
validity gate — walls a mix of 1–2 voxels); **thinned-strict** (erosion to thin-fraction ≥ 0.95,
uniform 1-voxel walls). Both thin variants keep connectivity and fail only `min_wall_thickness`. A
fully solid 32³ gives proxy 0.000000 (floor).

Per-family proxy (mean ± std, n = 12):

| family | control | fragmented | thinned gentle | thinned strict |
|---|---|---|---|---|
| cubic_strut | 0.0277 ± 0.0233 | 0.0417 ± 0.0290 | 0.0297 ± 0.0330 | 0.0238 ± 0.0267 |
| octet_truss | 0.0000 ± 0.0001 | 0.0019 ± 0.0026 | 0.0000 ± 0.0001 | 0.0004 ± 0.0010 |
| gyroid | 0.0000 ± 0.0001 | 0.0259 ± 0.0212 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Paired contrasts across the 36 cells:

| contrast | mean | median | +/0/− | Wilcoxon p |
|---|---|---|---|---|
| fragmented − control | +0.0139 | +0.0103 | 28/5/3 | **3.3e-05** |
| thinned gentle − control | +0.0007 | 0.0000 | 6/16/14 | **0.35 (n.s.)** |
| thinned strict − control | −0.0012 | 0.0000 | 7/14/15 | **0.18 (n.s.)** |

Material cost of the thinning "cheap fix" (achieved density): gentle thinning costs only **−8.6%**
relative density at the gate boundary (−0.033 absolute); uniform 1-voxel walls cost −49%. "Hover
just past the 5% gate" is genuinely cheap — and completely unpenalized by the proxy.

Convergence diagnostic (fragmented deficit vs power-iteration budget, 6 cells at 60/150/400/1000
iterations): slab cuts score 0.00–0.07 at 60 iterations and only reach 0.35–0.95 at 400–1000. A
compact 2-island split scores 0.96 at just 60 iterations. The docstring's assumption that
multi-component samples converge to λ₂ = 0 quickly holds for compact islands but **not for slab-like
splits — the realistic failure geometry — at the 60-iteration training budget.**

### 1.4 Verdict

Option B **as configured** (λ₁ = 0.05, 60 power iterations) contributed no measurable median-scale
guidance and no protection against the filter's actual failure mode (thin walls), provided only a
~1.4%-of-range penalty against realistic fragmentation, and injected 14 adversarial clipped updates.
This is the mechanistic account of the legacy ablation. The *outcome* — physics-off matching or
beating physics-on — was reproduced on the corrected periodic proxy in both pipelines and on both
target sets (`PROJECT.md` §8), so the current verdict rests on matched corrected runs, not on the
legacy implementation.

---

## 2. Where validity actually fails — a low-density corner

Sources: per-sample evaluation CSVs — `outputs/acceptance_30k/evaluate_live/` (50 on-manifold,
physics-on, 92.0%), `outputs/synthetic_30k/evaluate/` (108 off-manifold, physics-on, 97.2%),
`outputs/synthetic_30k/evaluate_k5/` (540, 5 draws, 92.2%), `outputs/ablation_30k/evaluate/` and
`outputs/ablation_30k/evaluate_synthetic/` (physics-off: 94.0% / 98.1%).

### 2.1 Failure reasons

| set | fails | min_wall_thickness | connectivity | overhang |
|---|---|---|---|---|
| on-manifold ON (50) | 4 | 4 | 0 | 0 |
| off-manifold ON (108) | 3 | 2 | 1 | 0 |
| k5 (540) | 42 | 39 | 3 | 0 |
| on-manifold OFF (50) | 3 | 3 | 0 | 0 |
| off-manifold OFF (108) | 2 | 2 | 0 | 0 |

93%+ of failures are `min_wall_thickness`. The rare connectivity failures sit at ρ = 0.18–0.455 — a
separate, mid-density mode. No overhang failure anywhere.

### 2.2 Density is the dominant axis (pooled physics-on, n = 698)

| target ρ bin | n | fail | fail % |
|---|---|---|---|
| < 0.15 | 28 | 25 | **89.3%** |
| 0.15–0.25 | 119 | 17 | 14.3% |
| 0.25–0.40 | 306 | 6 | 2.0% |
| 0.40–0.55 | 228 | 1 | 0.4% |
| > 0.55 | 17 | 0 | 0.0% |

**ρ < 0.25 is 21.1% of samples but carries 42/49 = 85.7% of all failures. Above ρ = 0.25, pooled
validity is 98.7% (544/551).** The same monotone gradient holds inside every individual set and in
the physics-off arm.

### 2.3 The hard corners

* **Gyroid at ρ < 0.25** — all 4 legacy on-manifold failures are gyroid, ρ < 0.25, ν ≈ 0.30 (t14,
  t26, t34, t49; the ablation's 3 failures are a subset: t14, t26, t49). Every gyroid held-out
  target with ρ < 0.25 fails (4/4) while every non-gyroid ρ < 0.25 target passes (9/9); all 37
  targets with ρ ≥ 0.25 pass in both arms.
* **Low E at low ρ** — in the low-ρ half of the k5 set, low-E targets fail 24.4% vs 5.9% for high-E
  (4×; ν adds only a mild effect). Asking for stiffer-than-trend at low ρ fails more (19.3% vs 11.1%
  against the fitted log E–log ρ trend). The four targets that fail 0/5 draws are all octet-like, E
  ≤ 0.10, ρ = 0.132–0.239 (mean ρ 0.164 vs 0.367 for the 93 targets that succeed 5/5).
* Invalid k5 samples *undershoot* target density (mean −1.1 pp, max −5.3 pp, vs −0.05 pp for valid
  ones): wall failures are the tail of a continuous under-density bias, concentrated where walls are
  thinnest.
* The physics-off arm shares the same spatial pattern at slightly lower magnitude, so the weakness
  is a property of model/data, not of the physics loss.
* The corrected runs repeat the pattern: ρ ≥ 0.25 validity is 100% on the 50 held-out targets and
  97–99% off-manifold in every arm.

### 2.4 Verdict

The legacy 92% headline (95% CI [90.0, 94.5]) is not at risk from spread; it is a
**low-density-corner** number. On the held-out set (13/50 targets below ρ = 0.25) the ≥90% lock-in
bar holds. On a ρ < 0.25-heavy distribution it would not hold, so any validity figure should travel
with this conditioning.

---

## 3. Dataset limits: duplicates and physical floors

Sources: `data/manifest.parquet` (SHA-256 over grid bytes at `voxel_path`) plus a full re-run of the
committed filter (`evaluate_validity` with `configs/evaluate.yaml`: `min_wall_voxels: 2`,
`max_thin_fraction: 0.05`), reproducing the audit exactly (gyroid 197/800 fail, octet 99/800, cubic
0/800, all `min_wall_thickness`).

### 3.1 Duplicates are a resolution effect, worst for octet

| family | unique grids / 800 | duplicate rows | largest group | distinct achieved-ρ values |
|---|---|---|---|---|
| cubic_strut | 515 (64.4%) | 285 | 17 | 457 |
| gyroid | 368 (46.0%) | 432 | 12 | 174 |
| octet_truss | **140 (17.5%)** | **660** | **54** | **123** |

* Octet contributes 47.9% of all 1,377 duplicate rows; duplication is ≈100% across the entire
  (density, shape) domain. The largest plateaus: ρ = 0.2959 (one grid, 54 rows), 0.1191 (41), 0.3545
  (35), 0.1543 (34). At 32³ the rasterizer quantizes thin struts into ~123 densities regardless of
  where parameters are drawn.
* Integrity holds: 0 of 418 duplicate groups span more than one split, and within every group the
  `E_eff` spread is exactly 0 (byte-identical geometries, identical labels — pure loss
  re-weightings, not label noise).
* **Effective training set: 658 unique grids of 1,920 train rows**, with duplicates acting as
  up-to-54× loss weights concentrated at low-to-mid octet densities. Val/test are cleaner (183/243
  and 182/237 unique) and leak-free.
* The derived corrected dataset (`data/v1f/`) has 935 unique grids of 2,400 rows — fewer distinct
  geometries than v1's 1,023.

### 3.2 Gyroid has a hard, physically exact density floor

| achieved ρ | gyroid failure rate |
|---|---|
| < 0.20 | **100% (175/175)** |
| 0.20–0.22 | 47.7% |
| 0.22–0.24 | 25% |
| ≥ 0.24 | **0% (0/559)** |

The analytic density of a 2-voxel sheet at 32³ is ρ = 3.544 × 2/32 = **0.2215**, matching the
measured fail-maximum **0.2217** almost exactly. The filter is physically right; the [0.1, 0.6]
density band is too wide for a sheet family at the locked 32³ scale. **175 gyroid rows (7.3% of the
dataset) are guaranteed-invalid by construction.**

### 3.3 Octet floor and a marginal island

* 100% failure below ρ = 0.14 (65 cells; ~3.00-voxel diagonal struts), 0% from ρ = 0.15 (first
  all-pass plateau has 3.32-voxel struts). The analytic 2-voxel-diameter floor (ρ ≈ 0.07) does not
  apply because the axis-aligned 2³ opening cannot fit inside a diagonal cylinder thinner than ~3.46
  voxel diameters. Measured floor: **ρ ≈ 0.15**.
* A residual non-monotone island at ρ = 0.237–0.240 (34 rows, only 6 unique grids for all 99 octet
  failures) breaches the 0.05 thin budget marginally (0.061) and is rescued by fattened joints;
  raising `max_thin_fraction` to ~0.07 would admit it.
* Cubic passes universally at every density — confirming `min_wall_voxels: 2` is not over-strict as
  a threshold.

### 3.4 Band overshoots are cosmetic

All 35 out-of-band cells are *in-band targets* that rasterization pushed out (every low-edge target
∈ [0.1003, 0.1076], every high-edge target ∈ [0.5903, 0.5998]); max overshoot 0.0121 below 0.1 /
0.0143 above 0.6 — smaller than the worst in-band |achieved − target| miss anywhere (0.0293). No
action needed beyond disclosure; if desired, sample targets from [0.115, 0.585].

### 3.5 Dataset-v2 guidance (all compatible with the locked decisions)

1. **Family-specific density floors**: gyroid ≥ ~0.24, octet ≥ ~0.15 — deletes the ~240
   guaranteed-invalid rows and much of the duplicate-prone low-density mass at once.
2. **Dedup-aware sampling** (skip parameters whose rasterized grid hashes to an existing one) — the
   only lever that actually raises unique-grid count at 32³, since no parameter-space re-sampling
   can (§3.1).
3. Higher resolution would fix both effects but is a locked-out research-track decision (32³ is
   baseline).

The derived `data/v1f/` dataset implements a gate-only variant of the first lever (drops the 296
invalid rows, 100% valid) but not the density floors or dedup; those remain open. See
`data/v1f/data_card.md` and §5.2.

---

## 4. Historical implementation and audit notes

These are dated notes retained for provenance. They are not current status. The code and configs are
authoritative.

### 4.1 Source and pipeline fixes

| Area | Finding | Resolution |
|---|---|---|
| Legacy resume | A legacy checkpoint has no `sample_index`; the key stayed `None`, so the first `_save_checkpoint` (`int(None)`) or records sort raised `TypeError` | Backfill the index from the voxel filename on resume |
| Export manifest | A `source_dir` manifest column containing NaN became the truthy string `"nan"`, so candidates resolved against a nonexistent directory | `pd.notna` guard; manifest loaded once |
| Resolution check | `evaluate` checked only `shape[0]`, so a `(32,16,16)` grid passed and failed later inside FEA | Full cubic check `shape == (res, res, res)` |
| Worker threads | Thread-cap env vars are set after the spawned child imports numpy/torch, so caps bind only to libraries loaded afterwards | Comments name exactly what the cap binds to (dolfinx/PETSc/scipy) |
| Physics option | The `physics_loss.option` config key was never read — `option: A` would silently run Option B | Enforced in `_resolve_physics_lambda`; raises unless `"B"` |
| Dead code | `has_metadata`, `void_scale()`, unreachable sparse-LU branch, duplicated `_safe_sample_name` | Removed; shared `src.validation.safe_sample_name()` |
| Naming | `self.siLU` violated snake_case | Renamed `self.silu` (checkpoint-safe) |
| Seed comment | Claimed “each draw needs its own seed”; the mechanism is one seed per target, with draws as distinct rows of one batched `torch.randn` | Corrected in `src/sample.py` and `RUN-BASE.md` |
| Grad comment | Claimed pre-clip norms “up to ~18”; the measured peak was 4,229 | Corrected in `configs/train_baseline.yaml` |

### 4.2 STL-export bugs

1. **Inverted mesh winding** — a unit sphere surfaced at −4204.7 instead of the true +4188.8. Fixed
   in both branches of `voxels_to_surface`.
2. **Missing `merge_vertices()`** — open/fragmented surfaces. Added to `_trimesh_surface`.
3. **Wrong arm resolution** — `_resolve_grid` globbed `samples/` (EMA) before `samples_live/`,
   silently exporting the wrong arm. Now prefers `_live`.

Void padding plus `--tiles 2` produced 5/5 watertight STLs. At `--tiles 1` *without* padding, unit
cells that intersect their own faces are inherently open (0/5 watertight) — geometry, not a bug.

### 4.3 Augmentation, precision, clamp, and sampling checks

* **Augmentation (A1):** rotations were fixed per sample index across epochs. Production now applies
  the 24 proper rotations on GPU per batch with a fresh per-(seed, epoch, step) generator. Tests
  cover epoch diversity, determinism, exact equality with `rotate_voxels`, and voxel-count
  preservation.
* **Precision (A2):** the corrected-proxy fp32/bf16 probe paired eight power-iteration seeds on
  CUDA. Mean losses were exactly equal (0.2409067452), paired gradient relative RMS difference 0,
  and fp32 seed-to-seed variation 0.870. Precision rounding is not supported as an explanation; no
  autocast exclusion was added. Evidence: `outputs/physics_precision.json`.
* **Clamp range (A3):** same historical checkpoint, targets, live weights, and draw seeds; 540
  samples per arm. `[0,1]`: 525/540 valid (97.22%), mean absolute density error 0.00479. Legacy
  `[-1,1]`: 524/540 valid (97.04%), 0.00609. Paired arms differ by a mean 3,155 voxels (9.63%). The
  validity difference is inconclusive; density error favors `[0,1]`. A 12-pair FEA follow-up had
  mixed directions and is descriptive only. Evidence: `outputs/audit_clamp_summary.json`,
  `outputs/audit_clamp_fea/paired_fea_summary.md`.
* **Sampling batch (P1-7):** 20 targets × 5 draws, same seeds and clamp. Batch size 5 and chunk size
  32 both gave 97/100 valid; mean absolute density errors 0.005455 vs 0.005448; 0.55 mean voxel
  flips per sample. Supports validity/density equivalence, not bitwise equivalence. Evidence:
  `outputs/audit_sampling_batch/summary.json`.
* **Filter before FEA (P1-1):** a fixed mixed 12-grid sample took 35.6 s for 8 FEA solves with
  filtering vs 89.4 s for 12 solves unfiltered — a 60.2% wall-time reduction for that one mix. Do
  not generalize the number. Evidence: `outputs/filter_timing/`.

### 4.4 Profiling and deferred optimizations

* Six full-resolution FEA solves (two densities per family) took 12.1–29.3 s each. Matrix assembly
  and condensation were ~0.20 s combined; six CG solves dominated at 289–636 iterations per load
  case. The historical full-dataset mean was 85.5 s/cell (p95 126, max 170). Reports:
  `outputs/fea_profile.json`.
* A 50-step CUDA profile of the λ=0 warmup path at batch 8 averaged 55 ms/step; convolution
  forward/backward dominated, AdamW ~4.2 ms/step. Batch 32 exceeded the profiler memory budget.
  Reports: `outputs/training_profile.{txt,json}`.
* **Deferred, no expected payoff at this stage:** sparse Cholesky / AMG preconditioning, per-sample
  dolfinx assembly precomputation, and resolution or architecture changes. The profile did not
  justify them.

### 4.5 Test and audit coverage added

Regression coverage now pins: the periodic physics pool vs a reference convolution (value and
gradient), the 24 GPU rotation permutations, multi-decay EMA vs a reference implementation,
filter-before-FEA metrics, zero grid-hash overlap across train/val/test, the 12,154,113-parameter
model with three `SelfAttention3D` modules at 4³, diagonal wrap-around connectivity, and the
sampler's seeded stochastic-DDIM path. The persistent 12-cell FEA golden fixture spans all families
at low/high density and passed at 1e-4 relative scale. The six-cell audit reproduced stored C₆ to
~1e-8.

---

## 5. Nonbinding proposals (not baseline work)

These are retained as decision records. They require supervisor sign-off and must not be treated as
current scope. See `ROADMAP.md` §7.3 and §8.4.

### 5.1 External review and research plan

An external review argued that the baseline is a weaker test of the model than the headline numbers
suggest, because the design space may be effectively one-dimensional (three families, one meaningful
free parameter after 32³ quantization), there is no trivial lookup baseline, and the ablation rests
on one seed and one λ. The concerns are a mix of measured facts and untested inferences; the
proposed research questions remain open.

Proposed sequence, each phase ending in a go/no-go gate:

1. **Phase 0 — audit and complete the baseline.** Add nearest-neighbour lookup and
   family-classifier-plus-regression baselines; run dimensionality and duplicate analyses; use
   harder splits (whole density bands or parameter intervals); establish the training-data validity
   base rate; run 3 seeds, a λ₁ sweep, an EMA-decay check, the corrected-target 540-draw test, a
   DDIM step sweep, and classifier-free guidance; define per-target success and pass@k; add
   family/density breakdowns, a diversity metric, and a failure taxonomy; make headline tables
   reproducible from one command; add CI or a pre-merge script; keep a claims-to-artifact map and an
   explicit limitations section. **Gate G1:** does the diffusion model clearly beat lookup and
   regression off-manifold? If not, go to Phase 1 before making a research claim.
2. **Phase 1 — enrich the design space.** Invoke the ROADMAP §3.1 diversity clause formally; give
   each family at least three free parameters or blend families; condition on the three
   cubic-symmetry stiffness constants (C11, C12, C44) alongside ρ; check homogenization runtime.
   **Gate G2:** the lookup baseline must be clearly beatable on targets that require different
   structures, not just a different density.
3. **Phase 2 — one research question.** Compare training-time physics, sampling-time guidance from a
   small property surrogate (Option A), and a differentiable synthesizability term (soft
   minimum-wall penalty). Real FEA remains the final arbiter; report per-target success, pass@k,
   validity, diversity, and cost over at least 3 seeds. **Gate G3:** do the gains hold against
   unguided, CFG-only, and lookup?
4. **Phase 3 — stretch goal.** Physical printing/testing or one narrow nonlinear target, depending
   on time and printer access. Reserve 6–8 weeks for writing.
5. **Fallback.** A well-controlled “how not to evaluate inverse design” study is publishable at
   workshop level: unreachable targets, R² deflated by target spread, exact-cell held-out targets,
   and a near-1-D benchmark.

Decisions still needed: actual deadline (rescale ROADMAP §8.1), printer and mechanical-test access,
and written sign-off before reopening any locked decision. Literature pointers from the review
(verify before citing): Zheng/Shiomi/Yamada 2025 (3D conditional diffusion); Yang et al. (guided
voxel diffusion); Bastek & Kochmann (video diffusion); DiffMat; DiffuMeta; HyperDiff; Liu et al.
(SDF + neural operator); physics-guided diffusion for disordered foams; style-constrained
unconditional diffusion; dual-constrained diffusion (training vs sampling constraints);
Design-GenNO; curved-metamaterial latent diffusion; cVAE resonant structures vs lookup; the
data-driven metamaterials review.

### 5.2 Dataset-v2 proposal digest

v1 has two independent defects: **296 of 2,400 cells (12.3%) are invalid by construction** (all
`min_wall_thickness`), and **1,377 of 2,400 rows are duplicate grids** (1,023 unique), with up to 54
rows sharing one octet label. The derived `data/v1f/` dataset already implements the gate-only
correction: 2,400 rows, 100% valid, 935 unique grids, no new FEA. This proposal covers the remaining
levers and the decision record.

Measured effect of each candidate lever, applied to the existing v1 draws:

| variant | rows | cubic/octet/gyroid | valid | validity | unique grids | max rows/grid |
|---|---|---|---|---|---|---|
| L0 — v1 as-is | 2,400 | 800/800/800 | 2,104 | 87.7% | 1,023 | 54 |
| L1a — floor on achieved ρ | 2,110 | 798/735/577 | 2,076 | 98.4% | 921 | 54 |
| L1b — floor on target ρ | 2,068 | 800/705/563 | 2,034 | 98.4% | 919 | 54 |
| L2 — L1b + validity gate | 2,034 | 800/671/563 | 2,034 | 100% | 917 | 54 |
| L3 — L2 + dedup (1 row/grid) | 917 | 515/134/268 | 917 | 100% | 917 | 1 |
| L4 — L2 + balanced allocation | 2,400 | 800/800/800 | 2,400 | 100% | 917 | 6 |

The P0–P3 decision record:

* **P0 — fix the acceptance protocol (minutes, no compute).** Re-score the existing samples against
  the existing manifest, excluding the 3 unmanufacturable held-out gyroid targets (ρ = 0.170, 0.161,
  0.110) and 2 unreachable off-manifold targets; report the full-set and satisfiable-subset numbers
  side by side. This is a reporting-correction, not a model change.
* **P1 — derive a corrected dataset (minutes, no FEA; retrain optional).** Drop invalid rows and
  balance per split/family. `data/v1f/` is the completed variant (100% valid, 935 unique grids,
  split sizes preserved).
* **P2 — regenerate (≈1.3–1.7 h generation plus retraining).** Buys ~30–45 extra unique grids (+4%)
  but cannot fill the gyroid hole; needs the §7.3 decision.
* **P3 — enrich the within-family parameterization.** The only lever that raises effective sample
  size; changes what a family *is*, so it is beyond the three sanctioned levers and needs its own
  sign-off.

Cost model (v1 observed, v2 estimated): parent-side rasterization was ~5,128 s serial (~45% of the
11,251 s generation); FEA for 1,023 unique grids was ~78,222 s serial / ~4,889 s on 16 workers.
Parallel rasterization brings generation to ~1.3–1.7 h; two-arm retraining is ~6.5–7 h; the full
pipeline is ~8–9 h. The matched physics-on vs λ₁=0 conclusion is dataset-independent (both arms
share the data), so P1's retrain buys better absolute numbers, not a different ablation verdict.

**Status caveat.** The old proposal used line numbers and config filenames that may already have
changed; the working tree now contains generator changes for duplicate-grid caching and
parameterized geometry. Re-check `src/data/generate_dataset.py`, `configs/dataset.yaml`, and the
actual `data/manifest.parquet` before acting on any line-number-level detail here.
