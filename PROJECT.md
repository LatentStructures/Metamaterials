# Project Overview

An easy-to-read tour of this repository. For the binding spec, scope tags, and acceptance criteria,
read `ROADMAP.md`; this file only explains *what* the project is and *how* the code fits together.

## 1. What is this project?

We build **mechanical metamaterials by design** using AI.

A *mechanical metamaterial* is a material whose properties come not from its chemistry but from its
**microstructure** — the tiny repeating pattern ("unit cell") inside it. Think of a sponge: it is
mostly air, yet it can support weight with the right structure. By choosing the 3D pattern of solid
and void inside a unit cell, we can get unusual effective properties, like deforming sideways when
you squeeze it, or being almost incompressible while staying very light.

The hard part is the *inverse* problem:

> **Forward:** given a unit-cell pattern, compute its properties (easy-ish — simulate it).
> **Inverse:** given desired properties, find a unit-cell pattern that has them (very hard —
> infinite possible patterns).

Our answer: train a **conditional generative model** (a type of AI, a diffusion model) that has seen
thousands of "pattern → property" examples, so that when you give it a target like *"normalized
Young's modulus E/E_s = 0.1, relative density ρ\* = 0.3, Poisson ratio ν = 0.2"*, it draws a new 3D
unit cell that should achieve those values. We then **verify every generated design with a physics
simulation (FEA)** before shipping it, and export the survivors as STL files ready for a 3D printer.

**Everything in this baseline is simulation-only.** We do not fabricate anything; "manufacturable"
means the structure passes automated checks that predict whether it *could* be printed.

## 2. Vocabulary you will meet

| Term | Meaning, in plain language |
| --- | --- |
| **Unit cell** | The smallest repeating block of a lattice; a 32×32×32 grid of solid(1)/void(0) voxels in this project |
| **Voxel** | A 3D pixel — a cube in a grid; the "Lego brick" of our representation |
| **Relative density ρ\*** | The fraction of the cell that is solid (between 0 and roughly 0.6 here) |
| **Stiffness tensor C** | A 6×6 matrix that completely describes how the material resists stretching and shearing |
| **Young's modulus E** | Effective stiffness; with the configured `E_s = 1`, values are dimensionless ratios `E_eff/E_s` |
| **Poisson ratio ν** | How much it thins sideways when you pull it along one axis (rubber ≈ 0.5, cork ≈ 0) |
| **Homogenization** | Treating the detailed repeating pattern as an *effective* solid and computing its overall (homogenized) properties via simulation |
| **FEA / FEniCSx** | Finite-Element Analysis — the physics solver (FEniCSx is the particular library we use) |
| **DDPM / DDIM** | Denoising Diffusion Probabilistic Models — the "draw from noise step by step" generative AI technique (DDIM is a fast sampler for it) |
| **U-Net** | A neural network shaped like a U (compress → think → expand) used as the "brain" of the diffusion model |
| **Conditioning vector** | The three numbers `[E, ρ*, ν]` we ask the model to match |
| **Synthesizability filter** | Automatic checks (connectedness, wall thickness, overhang) that predict whether a design can be 3D-printed |

## 3. The pipeline at a glance

```
 1. GENERATE         2. SIMULATE (+verify)     4. TRAIN
 ┌──────────┐        ┌────────────────────┐    ┌───────────────┐
 │ 3 families→│──▶ 32³ voxel cell ──▶ FEA homogenization
 │ random    │       └────────────────────┘    │ conditional    │
 │ densities │            │                    │ DDPM (U-Net)   │
 └──────────┘            ▼                     └───────────────┘
                  (pattern, E, ρ*, ν) pairs           │ (learns to
                    → HDF5 dataset + manifest         │  draw cells
                                                      │  for a target)
                                                      ▼
 5. VERIFY   ◀── 3. SAMPLE: give target [E,ρ*,ν] ──▶ new voxel cell
 FEA re-run      the model "denoises noise" into        │
 + filter        a candidate structure                  ▼
      │                                         6. EXPORT
      ▼                                        best cells as STL
 (achieved E,ρ*,ν) vs. target ──▶ error report
```

Concretely, the five user-facing commands in `README.md` map to the five boxes:

| Step | Command | What it does |
| --- | --- | --- |
| 1+2. Generate dataset | `python -m src.data.generate_dataset configs/dataset.yaml` | Rasterizes thousands of unit cells from the 3 geometry families, solves each with FEA, writes labeled HDF5 files + a manifest |
| 3. Train | `python -m src.train` | Trains the conditional diffusion model to draw cells matching `[E, ρ*, ν]` |
| 4. Sample | `python -m src.sample --checkpoint ... --targets ...` | Generates candidate voxel grids from held-out target property vectors |
| 5. Verify | `python -m src.evaluate` | Re-runs real FEA on every generated sample, applies the filter, reports reconstruction error |
| 6. Export | `python -m src.export.export_stl` | Exports the top-N valid structures as STL files |

## 4. The five ingredients, explained

### 4.1 The geometry — three unit-cell families

We use exactly three pattern families, each with a tunable parameter
(`src/geometry/lattice_families.py`):

- **cubic_strut** — three sets of rods running along the x, y, z directions. Parameter: rod radius.
- **octet_truss** — the classic strong lightweight truss (face-centered-cubic nearest-neighbour
  struts). Parameter: strut radius.
- **gyroid** — a smooth, undulating 3D surface described by a math formula (a "triply periodic
  minimal surface"). Parameter: the level-set threshold, which controls solid fraction.

For a chosen target density, `sample_for_density` performs a bisection search over the parameter to
hit the requested density band (0.1–0.6). Geometries are **periodic**: a strut that leaves one face
of the cell continues on the opposite face, exactly matching what the FEA solver assumes.

The rasterization (`src/geometry/voxelize.py`) turns each shape into a binary `32×32×32` grid, and
`src/geometry/mesh_utils.py` converts that grid into a tetrahedral mesh for the solver (with one
subtlety — a translation-invariant Kuhn triangulation — so the mesh has no cracks between
neighbouring voxels).

### 4.2 The physics — FEA homogenization

`src/fea/homogenization.py` implements **computational homogenization** (the standard method of
Andreassen & Andreasen, 2014):

1. The unit cell is turned into a tetrahedral mesh.
2. **Periodic boundary conditions** are enforced exactly: the displacement on each +face is tied to
   the corresponding point on the −face (master–slave constraints), which is the fiddliest part of
   the whole pipeline.
3. Six "unit strain" load cases are solved (3 stretch + 3 shear), giving the full 6×6 stiffness
   tensor **C** of the effective material.

`src/fea/property_extraction.py` reduces C to the three conditioning scalars: `E`, `ρ*`, and `ν`
(via Voigt–Hill averaging for near-cubic cells). The shear modulus G is computed only as a side
metric — it is deliberately **not** a conditioning target.

**Validation gate:** before ANY dataset is allowed to be generated, the solver must reproduce three
analytically-known cases within 5% — a solid cube, a laminate, and aligned rods
(`src/fea/benchmarks.py`, `validate_vs_analytical`). A silently wrong solver would poison every
label, so this check is mandatory and stored in `data/validation_table.parquet`.

### 4.3 The AI — conditional diffusion (DDPM + U-Net)

- **Diffusion process** (`src/models/diffusion.py`): DDPM training with T = 1000 timesteps. Training
  = add noise to a real cell, then teach the model to predict the noise back (standard ε-MSE loss).
  Sampling = start from pure noise and iteratively remove it, guided by the target properties, using
  a **DDIM** sampler (50–100 steps instead of 1000, for speed).
- **Base architecture** (`src/models/unet3d.py`): a 3D U-Net with four resolution stages (channel
  widths 32→64→128→256), residual blocks, and **self-attention** at the lowest 4³ resolution.
- **Conditioning** = a small MLP that maps `[E, ρ*, ν]` into the network via **FiLM**
  (scale-and-shift) layers, so the drawn structure adapts to the requested properties. The
  conditioning order `[E, relative_density, nu]` is a **locked invariant** enforced across the
  manifest, the HDF5 files, and the model.

### 4.4 The physics-informed loss

Running a full FEA solve on every training step would be prohibitively slow. Instead we add a cheap,
differentiable **connectivity proxy** (Option B, `src/losses/physics_loss.py`): an algebraic measure
(Fiedler eigenvalue of a soft voxel graph) that is ~0 for a well-connected structure and ~1 for one
with floating/disconnected parts. It discourages the model from producing obviously unphysical,
disjoint debris. The total loss is

```
L = L_diffusion + λ₁(t) · L_connectivity
```

with λ₁ warmed up from 0 over the first 20% of training. (A stronger "neural surrogate" critic —
Option A — exists only in the docs; it is a contingency and is not built.)

### 4.5 Verification, filter, export

- `src/sample.py`: draws cells for targets from a CSV file.
- `src/evaluate.py`: re-homogenizes every generated cell with the **real** FEA solver (not the
  model), so we know the *achieved* properties; applies the **synthesizability filter** — single
  connected component, minimum wall thickness, overhang estimate; reports validity rate and
  reconstruction error (target vs. achieved) with parity plots.
- `src/export/export_stl.py`: ranks valid candidates by reconstruction error, exports the top-N to
  smooth STL, and checks watertightness.
- **Augmentation** (`src/data/augment.py`): each cell can be rotated by any of the 24 proper cube
  rotations to multiply the effective dataset ~24× — but only *proper rotations* (no mirror
  reflections), and the stiffness tensor is rotated to match so labels stay correct.

## 5. Data format

One HDF5 file per dataset split (`data/dataset_{train,val,test}.h5`):

| Dataset | Contents |
| --- | --- |
| `/voxels` | `(N, 32, 32, 32)` uint8 binary grids |
| `/conditioning_vector` | `(N, 3)` float32 `[E_norm, ρ*_norm, ν_norm]` (normalized; mean/std stored as attributes) |
| `/stiffness_tensor` | `(N, 6, 6)` float32 effective stiffness C |
| `/metadata` | family, generating parameters, validity |
| `/conditioning_order` | literal `["E", "relative_density", "nu"]` so loaders can reject mismatches |

Plus `data/manifest.parquet` (cell_id → split → family → properties → file paths, with voxel paths
relative to the repository root) and `data/data_card.md` (a one-page summary of the dataset). Splits
are 80/10/10, **stratified by family and density bucket**, and checked for near-duplicate leakage
across splits. Data binaries are gitignored.

A second, **corrected** copy of the dataset lives in `data/v1f/`: the same 2,400 rows with the 296
rows that fail the periodic synthesizability filter removed and the per-(split × family)
grid-multiplicity skew evened out. It therefore has **100% valid rows** but only **935 unique
grids** (vs V1's 1,023). It was derived by `scripts/derive_corrected_dataset.py` **without any new
FEA solve** (labels copied bit-for-bit) and is documented in `data/v1f/data_card.md`. Training on it
uses `configs/train_baseline_v1f.yaml`, and sampling/evaluating a V1f checkpoint requires
`--train-config configs/train_baseline_v1f.yaml` because the conditioning normalization statistics
differ.

## 6. Where things live

```
configs/          One YAML per run type: dataset, model, train (baseline),
                  evaluate
src/config.py     Config loaders + the locked conditioning-order check
src/geometry/     Unit-cell generators, 32³ rasterization, mesh building
src/fea/          FEniCSx homogenization, property extraction, benchmarks
src/data/         Dataset generation, HDF5 loader, rotation augmentation
src/models/       3D conditional U-Net + DDPM/DDIM diffusion
src/losses/       Option-B connectivity proxy (+ functional-form README)
src/validation/   Synthesizability filter: connectivity / wall / overhang checks
src/export/       STL export with watertightness reporting
src/train.py      Training loop (multi-decay EMA, cosine LR, bf16, W&B/TensorBoard)
src/sample.py     Conditional DDIM sampling → .npy / STL
src/evaluate.py   FEA re-verification + filter + error report
src/voigt.py      The single shared Voigt index convention
tests/            170 passing tests + tests/smoke_test.py (env DoD; run it directly — it contributes no pytest items)
```

## 7. Key decisions (locked — do not revisit)

- **32³ binary voxels** for the whole baseline (no higher resolution).
- **Exactly three** topology families (cubic strut, octet-truss, gyroid).
- Conditioning vector order is fixed as `[E, relative_density, nu]`; G is never a conditioning
  target.
- FEA = **FEniCSx** (`dolfinx`), never the legacy `dolfin`.
- Model = custom 3D U-Net with FiLM conditioning and bottleneck self-attention; no cross-attention,
  no DiT.
- Physics loss = **Option B** proxy only; the neural-surrogate critic (Option A) is a contingency.
- Augmentation = the 24 proper cube rotations only, never mirrors.
- Standard DDPM (T=1000) for training; **DDIM 50–100 steps** at inference.
- Every run reproducible from a single config file + fixed seed; the FEA workers cap their thread
  counts (this is what prevents the OOM/oversubscription crash that killed an earlier full dataset
  run).

## 8. Current status & what's left

**Scope note (2026-10-22).** The full suite passes **170 tests** and `ruff check src tests scripts`
is clean. A code review found that the old Option-B soft graph used zero-padded neighbours while the
hard validity filter is periodic. The proxy now uses circular boundaries, matching the filter's
convention. Everything below labeled *legacy* therefore describes the old zero-padded proxy and is
retained as an audit trail, not as evidence about the current model. The current evidence is the
matched corrected-proxy comparison in §8.1.

### 8.1 Corrected-proxy matched results (current evidence)

Two pipelines were run under an identical protocol (same targets, draw count, seeds, and FEA path):

| pipeline | dataset | physics-on run | physics-off run | artifacts |
|---|---|---|---|---|
| **V1** (corrected proxy) | `data/` — 2,400 cells, 87.7% valid, 1,023 unique grids | `runs/final_periodic_physics_on_gpu` | `runs/final_periodic_physics_off` | `outputs/final_acceptance/` |
| **V1f** (corrected dataset) | `data/v1f/` — 2,400 cells, **100% valid**, 935 unique grids | `runs/final_v1f_physics_on` | `runs/final_v1f_physics_off` | `outputs/final_acceptance_v1f/` |

All four runs reached 30,000/30,000 iterations with `optimizer.grad_clip: 1.0`, seed 0, all three
EMA shadows present, and **zero non-finite** logged scalars (`outputs/audit/verify_*.json`). Every
recorded evaluation completed with **0 FEA failures**.

**Protocol.** Both arms of each pipeline use `configs/train_baseline.yaml` (seed 0, batch 32,
T=1000, λ₁=0.05, Option B, EMA decays 0.999/0.9995/0.9999). The λ₁=0 arm is produced with
`--physics-ablation`. Sampler weights are chosen **on the 90-target validation set only** (highest
validation validity; ties prefer `live`, then the smallest EMA decay) — never on the acceptance or
off-manifold sets. Acceptance uses the checked-in `data/targets_test_50.csv` (50 targets) and
`data/targets_synthetic.csv` (108 targets), 1 draw per target, seed 0, and the full hard-validity +
real-FEA path. Every number below is re-derived from the stored `per_sample.csv` / `report.json`.

**Selected sampler weights** (`outputs/ema_validation*/`): V1 selected `ema-0.9999` for *both* arms;
V1f selected `ema-0.9995` (physics-on) and `live` (physics-off). The multi-decay tracking therefore
resolves the old "EMA is broken" symptom differently per run — see the caveat in §8.2.

Cite the four finished runs, not any interrupted attempts.

#### Headline numbers

| run | n | validity | FEA fails | E err / R² | ρ err / R² | ν err / R² | ρ<0.25 | ρ≥0.25 |
|---|---|---|---|---|---|---|---|---|
| **V1** corrected physics-on (50 held-out) | 50 | 88.0% | 0 | 0.0682 / 0.961 | 0.0123 / 0.998 | 0.0235 / 0.765 | 7/13 (53.8%) | 37/37 (100%) |
| **V1** corrected physics-off (50 held-out) | 50 | 94.0% | 0 | 0.0288 / 0.996 | 0.0112 / 0.998 | 0.0101 / 0.955 | 10/13 (76.9%) | 37/37 (100%) |
| **V1** corrected physics-on (108 off-manifold) | 108 | 95.4% | 0 | 0.1615 / 0.719 | 0.0139 / 0.993 | 0.0302 / 0.595 | 6/9 (66.7%) | 97/99 (98.0%) |
| **V1** corrected physics-off (108 off-manifold) | 108 | 98.1% | 0 | 0.1259 / 0.849 | 0.0195 / 0.983 | 0.0140 / 0.896 | 8/9 (88.9%) | 98/99 (99.0%) |
| **V1f** corrected-dataset physics-on (50 held-out) | 50 | 90.0% | 0 | 0.0553 / 0.989 | 0.0112 / 0.998 | 0.0307 / 0.564 | 8/13 (61.5%) | 37/37 (100%) |
| **V1f** corrected-dataset physics-off (50 held-out) | 50 | 94.0% | 0 | 0.0326 / 0.996 | 0.0116 / 0.999 | 0.0123 / 0.934 | 10/13 (76.9%) | 37/37 (100%) |
| **V1f** corrected-dataset physics-on (108 off-manifold) | 108 | 95.4% | 0 | 0.1396 / 0.792 | 0.0132 / 0.994 | 0.0358 / 0.418 | 7/9 (77.8%) | 96/99 (97.0%) |
| **V1f** corrected-dataset physics-off (108 off-manifold) | 108 | 96.3% | 0 | 0.1307 / 0.825 | 0.0123 / 0.995 | 0.0344 / 0.432 | 8/9 (88.9%) | 96/99 (97.0%) |

Sources: `outputs/final_acceptance/summary.{md,json}` and
`outputs/final_acceptance_v1f/summary.{md,json}`. `ρ<0.25` / `ρ≥0.25` are validity counts split at
the low-density corner identified in `FINDINGS.md` §2.

#### What the matched comparison says

**The physics-off arm leads on validity in both pipelines, on both target sets.** V1: 94.0% vs 88.0%
held-out, 98.1% vs 95.4% off-manifold. V1f: 94.0% vs 90.0% held-out, 96.3% vs 95.4% off-manifold.
The property metrics agree — physics-off has lower E and ν error and higher R² in 7 of the 8
property × dataset × target-set cells; density is the exception (physics-off is worse on ρ in V1
off-manifold and V1f held-out).

This **reproduces the legacy direction** on the corrected code, so the legacy verdict is now
supported by a matched run rather than only by the legacy implementation. It remains a
**single-seed** result (one seed per arm), so it is a consistent direction across two independent
pipelines, not a replicated estimate.

**The corrected gradient is far better behaved than the legacy one.** Max pre-clip gradient norm was
**37.2** (V1 physics-on) and **23.2** (V1f physics-on) versus **5.91** and **6.04** for their
physics-off arms — the same ordering and roughly the same 4–6× ratio as the legacy 4,229 vs 1.31,
but two orders of magnitude smaller in absolute terms. The final logged loss is 0.00302 (V1 on) vs
0.00263 (V1 off) and 0.00378 (V1f on) vs 0.00396 (V1f off): the legacy "3.6× loss ratio" does
**not** reproduce (ratios 1.15× and 0.95×), because the physics term's magnitude changed with the
boundary condition. The legacy gradient-forensics mechanism in `FINDINGS.md` §1 therefore does not
transfer; only the outcome does.

**Failure-mode shift with the corrected dataset.** V1's physics-on failures on the held-out set were
`min_wall_thickness` 5 + `connectivity` 1; V1f's are `connectivity` 5 and `min_wall_thickness` 0.
Training on 100%-valid rows removed the thin-wall failure mode from the physics-on arm entirely, and
what remains is fragmentation. The physics-off arm moved the same way (V1: `min_wall` 3 → V1f:
`connectivity` 3).

#### Full-set vs satisfiable-subset validity

Some targets cannot be satisfied by *any* valid geometry: three held-out targets are gyroid cells
below the 2-voxel-sheet density floor (their own reference cells fail the wall rule), and one
off-manifold octet target lies below the lowest valid octet density. The **full-target-set number
remains the acceptance-criterion input**; the satisfiable-subset number is reported alongside as
disclosure, never as a substitute. Verdicts are taken from the ground-truth `data/manifest.parquet`
for both pipelines, so the two are directly comparable.

| run | full-set validity | satisfiable-subset validity | unsatisfiable targets |
|---|---|---|---|
| V1 physics-on (50) | 88.0% (44/50) | **93.6%** (44/47) | 3 |
| V1 physics-off (50) | 94.0% (47/50) | **100.0%** (47/47) | 3 |
| V1 physics-on (108) | 95.4% (103/108) | **96.3%** (103/107) | 1 |
| V1 physics-off (108) | 98.1% (106/108) | **99.1%** (106/107) | 1 |
| V1f physics-on (50) | 90.0% (45/50) | **95.7%** (45/47) | 3 |
| V1f physics-off (50) | 94.0% (47/50) | **100.0%** (47/47) | 3 |
| V1f physics-on (108) | 95.4% (103/108) | **96.3%** (103/107) | 1 |
| V1f physics-off (108) | 96.3% (104/108) | **96.3%** (103/107) | 1 |

Reconstruction metrics are computed on valid samples only, so they are identical in both columns in
7 of 8 rows. The exception is V1f physics-off off-manifold, where the one excluded (unsatisfiable)
target happened to produce a *valid* sample, so dropping it shifts the reconstruction statistics
slightly (E R² 0.825 → 0.814); that is reported explicitly in
`outputs/final_acceptance_v1f/satisfiable_v1f_physics_off_synthetic.json`
(`reconstruction_unchanged: false`). Source: `outputs/final_acceptance*/satisfiable_*.{md,json}`.

#### V1 vs V1f — the dataset-correction trade-off

| 50 held-out, physics-on | validity | E err / R² | ρ err / R² | ν err / R² |
|---|---|---|---|---|
| V1 (corrected proxy, 87.7%-valid dataset) | 88.0% | 0.0682 / 0.961 | 0.0123 / 0.998 | 0.0235 / 0.765 |
| V1f (corrected proxy, 100%-valid dataset) | **90.0%** | **0.0553 / 0.989** | 0.0112 / 0.998 | 0.0307 / **0.564** |

V1f trades **+2 pp validity** and a better E reconstruction for a **worse ν reconstruction** (R²
0.765 → 0.564 on held-out; 0.595 → 0.418 off-manifold). The physics-off arms are nearly unchanged
between the datasets (94.0% validity both; E/ρ R² ≥ 0.996 both), which locates the ν regression in
the physics-on × corrected-dataset combination rather than in the dataset change alone — but with
one seed per cell that is a hypothesis, not a mechanism.

**Recommendation: keep V1 as the historical control and treat V1f as the candidate replacement
baseline, but report both.** Neither pipeline dominates on every metric, and the V1f dataset has a
real cost that must travel with it: **935 unique grids vs V1's 1,023** (effective sample size, see
`data/v1f/data_card.md`). V1f is better on validity, E accuracy, failure-mode cleanliness, and
training-data validity (100% vs 87.7%); V1 is better on ν accuracy. The dataset correction moves the
error budget from E toward ν rather than removing it.

**Normalization is dataset-specific.** V1f's conditioning statistics differ from V1's (train mean
`[0.1809, 0.4055, 0.2668]` vs `[0.1457, 0.3480, 0.2735]`). Sampling or evaluating a V1f checkpoint
**must** pass `--train-config configs/train_baseline_v1f.yaml`; using the default config silently
normalizes with the wrong statistics and produces a materially worse model (this happened once
during the run and was caught by re-sampling — the discarded pass showed E R² 0.803 instead of 0.989
on the held-out set).

#### Known limitations of these results

1. **Single seed.** One seed (0) per arm per pipeline. The physics-off advantage is consistent
   across 2 pipelines × 2 target sets, but no per-target variance or seed variance is available.
2. **Effective sample size.** V1 has 1,023 unique grids among 2,400 rows; V1f has **935** among
   2,400. Duplicates act as implicit loss weights, so the effective training set is ~935–1,023
   geometries, not 2,400.
3. **The 26-neighbour connectivity check is topological only.** Edge- and corner-only contacts have
   no finite cross-section, so "one connected component" does not establish finite contact area,
   load-bearing strength, or manufacturability. Wall-thickness checks and FEA remain separate gates.
4. **Off-manifold E R² still misses 0.85** in every corrected arm (0.719–0.792 physics-on,
   0.825–0.849 physics-off). The E *error* gate passes everywhere (≤ 16.2%). As established for the
   legacy run, the R² gap is target-spread dominated, not model-error dominated.
5. **Multi-draw scope.** The corrected-target V1f estimate uses 5 draws on each of 108 synthetic
   targets: physics-on 95.6% (516/540), physics-off 96.5% (521/540), 0 FEA failures, with
   overlapping target-level 95% intervals ([93.5%, 97.6%] vs [94.0%, 98.9%]) — so no reliable
   validity difference between arms is established by that run. It remains single-seed and does not
   measure between-seed variability. The legacy 92.2% interval belongs to its original pre-fix
   target set; do not pool it with the corrected estimate.
6. **The EMA finding is now run-dependent.** V1 selected `ema-0.9999` for both arms; V1f selected
   `ema-0.9995` for physics-on and `live` for physics-off. The old "EMA weights are broken" claim
   was about the *legacy* single-decay shadow under the old schedule; the multi-decay implementation
   is selected per run on validation data and is not broken. `live` remains the sampler default for
   historical reproducibility.

### 8.2 Legacy results (historical, zero-padded proxy)

**Scope:** every number, mechanism, and causal statement in this subsection refers to the legacy
zero-padded physics proxy and fixed-rotation training behaviour. It is retained as an audit trail
only; §8.1 supersedes it for the physics-loss comparison.

**Legacy acceptance run.** The §7.1 acceptance run (50 held-out targets, real FEA, **live** weights)
reached validity **92.0%** (46/50) with 0 FEA failures and reconstruction error E 3.63% / ρ 1.51% /
ν 0.99%, R² 0.995 / 0.998 / 0.958. Export produced 5/5 watertight STLs
(`outputs/acceptance_30k/export_final`). Validity is a **low-density-corner** number (`FINDINGS.md`
§2): all 4 failures are gyroid at ρ < 0.25 (every non-gyroid ρ < 0.25 target passes; every ρ ≥ 0.25
target passes), and pooled across the recorded legacy eval sets 85.7% of failures sit below ρ =
0.25, where pooled validity is 98.7% above that line.

**Off-manifold control.** Every one of the 50 held-out targets is an exact dataset cell (distance
0.000000), and 25/50 generated samples had IoU > 0.9 against their source cell, so the acceptance
numbers alone are partly circular. Against **108 synthetic off-manifold targets** (≥0.012 from any
real cell) the legacy model reached validity **97.2%** (105/108) with 0 FEA failures, so validity
does not collapse and the model interpolates rather than recites. Errors degrade as they should (E
3.6% → 13.8%, ν 1.0% → 1.5%).

The target builder was rebuilt on 2026-10-20: the previous version drew ν uniformly over each
family's band independent of `(E, ρ)`, leaving **46 of 108 targets physically unreachable**. Those
pre-fix numbers (94.4% validity, ν R² 0.315, "variance compression") are superseded and must not be
quoted. A regression test (`test_targets_are_physically_reachable`) now fails the old builder's
output. The 108-target evaluation itself gives **92.2%** (498/540) over 5 draws, 95% CI [90.0%,
94.5%], 0 FEA failures — a conservative lower bound for that old set, with 93/108 targets passing
5/5 and 4 failing 0/5. The residual E R² miss (0.808 vs 0.85) is target-spread dominated: E's error
is 13.75%, inside the ≤15% gate.

**λ₁ = 0 ablation (legacy).** The control was trained identically except λ₁ = 0
(`runs/baseline_physics_off`, 30k iterations, config snapshot confirms `lambda_1: 0.0`):

| metric | physics-ON (λ₁=0.05) | physics-OFF (λ₁=0) |
|---|---|---|
| §7.1 validity, 50 held-out | 92.0% | **94.0%** |
| FEA solver failures | 0 | 0 |
| E mean error / R² | 3.63% / 0.9954 | **2.44% / 0.9979** |
| ρ mean error / R² | 1.51% / 0.9978 | **1.30% / 0.9981** |
| ν mean error / R² | 0.99% / 0.9584 | **0.56% / 0.9898** |
| final loss (mean, last 5k) | 0.0118 | **0.0033** |
| max pre-clip grad, iters 20k–30k | **4,229** | 1.31 |
| post-warmup grad excursions >1 | 12 | 2 |

On the 108 off-manifold targets: validity 97.2% (105/108) vs **98.1% (106/108)**; E 13.75% / R²
0.808 vs **12.36% / R² 0.849**; ρ **1.84%** / 0.988 vs 2.15% / 0.982; ν 1.48% / R² 0.858 vs **1.17%
/ R² 0.931** (`outputs/ablation_30k/evaluate_synthetic`). A paired per-target check shows 0 targets
valid under physics-on only and 1 under physics-off only, with 107 agreeing.

Two conclusions. First, the physics gradient was the source of the legacy instability: iters 1–1000
are bit-identical between the runs (5.905) because λ₁ warms from 0, and once it activates the
physics-on run spikes to 4,229 while physics-off never exceeds 1.31 — so `grad_clip` was necessary
*because of* the physics term. Second, the physics term did not improve validity or any property
metric; the control matched or beat it everywhere on the same targets, seed, and schedule, and the
loss was 3.6× lower.

**Mechanism (wall thickness, legacy).** The physics-on arm produced measurably more thin wall: mean
thin-voxel fraction 0.0472 vs 0.0450, and paired across the same 50 targets, **37 targets are
thinner under physics-on vs 11 under physics-off** (Wilcoxon signed-rank p = 0.0015; p = 0.0040
excluding the three pathological low-density targets). This is consistent with the Option-B
connectivity proxy rewarding connectedness, since thin struts are the cheapest way to stay connected
at low density — the loss appears to buy connectivity by spending wall thickness, a trade with no
net benefit given the filter already enforces `min_wall_voxels: 2`. Thin-ness in both arms is
strongly anti-correlated with target density (r ≈ −0.43), and the worst cases are the *same*
low-density targets in both arms, so the paired test is what establishes the per-target difference.
This mechanism is a legacy-proxy finding; do not carry it over to the corrected proxy.

**How the ablation verdict should be read (scoping, not science).** `ROADMAP.md` §7.2 commits the
baseline only to the ablation being **documented, even if the effect is small** — not to the loss
*working*. On that reading the baseline obligation is fully met. What the ablation does **not**
support is any claim that Option B is *necessary* or *beneficial* at λ₁ = 0.05 on this architecture,
and the honest statement is that the evidence points the other way, in both the legacy and corrected
runs. Stronger interpretations ("Option B is broken as specified", "λ₁ is mis-tuned", "a
connectivity penalty cannot steer an ε-MSE DDPM") each require experiments this baseline does not
contain (a λ sweep, a second seed, an alternative formulation) and are **research-track
hypotheses**.

**The legacy single-decay EMA arm.** The EMA weights scored **14.0%** (7/50) with one FEA solver
failure versus **92.0%** for the live weights. This is a **config-scaling artefact, not a defect in
the EMA implementation or the model**. Ruled out: a broken/stale average (all 126 parameter keys
present; `‖EMA − live‖ / ‖live‖ = 0.052`, `cos(EMA, live) ≈ 1.0003`); model specificity (the
physics-off checkpoint fails identically under its own EMA weights: mean 9.34 connected components,
29/50 fragmented, versus 1.62 live); and generic sensitivity to a 5% weight change (isotropic
Gaussian noise at the same 5.2% relative magnitude leaves geometry clean, 1.54 components vs live
1.62).

The cause is a **dose–response along the live→EMA direction**:

| α (0 = live, 1 = EMA) | mean connected components | fragmented samples |
|---|---|---|
| 0.00 | 1.62 | 15/50 |
| 0.25 | 1.72 | 17/50 |
| 0.50 | 3.40 | **33/50** |
| 0.75 | 4.90 | 39/50 |
| 1.00 | 11.50 | 44/50 |

`decay = 0.9999` gives an effective averaging window of `1/(1 − decay) = 10,000` steps, centred
~step 25,000 of a 30,000-step run. The cosine schedule anneals the live weights to **lr = 0.0** at
the final step, and the training loss is still falling steeply there (1.63e-2 at step 27k → 5.16e-3
at step 30k). The EMA therefore averages over a **less-converged region**; moving even halfway
toward it doubles the fragment rate, and the periodic-connectivity filter then correctly rejects
those grids — hence 14%.

Training now tracks 0.999 / 0.9995 / 0.9999 with early-step warmup, and the sampler selects a shadow
per run on the 90-target validation set (V1 → `ema-0.9999` both arms, 445–447/450; V1f →
`ema-0.9995` physics-on and `live` physics-off, 445/450 and 447/450). The multi-decay implementation
is not broken; the legacy 14% is a property of the old single-decay shadow under the old schedule.
Evidence: `outputs/ema_validation/*_weights.json`, `outputs/ema_validation_v1f/*_weights.json`.

### 8.3 Integrity audits (2026-10-21)

Every headline number was re-derived from the artifacts rather than quoted from docs; two artifacts
that failed the check were corrected.

**Dataset (2,400 cells) — PASS.** The manifest is complete (2,400 rows, 800 per family, zero NaN,
all 2,400 voxel files on disk). Re-deriving (E, ν) from every stored 6×6 stiffness tensor through
the generator's own `effective_properties` path reproduces the conditioning labels to median
relative diff **2.68e-08** (Pearson r ≈ 1.000000), and each grid's solid fraction matches its
conditioning ρ to **8.6e-09**. All 2,400 tensors are exactly symmetric (major and minor) and SPD
(min eigenvalue 1.008e-03); zero non-finite values anywhere. Re-running FEA on 6 cells (2 per
family) with the current code reproduces the stored tensors to **~1e-8** — solver-noise level — and
the pre-generation validation table's three closed-form benchmarks all sit within 6.3e-07 of
analytic. Splits are 1,920/243/237 with **no duplicate-grid leakage** (all bitwise duplicates of a
cell sit in the same split and family), and the HDF5 ↔ manifest ↔ generator-checkpoint row-level
match is **2,400/2,400**. Disclosed anomalies, none disqualifying: only **1,023 unique grids** of
2,400 rows (deterministic 32³ rasterization of continuous parameters — duplicates carry
bitwise-identical stiffness, so effective diversity is ~43% of nominal), and **35 cells** sit
marginally outside the [0.1, 0.6] band in *achieved* (not target) density with labels still
self-consistent.

**Training (both legacy runs) — PASS.** `baseline_clipfix_30k` and `baseline_physics_off` each ran
the full 30,000/30,000 iterations (601 logged scalar points apiece) with **zero non-finite** loss or
gradient values, cosine LR fully annealed, final checkpoints at `iter=30000` with all six keys and
12,154,113-parameter model + EMA + optimizer + scheduler all finite, and eval sample renders at
every 5k interval. `grad_clip: 1.0` is proven active three ways (`train/grad_norm_preclip` scalars
only exist when clipping is on; the tfevents embeds the full config text; the pipeline log prints
`grad_preclip` every iteration): physics-on bursts peak at **4,229 @ iter 27,600** and clip on
14/601 steps, while physics-off bursts stay ≤ 1.31 after step 1 — confirming the Option-B physics
term as the burst source. The ablation's "3.6× lower loss" claim measures out at **3.575**
(final-100 means 0.01184 vs 0.00331).

**Two corrections made alongside the audits.**

1. `data/data_card.md`'s per-family validity split did not reproduce (the overall 87.7% = 2104/2400
   was exact). Recomputed with the committed filter: **cubic 800/800 (100%), gyroid 603/800 (75.4%),
   octet-truss 701/800 (87.6%)** — all 296 failures are `min_wall_thickness`, zero
   connectivity/overhang.
2. `runs/baseline_clipfix_30k/config.json` was reconstructed from the config text embedded in the
   run's own tfevents (the run predates config snapshotting) and carries an `_reconstructed`
   provenance key; it deep-equals the ablation's `train`/`model` blocks and differs only in
   `physics_lambda_1` / `physics_ablation`, as it must.

**Evidence pack (legacy checkpoint).** `python -m src.evidence` renders the "target → verified
candidate" loop into `outputs/acceptance_30k/evidence/` — reference-vs-generated grids for all 50
targets, the selected candidate's requested-vs-achieved properties and structural-validity verdict,
a 3D marching-cubes render, the STL export montage with watertightness, and an
`evidence_manifest.json` mapping every claim to its source artifact, entirely from recorded
artifacts with no FEA re-run. The corrected runs have parity and relative-error plots but no
equivalent 3D claim-to-artifact pack yet.

### 8.4 What's left

The binding lock-in checklist with per-item status is [`ROADMAP.md`](ROADMAP.md) §7.2. Open items are:

1. **Second seed.** A seed-1 physics-on/off pair on the V1f dataset is the highest-value robustness
   check (~5 GPU hours). Do not resume the cancelled partial run.
2. **Broader multi-draw uncertainty.** Current intervals describe draw variability on one seed and
   one fixed target set; additional seeds are needed for training-run uncertainty.
3. **Deadline.** Confirm the submission deadline and rescale the week-based timeline in `ROADMAP.md`
   §8.1.
4. **Human gates.** Clean-checkout reproduction by each team member and supervisor sign-off remain
   open.

**Acceptance status (summary):**
- [x] Pipeline runs end to end from dataset generation through evaluation and STL export; recorded
  evaluation sets report zero FEA failures.
- [ ] **All stated held-out property criteria are not uniformly met by the physics-on model.** V1f
  physics-on reaches 90.0% validity but held-out ν R² is 0.564. Physics-off reaches 94.0% validity
  and held-out R² 0.996/0.999/0.934. Off-manifold E R² remains below 0.85 in every corrected arm.
  Keep these results and caveats explicit; do not label the physics-on acceptance a clean pass.

Nothing above is committed. The working tree contains ongoing audit changes and earlier uncommitted
research/code/docs artifacts; stage, commit, or push only on explicit instruction.

## 9. Get running

```sh
export MAMBA_ROOT_PREFIX=$PWD/.mamba
micromamba run -n metamaterials python tests/smoke_test.py     # env check
micromamba run -n metamaterials python -m pytest tests -q      # 170 tests
micromamba run -n metamaterials ruff check src tests scripts  # lint
```

`RUN-BASE.md` is the end-to-end walkthrough (dataset → train → sample → evaluate → export →
evidence). See `README.md` for setup (micromamba + conda env + CUDA torch) and the six command-line
entry points.
