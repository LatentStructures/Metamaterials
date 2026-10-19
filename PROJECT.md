# Project Overview

An easy-to-read tour of this repository. For the binding spec, scope tags, and
acceptance criteria, read `ROADMAP.md`; this file only explains *what* the
project is and *how* the code fits together.

## 1. What is this project?

We build **mechanical metamaterials by design** using AI.

A *mechanical metamaterial* is a material whose properties come not from its
chemistry but from its **microstructure** — the tiny repeating pattern
("unit cell") inside it. Think of a sponge: it is mostly air, yet it can
support weight with the right structure. By choosing the 3D pattern of solid
and void inside a unit cell, we can get unusual effective properties, like
deforming sideways when you squeeze it, or being almost incompressible while
staying very light.

The hard part is the *inverse* problem:

> **Forward:** given a unit-cell pattern, compute its properties (easy-ish —
> simulate it).
> **Inverse:** given desired properties, find a unit-cell pattern that has them
> (very hard — infinite possible patterns).

Our answer: train a **conditional generative model** (a type of AI, a
diffusion model) that has seen thousands of "pattern → property" examples, so
that when you give it a target like *"normalized Young's modulus E/E_s = 0.1,
relative density ρ\* = 0.3, Poisson ratio ν = 0.2"*, it draws a new 3D unit cell that should
achieve those values. We then **verify every generated design with a physics
simulation (FEA)** before shipping it, and export the survivors as STL files
ready for a 3D printer.

**Everything in this baseline is simulation-only.** We do not fabricate
anything; "manufacturable" means the structure passes automated checks that
predict whether it *could* be printed.

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

- **cubic_strut** — three sets of rods running along the x, y, z directions.
  Parameter: rod radius.
- **octet_truss** — the classic strong lightweight truss (face-centered-cubic
  nearest-neighbour struts). Parameter: strut radius.
- **gyroid** — a smooth, undulating 3D surface described by a math formula
  (a "triply periodic minimal surface"). Parameter: the level-set threshold,
  which controls solid fraction.

For a chosen target density, `sample_for_density` performs a bisection search
over the parameter to hit the requested density band (0.1–0.6). Geometries are
**periodic**: a strut that leaves one face of the cell continues on the
opposite face, exactly matching what the FEA solver assumes.

The rasterization (`src/geometry/voxelize.py`) turns each shape into a binary
`32×32×32` grid, and `src/geometry/mesh_utils.py` converts that grid into a
tetrahedral mesh for the solver (with one subtlety — a translation-invariant
Kuhn triangulation — so the mesh has no cracks between neighbouring voxels).

### 4.2 The physics — FEA homogenization

`src/fea/homogenization.py` implements **computational homogenization**
(the standard method of Andreassen & Andreasen, 2014):

1. The unit cell is turned into a tetrahedral mesh.
2. **Periodic boundary conditions** are enforced exactly: the displacement on
   each +face is tied to the corresponding point on the −face (master–slave
   constraints), which is the fiddliest part of the whole pipeline.
3. Six "unit strain" load cases are solved (3 stretch + 3 shear), giving the
   full 6×6 stiffness tensor **C** of the effective material.

`src/fea/property_extraction.py` reduces C to the three conditioning scalars:
`E`, `ρ*`, and `ν` (via Voigt–Hill averaging for near-cubic cells). The shear
modulus G is computed only as a side metric — it is deliberately **not** a
conditioning target.

**Validation gate:** before ANY dataset is allowed to be generated, the solver
must reproduce three analytically-known cases within 5% — a solid cube, a
laminate, and aligned rods (`src/fea/benchmarks.py`,
`validate_vs_analytical`). A silently wrong solver would poison every label, so
this check is mandatory and stored in `data/validation_table.parquet`.

### 4.3 The AI — conditional diffusion (DDPM + U-Net)

- **Diffusion process** (`src/models/diffusion.py`): DDPM training with
  T = 1000 timesteps. Training = add noise to a real cell, then teach the model
  to predict the noise back (standard ε-MSE loss). Sampling = start from pure
  noise and iteratively remove it, guided by the target properties, using a
  **DDIM** sampler (50–100 steps instead of 1000, for speed).
- **Base architecture** (`src/models/unet3d.py`): a 3D U-Net with four
  resolution stages (channel widths 32→64→128→256), residual blocks, and
  **self-attention** at the lowest 4³ resolution.
- **Conditioning** = a small MLP that maps `[E, ρ*, ν]` into the network via
  **FiLM** (scale-and-shift) layers, so the drawn structure adapts to the
  requested properties. The conditioning order `[E, relative_density, nu]` is a
  **locked invariant** enforced across the manifest, the HDF5 files, and the
  model.

### 4.4 The physics-informed loss

Running a full FEA solve on every training step would be prohibitively slow.
Instead we add a cheap, differentiable **connectivity proxy** (Option B,
`src/losses/physics_loss.py`): an algebraic measure (Fiedler eigenvalue of a
soft voxel graph) that is ~0 for a well-connected structure and ~1 for one with
floating/disconnected parts. It discourages the model from producing obviously
unphysical, disjoint debris. The total loss is

```
L = L_diffusion + λ₁(t) · L_connectivity
```

with λ₁ warmed up from 0 over the first 20% of training. (A stronger "neural
surrogate" critic — Option A — exists only in the docs; it is a contingency
and is not built.)

### 4.5 Verification, filter, export

- `src/sample.py`: draws cells for targets from a CSV file.
- `src/evaluate.py`: re-homogenizes every generated cell with the **real** FEA
  solver (not the model), so we know the *achieved* properties; applies the
  **synthesizability filter** — single connected component,
  minimum wall thickness, overhang estimate; reports validity rate and
  reconstruction error (target vs. achieved) with parity plots.
- `src/export/export_stl.py`: ranks valid candidates by reconstruction error,
  exports the top-N to smooth STL, and checks watertightness.
- **Augmentation** (`src/data/augment.py`): each cell can be rotated by any of
  the 24 proper cube rotations to multiply the effective dataset ~24× — but
  only *proper rotations* (no mirror reflections), and the stiffness tensor is
  rotated to match so labels stay correct.

## 5. Data format

One HDF5 file per dataset split (`data/dataset_{train,val,test}.h5`):

| Dataset | Contents |
| --- | --- |
| `/voxels` | `(N, 32, 32, 32)` uint8 binary grids |
| `/conditioning_vector` | `(N, 3)` float32 `[E_norm, ρ*_norm, ν_norm]` (normalized; mean/std stored as attributes) |
| `/stiffness_tensor` | `(N, 6, 6)` float32 effective stiffness C |
| `/metadata` | family, generating parameters, validity |
| `/conditioning_order` | literal `["E", "relative_density", "nu"]` so loaders can reject mismatches |

Plus `data/manifest.parquet` (cell_id → split → family → properties → file
paths) and `data/data_card.md` (a one-page summary of the dataset). Splits are
80/10/10, **stratified by family and density bucket**, and checked for
near-duplicate leakage across splits. Data binaries are gitignored.

## 6. Where things live

```
configs/          One YAML per run type: dataset, model, train, evaluate
src/config.py     Config loaders + the locked conditioning-order check
src/geometry/     Unit-cell generators, 32³ rasterization, mesh building
src/fea/          FEniCSx homogenization, property extraction, benchmarks
src/data/         Dataset generation, HDF5 loader, rotation augmentation
src/models/       3D conditional U-Net + DDPM/DDIM diffusion
src/losses/       Option-B connectivity proxy (+ functional-form README)
src/validation/   Synthesizability checks (re-exported from evaluate)
src/export/       STL export with watertightness reporting
src/train.py      Training loop (EMA, cosine LR, bf16, W&B/TensorBoard)
src/sample.py     Conditional DDIM sampling → .npy / STL
src/evaluate.py   FEA re-verification + filter + error report
src/voigt.py      The single shared Voigt index convention
tests/            113 passing tests + tests/smoke_test.py (env check; counts may increase)
```

## 7. Key decisions (locked — do not revisit)

- **32³ binary voxels** for the whole baseline (no higher resolution).
- **Exactly three** topology families (cubic strut, octet-truss, gyroid).
- Conditioning vector order is fixed as `[E, relative_density, nu]`; G is never
  a conditioning target.
- FEA = **FEniCSx** (`dolfinx`), never the legacy `dolfin`.
- Model = custom 3D U-Net with FiLM conditioning and bottleneck self-attention;
  no cross-attention, no DiT.
- Physics loss = **Option B** proxy only; the neural-sanitized surrogate
  (Option A) is a contingency.
- Augmentation = the 24 proper cube rotations only, never mirrors.
- Standard DDPM (T=1000) for training; **DDIM 50–100 steps** at inference.
- Every run reproducible from a single config file + fixed seed; the FEA
  workers cap their thread counts (this is what prevents the OOM/oversubscription
  crash that killed an earlier full dataset run).

## 8. Current status & what's left

**Implemented and green at test scale:** the full pipeline is coded and tested — smoke test +
113 pytest cases pass, `ruff` is clean. Geometry, FEA homogenization (validated
against benchmarks), dataset writer/loader, U-Net, DDPM/DDIM, training loop,
sampling, FEA-verified evaluation, synthesizability filter, and STL export all
exist and work end-to-end on a smoke scale.

**Remaining = production execution only:**

1. **Finish generating the real dataset** (the current config targets 800 cells
   per family = 2,400 total; ROADMAP's baseline target is ≥2,000). The current
   resumable checkpoint is documented in `AGENTS.md`; complete generation and
   confirm the final manifest/HDF5 counts before training.
2. **Train** the conditional DDPM (`configs/train_baseline.yaml`, logged to
   W&B, run the λ₁ ablation pair).
3. **Acceptance run** (ROADMAP §7): ≥50 unseen targets → sample → FEA-verify →
   report validity rate (≥90% target), reconstruction error (≤15% mean), and
   R² (≥0.85); export top candidates to STL.
4. Then the team can lock in a research-track differentiator.

## 9. Get running

```sh
export MAMBA_ROOT_PREFIX=$PWD/.mamba
micromamba run -n metamaterials python tests/smoke_test.py     # env check
micromamba run -n metamaterials python -m pytest tests -q      # 93 tests
micromamba run -n metamaterials ruff check src tests           # lint
```

See `README.md` for the full setup (micromamba + conda env + CUDA torch) and
the five command-line entry points.
