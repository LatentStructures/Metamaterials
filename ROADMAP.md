# Roadmap: Physics-Informed Generative AI for Inverse Design of Synthesizable Functional Metamaterials

## Baseline Phase

> **How to read this document (scope tags).** This roadmap is the **baseline phase only**.
> Every line is baseline work to build **unless** a tag says otherwise. Implement exactly the
> tagged-and-untagged *baseline* scope: nothing more, nothing less.
>
> | Tag | Meaning | Your action |
> | --- | --- | --- |
> | *(no tag)* | Baseline work: build it exactly as specified | **Implement** |
> | **Contingency** | Needed only if a named trigger fires (acceptance metric missed, underfitting) | **Hold, do not implement** until the trigger fires |
> | **Cut / Out of scope** | Decided against for the baseline | **Do not implement** |
> | **Research track** | Post-baseline differentiator, chosen after the Section 7.2 lock-in | **Do not build now** |
>
> **The baseline = this exact scope:** 32³ binary-voxel dataset from three topology families
> (cubic strut, octet-truss, gyroid) → FEniCSx homogenization → 3D conditional U-Net + DDPM
> (FiLM conditioning, bottleneck self-attention) → Option B physics proxy → synthesizability
> filter → STL export.
>
> **Never implement in the baseline** (even though older drafts and the literature mention
> them): 128³ / any resolution change, mirror-reflection augmentation, cross-attention,
> a Diffusion Transformer, additional topology families, the neural-surrogate loss (Option A),
> or >5,000-cell datasets. The last two are **contingencies** and fire only under the
> conditions named in Phase 1, Section 3.1 and Phase 2, Section 4.4.

**Team AA10:** Nidish S R · Gowtham Kumaresan · Sriman Rakshan N · Lalith Kishore J

**Supervisor:** Dr. Abhijith Anandakrishnan

**Status:** Baseline phase. This pipeline is the shared foundation. It is deliberately scoped to standard, well-understood techniques (voxel representation, 3D conditional U-Net diffusion, FEniCS/PyVista homogenization) so it can be built quickly and correctly. Once it runs end-to-end and hits the acceptance criteria in Section 7, the team branches into the research-track differentiator it locks in.

---

## 0. Purpose, Scope & Assumptions

This document is the team's single source of truth for the **baseline phase** of the project: a complete, working, defensible physics-informed conditional diffusion pipeline for metamaterial inverse design, built on the original proposed methodology (voxel representation, 3D conditional U-Net, PDE-informed loss, FEA homogenization loop, synthesizability filter).

- **In scope:** linear-elastic mechanical metamaterials, unit-cell representation, binary voxel grids (32³), simulation-only validation (no physical fabrication yet), conditioning on the scalar mechanical targets (effective Young's modulus *E*, relative density ρ*, Poisson's ratio *ν*).
- **Explicitly out of scope for the baseline:** research-track differentiators (SDF/implicit representations, non-linear/large-deformation response, cyclic/hysteretic loading, Neural Operators, DFT/MD). Don't build these yet, but don't build the baseline in a way that blocks adding them later (see Section 7).

**Target duration:** Weeks 1–10 (~2.5 months of an 8-month total timeline), leaving 5.5 months for the research track, validation, and writing. Phase durations may overlap (e.g. the physics-loss term is built in parallel with the diffusion core).

**Definition of "baseline complete":** a reproducible, script-driven pipeline that:

1. Generates a labeled dataset of unit cells with FEA-verified elasticity tensors,
2. Trains a conditional diffusion model to generate new unit cells from target properties,
3. Samples new structures from held-out target properties,
4. Validates generated structures against a synthesizability filter and re-homogenizes them against ground-truth FEA (reporting property error), and
5. Exports validity-passing candidates as STL.

See the acceptance criteria and lock-in checklist in Section 7 before branching into the research track.

---

## 1. Tech Stack & Environment Setup

**Owner:** whole team | **Est. duration:** 2–3 days (Week 1)

| Layer | Tools | Notes |
| --- | --- | --- |
| Deep learning | PyTorch ≥2.2, (optional) PyTorch Lightning | Raw PyTorch is fine for a team this size; Lightning saves boilerplate if you want it |
| Diffusion utilities | Custom 3D U-Net (see Phase 2, Section 4.2): do not depend on Diffusers' 2D-centric abstractions | Diffusers' scheduler classes (DDPM/DDIM schedulers) are reusable even with a custom model |
| Geometry / meshing | PyVista, Trimesh, `scikit-image` (marching cubes), `meshio`, `gmsh` (via meshio) | |
| FEA | **Decided: FEniCSx (`dolfinx`)**, not legacy `dolfin` (better periodic BC support, actively maintained). Fallback: SfePy if `dolfinx` install is painful | Install via conda-forge in Week 1; this is the most common blocker |
| Data storage | HDF5 via `h5py` (one file per dataset split; chunked storage for voxel arrays) + a CSV/Parquet sample manifest (see Phase 1, Section 3.4) | |
| Experiment tracking | Weights & Biases (free academic tier) or TensorBoard | Log every training run, no exceptions. You'll need this for thesis/paper figures |
| Config management | Hydra or plain YAML + `argparse` | Every experiment should be reproducible from a config file alone |
| Environment | Conda env (`environment.yml`) or `uv`/Docker image, checked into repo | Pin Python 3.10/3.11 and all critical versions. FEniCSx dependency hell is real |
| Compute | 1× RTX 4090 (24GB) minimum per teammate for iteration; A100/cluster time reserved for full dataset generation and final training runs | 32³ voxels are small, so you don't need A100s for most of the baseline |
| Version control | Git, GitHub/GitLab, branch-per-feature, PR review by at least one other teammate before merging to `main` | Data binaries are gitignored and managed via DVC or a data manifest |

**Setup checklist:**

- [ ] Initialize the repository (structure in Section 2).
- [ ] Pin the environment (Python 3.10/3.11, `conda`/`uv` lockfile) with core deps: `torch`, `fenics-dolfinx`, `pyvista`, `trimesh`, `numpy`, `scikit-image`, `wandb`/`tensorboard`.
- [ ] Set up GPU access and confirm `torch.cuda.is_available()` on the actual training node, not just a laptop.
- [ ] Decide and document the fixed baseline voxel resolution: **32³** (rationale in Section 7).
- [ ] Set up experiment tracking before any training run, so every run from day 1 is logged, not just the "final" ones.

**Week 1 action item:** every team member gets the environment running and reproduces a trivial "hello world" FEA homogenization on a single cube before touching any generative code.

**Definition of done:** a teammate can `git clone`, create the env from the lockfile, and run a smoke-test script that generates one voxel cube, homogenizes it, and prints its stiffness tensor, with no manual fixes required.

---

## 2. Repository Structure

```text
Metamaterials/
├── environment.yml
├── README.md
├── ROADMAP.md
├── configs/
│   ├── dataset.yaml
│   ├── model_baseline.yaml
│   └── train_baseline.yaml
├── data/                          # raw + processed voxel datasets (gitignored; use DVC or a data manifest)
├── src/
│   ├── geometry/                  # unit-cell generators, voxelization utils
│   │   ├── lattice_families.py    # parametric unit cell generators
│   │   ├── voxelize.py            # implicit function -> voxel grid
│   │   └── mesh_utils.py          # voxel -> surface -> tet mesh
│   ├── fea/                       # FEniCS/PyVista homogenization scripts
│   │   ├── homogenization.py      # periodic BC + stiffness tensor extraction
│   │   ├── benchmarks.py          # analytical validation cases
│   │   └── property_extraction.py # C_ijkl -> E, nu, G
│   ├── data/
│   │   ├── generate_dataset.py    # end-to-end dataset build script (batch + checkpointing)
│   │   ├── dataset.py             # PyTorch Dataset/DataLoader, HDF5-backed
│   │   └── augment.py             # cubic symmetry augmentation
│   ├── models/
│   │   ├── unet3d.py              # conditional 3D U-Net
│   │   └── diffusion.py           # DDPM/DDIM process, noise schedule
│   ├── losses/
│   │   ├── physics_loss.py        # connectivity/proxy/neural-surrogate physics terms
│   │   └── README.md              # exact functional form of L_mechanics
│   ├── validation/                # synthesizability filter, connectivity checks
│   ├── export/
│   │   └── export_stl.py          # STL/3MF export
│   ├── train.py
│   ├── sample.py
│   └── evaluate.py                # FEA-verified evaluation + synthesizability filter
├── tests/
│   ├── test_homogenization.py     # against analytical benchmarks
│   └── test_dataset.py
└── notebooks/                     # exploration only, nothing load-bearing lives here
```

---

## 3. Phase 1: Dataset Generation & FEA Homogenization (Weeks 1–4)

**Owner:** Sriman Rakshan N (Simulation & FEA Lead), supported by Nidish (voxelization/data format) and Gowtham (validation of physics correctness). | **Est. duration:** 1.5–2 weeks for the geometry + homogenization core, then scale to full generation.

### 3.1 Unit Cell Topology Library

- **Decided (baseline starting library):** **cubic strut lattice, octet-truss, and gyroid (TPMS)**, three topologies, not five. Reasoning: the cubic lattice and octet-truss are required anyway for the closed-form validation step in Phase 1, Section 3.3 (Gibson & Ashby / Deshpande-Fleck scaling laws), so starting there means the validation benchmarks and the training-data topologies are the same geometries, so there is no wasted implementation effort. Gyroid adds real geometric diversity (smooth continuous surface vs. discrete struts) so the dataset isn't a strut-only monoculture, and it's the single most-published TPMS in the metamaterial-diffusion literature, so there's ample reference data to sanity-check your homogenized properties against even without a closed form.
  - **Strut-based (baseline):** cubic lattice, octet-truss, parameterized by strut radius and relative density.
  - **TPMS (baseline):** gyroid, generated from its closed-form implicit function, parameterized by level-set threshold (controls density) and unit cell period.
  - **Cut / Out of scope:** additional families (Kelvin foam, Schwarz-P, Diamond, honeycomb, perturbed strut). Revisit one only if the Phase 1 lock-in review demonstrates a dataset-diversity bottleneck, not before.
- Parameterize each of the three baseline families by 2–5 continuous knobs so you sample a design space rather than hand-crafting each cell.
- Sample continuous parameters (strut thickness / iso-value) to hit a target relative density range of roughly **0.1–0.6** (below 0.1 tends to produce disconnected structures; above 0.6 approaches bulk material and stops being an interesting "metamaterial" regime).
- **Target dataset size:** **2,000–5,000** labeled cells is enough to train a small conditional model without requiring a cluster. This is the baseline target. Scaling toward **5,000–10,000** is a **contingency only**: revisit it if the per-solve wall-clock log shows it's realistic *and* the model shows signs of underfitting; don't spend Phase 1 time on it before then. Document the actual size as a known limitation.

### 3.2 Voxelization & Meshing Pipeline

1. Sample each implicit/parametric function on a **32³** grid → binary occupancy voxel array. (**Research track:** 32³ matches the original proposal's Stage 1 resolution and keeps training tractable; higher resolutions are not a baseline requirement. Do not implement them.)
2. Use **consistent periodic boundary handling at the rasterization step**. Decide once, document it, and never revisit it ad hoc later.
3. Surface extraction via marching cubes (`skimage.measure.marching_cubes` or PyVista's `contour`).
4. Mesh repair with Trimesh: fill holes, remove duplicate/degenerate faces, confirm watertight (`mesh.is_watertight`). Reject and re-sample any unit cell that fails repair.
5. Tetrahedral volume meshing via `gmsh` (through `meshio`) for the FEA solver (or use a voxel-based FE homogenization method that skips explicit meshing; your choice, just validate it).

### 3.3 FEA Homogenization

This is the technical core of Phase 1. Get it right and validated before scaling up dataset generation.

- **Method:** computational (asymptotic) homogenization under periodic boundary conditions. For each unit cell, solve 6 independent unit macroscopic strain load cases (3 normal + 3 shear) to recover the full effective stiffness tensor **C_ijkl**. Cubic-symmetric families (e.g. the cubic strut lattice) can get away with fewer, as few as 2 (one normal, one shear), since cubic symmetry has only 3 independent elastic constants. **Don't apply that shortcut to orthotropic families**: orthotropic materials have 9 independent constants and still need all 3 shear cases to recover the shear moduli, which the normal-strain cases alone can't give you. When in doubt, just run all 6, since it's not much more expensive per cell and removes an entire class of subtle bugs.
- **Before implementing from scratch:** Andreassen & Andreasen, *"How to determine composite material properties using numerical homogenization,"* Computational Materials Science, 2014, is the standard practical reference for exactly this task (voxel-grid-based numerical homogenization for metamaterial/composite unit cells) and ships with a compact, well-tested MATLAB implementation. Even implementing in FEniCSx rather than MATLAB, porting the logic from a validated reference will save real debugging time versus deriving the PBC formulation from a general elasticity textbook.
- **Implementation in FEniCSx:**
  - Define the linear elasticity weak form on the tetrahedral mesh.
  - Implement periodic boundary conditions via multi-point constraints (MPC) linking opposite faces of the unit cell. This is the fiddliest part of the whole pipeline; budget extra time.
  - For each load case, solve for the periodic displacement fluctuation field, then compute the effective stiffness via the strain-energy homogenization formula:

    **C̄_ijkl = (1/|Y|) ∫_Y C_ijmn(y) (ε̄_mn^kl + ε_mn(u^kl)) dY**, integrated over the unit cell domain Y, where `u^kl` is the periodic fluctuation field solved for under the unit macroscopic strain case `ε̄_kl`, and the local stiffness `C_ijmn(y)` is contracted against the *local* strain field (`mn`), not the load-case index (`kl`). A notation slip here is easy to carry straight into buggy code, so double-check this against the Andreassen & Andreasen reference above before implementing.

- **Post-processing:** reduce the full C_ijkl tensor to the scalar conditioning targets that define the baseline: effective Young's modulus E and Poisson's ratio ν, via standard tensor-to-engineering-constant formulas, assuming isotropic or orthotropic symmetry as appropriate per family. Shear modulus G is **not a baseline target**: compute it only if you want an extra evaluation metric, and never add it to the conditioning vector (Phase 2, Section 4.1).
- **Validation (do this before generating the full dataset):** run the homogenization pipeline on ≥3 unit cells with known closed-form solutions (e.g., simple cubic lattice and octet-truss scaling laws from Gibson & Ashby / Deshpande-Fleck theory) and confirm agreement within **5%**. This validation is a required Phase 1 deliverable. **Do not skip it, because a silently wrong homogenization script poisons the entire dataset.**
- **Batch generation:** make the homogenization script run unattended overnight across N generated cells, with checkpointing (resume from cell K if the job dies at cell K+1, don't restart from zero). Log wall-clock time per FEA solve so you know how large a dataset is realistic on your hardware.

### 3.4 Dataset Assembly & Storage

- **Sample manifest:** a CSV/Parquet manifest mapping `cell_id → voxel_grid_path → (E, ρ*, ν)`, with the analytical validation case passing to within a documented tolerance (5%). **Order matters here**: keep this identical to the `/conditioning_vector` column order below and the model's input order in Phase 2, Section 4.1; a silent mismatch between manifest order and array order is a classic source of a model that trains fine but conditions on the wrong property. Keep the manifest in git; version the binary HDF5 files with DVC (or equivalently robust checksummed storage); never commit binaries.
- **HDF5 schema** (one file per dataset split, chunked storage for voxel arrays):
  - `/voxels`: shape (N, 32, 32, 32), `uint8`
  - `/stiffness_tensor`: shape (N, 6, 6), `float32`
  - `/conditioning_vector`: shape (N, 3), `float32` → [E, relative_density, ν], normalized using dataset-wide mean/std (store the normalization stats alongside the file)
  - `/metadata`: topology family (categorical), generating parameters, per-sample validation flags
- **Splits:** 80/10/10 train/val/test, **stratified by topology family and density bucket** so the test set isn't accidentally dominated by one family, and checked to ensure near-duplicate cells don't leak across splits (relevant if your generator produces near-identical variants).
- **Augmentation:** each unit cell admits up to **24 proper rotations** (the full rotation group of the cube) that are physically valid label-preserving transforms for the stiffness tensor (the tensor itself must be correspondingly rotated, not just the voxel grid; don't skip this step, it's a common source of silent bugs). This is a free ~24x data multiplier. **Cut / Out of scope:** do **not** add mirror reflections, because the flipped face-winding/normals risk in the marching-cubes/STL-export step isn't worth a marginal data bump for a baseline.

### 3.5 Phase 1 Deliverables (Definition of Done)

- [ ] `generate_dataset.py` runs end-to-end from a config file with a fixed seed and reproduces the same dataset.
- [ ] Homogenization pipeline validated against ≥3 analytical benchmarks, all within 5% error, results written up in `tests/test_homogenization.py` and a short validation note.
- [ ] ≥2,000 labeled cells (pre-augmentation; scale toward 5,000–10,000 only if the contingency in Phase 1, Section 3.1 triggers) in versioned HDF5 storage + a Parquet manifest, with a one-page "data card" documenting family distribution, density range, and any rejected/failed samples.

---

## 4. Phase 2: Baseline Conditional Diffusion Pipeline (Weeks 5–10)

**Owner:** Nidish S R (Generative Core Lead) for architecture/sampling, Gowtham Kumaresan (Physics & Loss Lead) for the physics-informed loss term, Sriman Rakshan N (Simulation & FEA Lead) for the FEA-verified evaluation loop in Phase 2, Section 4.6, Lalith Kishore J (Product & Manufacturing Lead) for the synthesizability filter and STL export. | **Est. duration:** the loss-term work overlaps with the generative-core work (build Phase 2, Sections 4.4 and 4.2–4.5, in parallel).

### 4.1 Data Representation & Conditioning

- Baseline representation: **binary voxel occupancy**, 32³ (matches Phase 1 output directly; do not switch to SDF for the baseline, as that's a research-track decision).
- Conditioning vector **v = [E_norm, ρ_norm, ν_norm]**, normalized with the stats stored in Phase 1.

### 4.2 Model Architecture: 3D Conditional U-Net

- **Input:** noisy voxel tensor (B, 1, 32, 32, 32) + timestep + conditioning vector.
- **Encoder:** 4 downsampling stages, channel widths [32, 64, 128, 256], each stage = 3D conv → GroupNorm → SiLU, wrapped in residual blocks.
- **Bottleneck:** self-attention at the lowest resolution (4³). At 64 tokens it's cheap in compute (~5–10% training overhead) and a standard, low-risk block, so keep it in v1.
- **Decoder:** mirrored upsampling stages with U-Net skip connections from the encoder.
- **Output:** predicted noise ε (standard ε-parameterization), same shape as input.
- **Time embedding:** sinusoidal positional embedding → 2-layer MLP, injected into every residual block (standard DDPM practice).
- **Conditioning embedding:** separate small MLP mapping [E, ρ, ν] → same dimensionality as the time embedding; sum or concatenate before injection (FiLM-style modulation, which is clean, simple, and sufficient for three scalar targets; skip the extra complexity of cross-attention).
- **Research track:** don't start with a Diffusion Transformer for baseline. It is not a baseline requirement, so do not implement it.
- **Toy test first:** confirm the model actually trains by overfitting on ~10 samples and verifying near-perfect reconstruction before training on the full dataset, because this catches architecture/shape bugs early and cheaply.

### 4.3 Diffusion Process

- Standard DDPM formulation (Ho et al., 2020), linear or cosine β-schedule, **T = 1000** training steps.
- Training loss: simple ε-MSE (`L_simple`).
- Inference sampler: **DDIM with 50–100 steps** to keep generation fast enough for iterative evaluation during development. Target: sample a voxel grid in under ~1 minute on the training GPU.

### 4.4 Physics-Informed Loss (Baseline Simplification)

Running a full FEA solve inside the training loop, every step, is computationally prohibitive. For the baseline, implement a **lightweight differentiable proxy** rather than the true PDE residual:

- **Option B (BASELINE, implement this):** a differentiable connectivity/force-equilibrium proxy computed on the (soft, pre-threshold) voxel occupancy, e.g., a soft penalty on isolated/floating voxel clusters via a differentiable relaxation of connected-component labeling, which discourages obviously unphysical structures without requiring a full elasticity solve. It is the cheapest defensible choice: no extra model to pretrain, no surrogate-bias risk, and it reuses the connected-component logic already needed for the synthesizability filter (Phase 2, Section 4.6). It protects the validity-rate target but does not steer properties toward the conditioning targets, and that weakness is exactly what Option A fixes.
- **Option A (stronger fidelity; contingency, not baseline scope):** if reconstruction-error/R² targets aren't being met, train a small neural surrogate (a simple 3D CNN regressor on the Phase 1 dataset predicting [E, ρ, ν] from a voxel grid; this takes a day, not a week) and use it as a frozen differentiable critic during diffusion training to penalize the denoised x₀ prediction's surrogate-predicted properties against the conditioning target.

**L_total = L_diffusion + λ₁ · L_physics_proxy**, with λ₁ warmed up from 0 over the first ~20% of training (let the model learn basic geometric structure before physics constraints kick in).

**Recommended path:** build Option B first and run it through one full training pass + ablation; layer Option A on top only if the acceptance metrics in Section 7.1 aren't met.

- **Run an ablation:** train one model with λ₁ = 0 and one with λ₁ > 0 so you have baseline evidence the physics term actually helps. This is your first piece of "results" for any report.
- **Document the exact functional form** of the loss in `/src/losses/README.md`. It's the piece most likely to need revision when you pivot to the research track, so make it legible to future-you.
- **Document this simplification explicitly** in the thesis/paper. The true PDE-residual injection from the original proposal (Stage 2) becomes a research-track upgrade once the baseline works end-to-end.

### 4.5 Training Loop

- Optimizer: AdamW, lr 1e-4, cosine decay, EMA of weights (decay 0.9999).
- Batch size: 32³ volumes are small, so batch sizes of 32–64 should fit comfortably on a 24GB GPU.
- Mixed precision (bf16 preferred if your GPU supports it).
- Checkpoint every N steps; log loss curves and rendered sample voxel grids (via PyVista) to W&B every eval interval, because sample quality matters as much as loss numbers.

### 4.6 Evaluation & Synthesizability Filter

1. Generate N samples conditioned on held-out test-set targets.
2. Run **every** generated sample through the Phase 1 FEA homogenization loop (not the neural surrogate) to get ground-truth-verified achieved properties. This closes the loop and gives their **true** effective properties for the property-reconstruction error check.
3. Report reconstruction error between requested and FEA-verified (E, ρ*, ν) on validity-passing samples.
4. **Synthesizability filter:**
   - Connectivity check: single connected solid component, zero floating/unconnected voxels (connected-component labeling on the binary voxel grid).
   - Minimum wall thickness check via morphological erosion.
   - Overhang angle / manufacturability estimate for additive manufacturing (approximate; full support-structure analysis is out of scope for baseline).
5. Export validity-passing structures to STL via marching cubes (`scikit-image` or PyVista), sanity-check a few in a mesh viewer for watertightness before assuming the export pipeline works.

### 4.7 Phase 2 Deliverables (Definition of Done)

- [ ] `train.py`, `sample.py`, `evaluate.py` all runnable from config files.
- [ ] Trained baseline checkpoint with a logged W&B run (loss curves, sample grids over training).
- [ ] Evaluation report: property reconstruction error, validity rate, FEA-verified error distribution, with plots.
- [ ] `export_stl.py` producing valid, watertight STL files for top-N generated candidates, manually inspected in a mesh viewer by at least one team member.

---

## 5. Engineering Practices (Applies Throughout Both Phases)

- **Git workflow:** feature branches off `main`/`dev`, at least one teammate reviews every PR before merge. No direct pushes to `main`.
- **Testing:** unit tests for the FEA pipeline (the analytical benchmark cases from Phase 1, Section 3.3) and the dataset loader (shape/dtype/normalization sanity checks), plus tests for geometry and loss functions. Run them before every merge to `main`, whether via CI or manually. Your choice given team size, but run them.
- **Reproducibility:** every training run must be launchable from a single config file with a fixed seed; no hardcoded hyperparameters in scripts.
- **Documentation:** keep the README current: setup instructions, how to reproduce the dataset, how to launch training, how to reproduce evaluation numbers. Treat it as if a 5th teammate will join in month 3 with zero context. Keep `/src/losses/README.md` current too.

---

## 6. Milestone Timeline

| Week | Focus | Primary Owner | Deliverable |
| --- | --- | --- | --- |
| 1 | Environment setup, repo scaffolding, FEniCSx "hello world" | All | Working dev environment for every teammate; `git clone` → env → smoke-test pass |
| 1–2 | Lattice family generators + voxelization | Nidish | `geometry/` module |
| 2–4 | FEA homogenization + PBC implementation | Sriman | `fea/` module + analytical validation (≤5%) |
| 4 | Full dataset generation run (batch, checkpointed) | Sriman + Nidish | HDF5 dataset + manifest, data card |
| 5–6 | 3D conditional U-Net + DDPM training loop (incl. 10-sample overfit test) | Nidish | `models/` module, first training run |
| 6–7 | Physics-loss proxy integration + λ₁ ablation | Gowtham | `physics_loss.py`, ablation logged |
| 7–8 | Full baseline training + hyperparameter passes | Nidish + Gowtham | Trained checkpoint |
| 8–9 | FEA-verified evaluation loop (re-homogenization) | Sriman | Evaluation report |
| 9–10 | Synthesizability filter + STL export | Lalith | `export_stl.py`, validated STL outputs |
| 10 | Baseline lock-in review + acceptance run (≥50 unseen targets) | All + Supervisor | Go/no-go decision on Section 7 checklist |

---

## 7. Acceptance Criteria, Baseline Lock-In & Bridge to Research Track

### 7.1 Baseline Acceptance Criteria

Run the full pipeline end-to-end (target properties in → STL files out) on a batch of **at least 50 unseen target property vectors**. The targets below are starting benchmarks, not guarantees. Adjust them once you have real numbers from early runs.

| Metric | Baseline target |
| --- | --- |
| Structural validity rate | ≥ 80% (starting benchmark); **≥ 90%** target for lock-in |
| Property reconstruction error | ≤ 15% mean relative error (validity-passing samples only) |
| R² on FEA-verified property reconstruction | ≥ 0.85 |
| Generation time per sample | Reported, not gated (DDIM ~50–100 steps; <1 min target), because the baseline doesn't need to be fast |
| Physics-loss ablation | Documented, even if the effect is small |

> **Why 32³ for baseline:** higher-resolution voxel diffusion is a significant VRAM/training-time jump for marginal baseline value. It's a resolution upgrade, not a validation of the *pipeline logic*. Get the full loop correct at 32³ first; don't budget time for any resolution change until the plumbing (dataset → train → sample → validate → export) is proven and the research-track direction is locked in.

### 7.2 Baseline Lock-In Checklist

Before branching into the research track (SDF representation, non-linear/hysteresis targets, or neural-operator surrogate), confirm:

- [ ] Dataset generation is reproducible from a fixed seed.
- [ ] FEA homogenization validated against analytical benchmarks (≤5% error).
- [ ] Diffusion model trains stably and produces non-degenerate samples (not mode-collapsed to a single topology).
- [ ] FEA-verified R² ≥ 0.85 and validity rate ≥ 90% on the held-out test set.
- [ ] At least one generated candidate successfully exported to STL and visually inspected as structurally sane.
- [ ] All four team members can independently run the full pipeline (dataset → train → sample → evaluate) from a clean checkout.
- [ ] Supervisor sign-off on baseline results before committing engineering time to the research track.

Only once every box above is checked should the team commit to a single research-track differentiator.

### 7.3 Bridge to the Research Track

This baseline is designed to fail gracefully into either pivot direction, but **none of the following is baseline work; do not implement any of it now**:

- **Unified integration track (Option A):** extend the physics loss of Phase 2, Section 4.4 with an uncertainty-aware term and a manufacturing-defect-robustness term, reusing the same dataset/training loop, with no architectural rewrite needed.
- **SDF + Neural Operator + latent diffusion track (Option B/C):** the homogenization script and dataset manifest are reusable almost as-is; the geometry generator additionally exports a continuous SDF representation (not just voxel occupancy) for the same unit cells, and the U-Net gets replaced/wrapped rather than the whole pipeline discarded.

Keep this in mind while building Phase 1 and Phase 2: don't hard-code voxel-only assumptions into the dataset manifest schema or the FEA wrapper's input interface if it costs nothing to keep them generic now.

---

## 8. Open Items & Appendix

### 8.1 Open Decisions

- [x] **FEA toolchain:** decided: FEniCSx (`dolfinx`).
- [x] **Baseline topology library:** decided: cubic strut lattice, octet-truss, gyroid (Phase 1, Section 3.1); additional families cut (see Phase 1, Section 3.1).
- [ ] **Remaining calendar time until submission/defense:** still needed from the team. The Week 1–10 baseline schedule in Section 6 assumes an ~8-month total timeline (Section 0). If your actual remaining time differs, the week numbers throughout this document (not just the total) need rescaling before Week 1 starts, since several deliverables (full dataset generation, first training run, evaluation loop) are sequenced with real dependencies, not just labeled with arbitrary week numbers.

### 8.2 Key References (carried over from literature review)

- Ho, Jain, Abbeel: *Denoising Diffusion Probabilistic Models* (DDPM), 2020
- Song et al.: *Denoising Diffusion Implicit Models* (DDIM), 2021
- Gibson & Ashby: *Cellular Solids: Structure and Properties* (scaling laws for FEA validation)
- Andreassen & Andreasen: *"How to determine composite material properties using numerical homogenization,"* Comput. Mater. Sci., 2014 (practical reference + example code for Phase 1, Section 3.3)
- Full project literature review (8 papers): see original project presentation

### 8.3 Open Questions to Revisit in the Research Phase

- Whether the neural-surrogate physics-loss proxy (Phase 2, Section 4.4, Option A) is a sufficient stand-in for true PDE-residual injection, or whether it introduces a systematic bias worth quantifying.
- Whether 32³ resolution is sufficient for the research-track topology family eventually chosen (compute cost implications).
- Final choice of research-track differentiator (representation swap vs. physics-target swap vs. unified integration). Decision deferred to the checkpoint after baseline lock-in.
