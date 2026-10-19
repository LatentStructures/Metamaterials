# AGENTS.md

Academic baseline project: physics-informed conditional diffusion (DDPM) for inverse design of synthesizable mechanical metamaterials. Team AA10. The intended package structure is `src/{geometry,fea,data,models,losses,validation,export}` per ROADMAP Section 2.

**Repo state:** env is built (torch 2.14.0+cu130 + dolfinx 0.11 in `.mamba/envs/metamaterials`); the full pipeline is implemented and green — `tests/smoke_test.py` (env DoD), `pytest tests` (94 passing), `ruff check src tests` (clean). The earlier real-FEA-generation crash (thread oversubscription → `MPI_Abort`) is fixed: `n_workers` is clamped to the CPU count and every worker process caps its BLAS/PETSc thread count. Remaining baseline work is **production execution only**: finish the real dataset generation — the resumable checkpoint `data/checkpoints/ckpt_dataset_s0.parquet` has ~950 completed cells of the current `configs/dataset.yaml` target (800/family = 2,400 total; reruns skip checkpointed `sample_index`es) — then the training/acceptance pair (ROADMAP §3.5, §4.7, §7). `sample_for_density(family, target, res)` returns the *achieved density* as its second value — never the parameter.

## Source of truth

`ROADMAP.md` is the binding baseline-phase spec. Read it before writing any code and obey its scope tags literally:
- **Untagged lines = baseline scope → implement exactly as specified.**
- **Contingency** items must be held until a named trigger fires (e.g., neural-surrogate loss only if acceptance metrics are missed).
- **Cut / Out of scope** and **Research track** items must never be implemented (e.g., DiT, cross-attention, SDF representation, mirror-reflection augmentation, additional topology families, resolution ≠ 32³, datasets >5,000 cells).

## Locked decisions — do not revisit

- FEA = FEniCSx (`dolfinx`), **not** legacy `dolfin`. Install via conda-forge with Python 3.11 (3.10 was dropped upstream); this is the most common setup blocker.
- Fixed **32³ binary voxel** resolution for the whole baseline.
- Exactly three topology families: cubic strut lattice, octet-truss, gyroid. Only these.
- Conditioning vector is `[E, relative_density, ν]` in that order; the manifest column order, HDF5 `/conditioning_vector`, and model input order must all match silently and exactly. Shear modulus G is **not** a conditioning target. Since the Sep-2026 cleanup the order is **enforced**, not just documented: `check_conditioning_order()` validates configs, the writer persists `/conditioning_order` in each HDF5 file, and the loader rejects files with a mismatched width/order. Do not "temporarily" reorder or extend the vector.
- Model: custom 3D conditional U-Net (channels [32,64,128,256], bottleneck self-attention at 4³, FiLM-style conditioning). Standard DDPM ε-MSE, T=1000; DDIM 50–100 steps at inference. No extra conditioning beyond FiLM.
- Physics loss: Option B only (differentiable connectivity proxy on soft voxel occupancy). Option A (neural surrogate) is a contingency.
- Augmentation: up to the 24 proper (rotation) symmetries of the cube, rotating the stiffness tensor to match — never mirror reflections.
- Homogenization: validate against ≥3 closed-form benchmarks within 5% *before* generating the full dataset; otherwise a wrong solver silently poisons all data.

## Environment setup

- Repo-local conda env (gitignored `.mamba/`): `MAMBA_ROOT_PREFIX=$PWD/.mamba`. Rebuild from `environment.yml`, then add CUDA torch with the exact pip line in README (`torch==2.14.0` from the `cu<ver>` PyTorch index, adjusted to your driver's CUDA major version — torch is intentionally not in `environment.yml`).
- Run anything in the env via `micromamba run -n metamaterials <cmd>`. The env is repo-local: without `MAMBA_ROOT_PREFIX=$PWD/.mamba` set first, micromamba looks under `~/.local/share/mamba` and fails with "prefix does not exist." Get it on every invocation (or export it in your shell) before running tests/linting.
- Python pinned 3.11 + `fenics-dolfinx` 0.11 (3.10 was dropped upstream).
- Gotchas:
  - dolfinx 0.11 `LinearProblem` requires a non-empty `petsc_options_prefix` AND an explicit `u=` Function — `Form.arguments()` is not exposed in Python, so without `u=...` it raises `AttributeError`.
  - conda `gmsh` is CLI-only (no `import gmsh`); use `meshio` for gmsh round-tripping or `pip install gmsh` in the env if the Python API is needed.
  - Real FEA generation and evaluation run on a spawned `ProcessPoolExecutor` (spawn, not fork). Every worker imports the entry module as `__main__`, so `generate_dataset.main` / `evaluate.main` guard with `multiprocessing.current_process().name != "MainProcess"`; keep that guard if you touch them. Each worker caps `OMP/OPENBLAS/MKL_NUM_THREADS` and caches the res-locked mesh + PBC pairings built once per resolution — do not rebuild the (~196k-tet) mesh per cell. Never run FEA workers with unlimited threads: that is what killed the original full run.
  - `torch.load(..., weights_only=True)` everywhere (checkpoints are plain tensors); no untrusted pickle loaders.
- Smoke test (env DoD): `micromamba run -n metamaterials python tests/smoke_test.py` — solid-cube homogenization, prints C1111.
- Gates on every change: `micromamba run -n metamaterials python -m pytest tests -q` and `micromamba run -n metamaterials ruff check src tests` (ruff is installed in the env; don't commit code that fails either). `tests/smoke_test.py` needs dolfinx and is the env-DoD check; the pytest suite covers the rest without it.

## Conventions

- Every run must be launchable from a single config file with a fixed seed; no hardcoded hyperparameters in scripts. `configs/{dataset,model_baseline,train_baseline}.yaml` already encode the locked decisions (conditioning order, 32³, channels, T=1000, DDIM 50–100, proper-rotation augmentation, physics option B) — new code must stay consistent, not re-derive them.
- Log every training run (W&B or TensorBoard); render sampled voxel grids at eval intervals, not just loss curves.
- Git: branch-per-feature, PR reviewed by ≥1 teammate, no direct pushes to `main`.
- Data binaries are gitignored (HDF5 one file per split + Parquet/CSV manifest); version binaries with DVC or equivalent checksums, keep the manifest in git.
- Checkpoint/resume: the generator's `data/checkpoints/ckpt_<config>_s<seed>.parquet` stores one row per completed cell with `C6_<k>` columns and `cond_{name}` conditioning columns (legacy checkpoints wrote `cond_E/cond_rho/cond_nu`; the loader falls back). `sample_index` keys resume; `voxel_path` inside a checkpoint is the bare `.npy` filename (relative — resolved against `data/checkpoints/vox/` so clones stay resumable), while the parquet manifest stores the absolute path. The Option-B proxy's power-iteration budget is config-driven (`physics_loss.iterations`, default 60 in-graph; the off-graph Fiedler reference uses 200) — don't hardcode it.
- Tests: FEA homogenization vs. analytical benchmarks (≤5%) and dataset loader shape/dtype/normalization sanity checks — run before merge.