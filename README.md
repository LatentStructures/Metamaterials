# Metamaterials

Physics-informed conditional diffusion for inverse design of synthesizable mechanical metamaterials (voxel + 3D U-Net + FEniCSx homogenization).

**Results:** see `outputs/README.md` for the headline numbers (92.0% on-manifold, 97.2% off-manifold, 94.0%/98.1% λ₁=0 ablation) and which files hold the sample grids, parity plots and printable STLs. The full write-up is `PROJECT.md`; the step-by-step runbook is `RuntheBaseline.md`.

## Setup

`ROADMAP.md` is the binding spec — read Section 1 (tech stack) and Section 2 (repo structure) before touching code.

1. Install `micromamba` (conda-forge toolchain; dolfinx has no reliable PyPI wheel, so `uv`/pip can't install FEniCSx):

   ```sh
   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
   install -m 755 bin/micromamba ~/.local/bin/micromamba
   ```

2. Create the environment. For a repo-local env (gitignored via `.mamba/`):

   ```sh
   export MAMBA_ROOT_PREFIX="$PWD/.mamba"   # keep the whole env inside this repo
   micromamba env create -f environment.yml
   ```

   Otherwise use your default conda root prefix and plain `micromamba env create -f environment.yml`.
3. Install PyTorch with CUDA (CUDA wheels are machine-specific, so torch is installed separately rather than pinned in `environment.yml` — adjust `cu130` to your driver's CUDA major version):

   ```sh
   micromamba run -n metamaterials pip install "torch==2.14.0" \
       --index-url https://download.pytorch.org/whl/cu130
   ```

   Verify GPU: `micromamba run -n metamaterials python -c "import torch; print(torch.cuda.is_available())"` must print `True`.
4. Smoke test (env + FEniCSx working end-to-end):

   ```sh
   micromamba run -n metamaterials python tests/smoke_test.py
   ```

## Usage

All commands below run from the repo root. Prefix them with
`MAMBA_ROOT_PREFIX=$PWD/.mamba` if you created the repo-local env (see above).
Every run is reproducible from a single config file plus a fixed seed.

```sh
# 1. Build the labeled dataset (FEA homogenization + HDF5 splits + manifest).
#    Writes data/dataset_{train,val,test}.h5, data/manifest.parquet, data/data_card.md.
#    The parameterized geometry run keeps its own checkpoint (dataset_v2), so
#    it never overwrites an earlier run's checkpoint.
micromamba run -n metamaterials python -m src.data.generate_dataset configs/dataset.yaml
#    --mock-fea is pipeline testing ONLY (unphysical placeholder stiffness).

# 2. Train the conditional DDPM (logs to W&B or TensorBoard).
#    Defaults to configs/train_baseline.yaml + configs/model_baseline.yaml,
#    which fits the 8 GB GPU at batch_size 32 (validated by the 30k run).
micromamba run -n metamaterials python -m src.train
#    Flags: --train-config, --model-config, --overfit N (10-sample architecture
#           test), --max-iterations N, --resume-from <ckpt.pt>, --run-dir <dir>.
#    Run the zero-physics ablation with --physics-ablation (tags the run
#    `abl_physics`); the default config trains with Option B at lambda_1=0.05.
#    `optimizer.grad_clip: 1.0` is required — without it the physics gradient
#    diverges.

# 3. Sample from held-out targets (CSV columns: E, relative_density, nu).
#    The 50 held-out acceptance targets live in data/targets_test_50.csv.
#    For the off-manifold control, regenerate the 108 in-between targets with
#    `python -m src.data.build_synthetic_targets` (writes data/targets_synthetic.csv).
#    Its `--nu-jitter` (default 1.0) scales the scatter around each family's fitted
#    nu = f(E, rho) relation; do NOT draw nu uniformly over the band — that produces
#    physically unreachable targets (fixed 2026-10-20; see PROJECT.md section 8).
#    The canonical repo target set uses `--nu-jitter 0.25 --seed 0 --with-family`.
#    Flags: --steps N (DDIM 50-100), --n-per-target N, --seed N,
#           --weights {ema,live} — default is live. The EMA arm is a documented
#           decay/schedule artefact (14% vs 92% validity; decay 0.9999 averages
#           a less-converged region — see PROJECT.md section 8). Use live for
#           every acceptance/headline number, --export-stl
#           (quick surface dump; it does NOT rank or check watertightness).
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/<run>/ckpt.pt --targets targets.csv

# 4. Re-homogenize every sample with real FEA, apply the synthesizability
#    filter, and report reconstruction error (outputs/evaluate/).
#    Run ONE evaluation at a time: each spawns 8 heavy PETSc workers, so two
#    concurrent runs abort with MPI_Abort. `--samples-dir` / `--out-dir` override
#    the config; `--no-filter` skips the filter.
micromamba run -n metamaterials python -m src.evaluate

# 5. Export top-N validity-passing candidates to STL (outputs/export/).
#    The surface is padded with void so it closes, giving watertight STLs as
#    ROADMAP 4.7 requires. --tiles N replicates the cell N times per axis
#    before surfacing; N=2 writes a printable 2x2x2 block instead of one cell.
micromamba run -n metamaterials python -m src.export.export_stl \
    --samples-dir outputs/evaluate --out-dir outputs/export --top-n 5

# Tests (136: FEA benchmarks, dataset loader, geometry, models, losses, export)
# and lint (rule set in ruff.toml).
micromamba run -n metamaterials python -m pytest tests
micromamba run -n metamaterials ruff check src tests
```
