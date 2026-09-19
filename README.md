# Metamaterials
Physics-informed conditional diffusion for inverse design of synthesizable mechanical metamaterials (voxel + 3D U-Net + FEniCSx homogenization).

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
micromamba run -n metamaterials python -m src.data.generate_dataset configs/dataset.yaml
#    --mock-fea is pipeline testing ONLY (unphysical placeholder stiffness).

# 2. Train the conditional DDPM (logs to W&B or TensorBoard).
micromamba run -n metamaterials python -m src.train
#    Flags: --overfit N (10-sample architecture test), --max-iterations N,
#           --resume-from <ckpt.pt>, --run-dir <dir>.

# 3. Sample from held-out targets (CSV columns: E, relative_density, nu).
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/<run>/ckpt.pt --targets targets.csv --export-stl

# 4. Re-homogenize every sample with real FEA, apply the synthesizability
#    filter, and report reconstruction error (outputs/evaluate/).
micromamba run -n metamaterials python -m src.evaluate

# 5. Export top-N validity-passing candidates to STL (outputs/export/).
micromamba run -n metamaterials python -m src.export.export_stl \
    --samples-dir outputs/evaluate --out-dir outputs/export --top-n 5

# Tests (FEA benchmarks, dataset loader, geometry, models, losses, export).
micromamba run -n metamaterials python -m pytest tests
```