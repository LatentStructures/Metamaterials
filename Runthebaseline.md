### Run the baseline

From the repository root:

cd /home/nidszxh/Projects/Metamaterials
export MAMBA_ROOT_PREFIX="$PWD/.mamba"

# Generate the dataset with real FEA; this is resumable.
micromamba run -n metamaterials python -m src.data.generate_dataset configs/dataset.yaml

# Train the Option B baseline.
micromamba run -n metamaterials python -m src.train \
  --train-config configs/train_baseline.yaml \
  --model-config configs/model_baseline.yaml \
  --run-dir runs/baseline_physics_on

# Train the zero-physics ablation for comparison.
micromamba run -n metamaterials python -m src.train \
  --train-config configs/train_baseline.yaml \
  --model-config configs/model_baseline.yaml \
  --run-dir runs/baseline_physics_off \
  --physics-ablation

# Sample the 50 held-out targets and export the voxel surfaces.
micromamba run -n metamaterials python -m src.sample \
  --checkpoint runs/baseline_physics_on/ckpt.pt \
  --targets data/targets_test_50.csv \
  --out-dir outputs/samples \
  --export-stl

# Re-homogenize and evaluate the generated samples with FEA.
micromamba run -n metamaterials python -m src.evaluate \
  --config configs/evaluate.yaml \
  --samples-dir outputs/samples \
  --out-dir outputs/evaluate

The generator will stop if the completed data cannot meet the configured minimum size or supply 50 distinct held-out conditions. If that happens, keep the new checkpoint and error output; we can use them to adjust the design parameterization before you spend time on training.

The geometry changes are in lattice_families.py (src/geometry/lattice_families.py), the run settings are in dataset.yaml (configs/dataset.yaml), and checkpointing and dataset output are handled in generate_dataset.py (src/data/generate_dataset.py).
