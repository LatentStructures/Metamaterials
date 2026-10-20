# Run the baseline

End-to-end walkthrough for the physics-informed conditional diffusion baseline:
generate the dataset, train it, sample held-out targets, score them with real FEA,
and export STL. Written for anyone on the team working from a clean checkout.

```bash
cd /home/nidszxh/Projects/Metamaterials
export MAMBA_ROOT_PREFIX="$PWD/.mamba"   # env is repo-local; omit this and micromamba looks in ~/.local/share
```

Every command is run from the repo root. All hyperparameters live in `configs/`;
none are hardcoded in the scripts.

---

## 0. One-time setup

```bash
micromamba env create -f environment.yml     # Python 3.11 + fenics-dolfinx 0.11
micromamba run -n metamaterials pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu130      # adjust cu130 to YOUR driver's CUDA major
```

`torch` is intentionally absent from `environment.yml`: CUDA wheels are
machine-specific. See `README.md` for the index URL matching your driver.

Confirm the GPU is actually visible before training:

```bash
micromamba run -n metamaterials python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Sanity check the environment (FEA install is the usual blocker):

```bash
micromamba run -n metamaterials python tests/smoke_test.py   # solid-cube homogenization, prints C1111
micromamba run -n metamaterials python -m pytest tests -q    # 136 tests (needs the full env — 14 run
#   real FEA benchmarks via src.fea; smoke_test.py contributes 0 of them and
#   only runs as a script, so execute it directly for the env DoD)
```

---

## 1. Dataset

Resumable: progress is checkpointed to `data/checkpoints/ckpt_<config-stem>_s<seed>.parquet`,
one row per completed cell. Re-running after an interruption continues where it stopped.

```bash
micromamba run -n metamaterials python -m src.data.generate_dataset configs/dataset.yaml
```

Already complete in this repo (2,400 cells, 800/family), so you can skip to step 2.

**Do not use `--mock-fea`** except to test plumbing — it produces unphysical cells
and any checkpoint it writes looks valid.

---

## 2. Train

```bash
micromamba run -n metamaterials python -m src.train \
    --train-config configs/train_baseline.yaml \
    --model-config configs/model_baseline.yaml \
    --run-dir runs/baseline
```

Writes `runs/baseline/{ckpt.pt,config.json,events.out.tfevents.*}`. The resolved
config is snapshotted to `config.json` so a run is reproducible even after
`configs/` changes.

> **Already-trained checkpoints in this repo** (skip straight to sampling):
> - `runs/baseline_clipfix_30k/ckpt.pt` — the physics-on baseline (λ₁=0.05), 30k iters.
>   This is the checkpoint behind every headline number. It predates `config.json`
>   snapshotting, so it carries no `config.json` — use the configs in `configs/`.
> - `runs/baseline_physics_off/ckpt.pt` — the λ₁=0 ablation, 30k iters (has `config.json`).
>
> Substitute either path for `runs/baseline/ckpt.pt` in steps 3–5 below.

Watch it with `micromamba run -n metamaterials python -m tensorboard --logdir runs`.

`configs/train_baseline.yaml` carries `optimizer.grad_clip: 1.0`. **Do not remove
it:** the Option-B physics gradient produces unbounded stochastic bursts
(pre-clip global norms ~1,577 early, up to 4,229 over iters 20k–30k) that
dominate post-warmup training
and diverged the first full run. The clip bounds them; λ stays 0.05.

### Flags

| flag | purpose |
|---|---|
| `--run-dir DIR` | where to write the run (default `runs/<timestamp>`) |
| `--max-iterations N` | stop after N steps; use for a smoke test |
| `--overfit N` | train on the first N samples only (sets `max_iters = N*50`) as an architecture test |
| `--resume-from ckpt.pt` | **explicit** resume; never automatic |
| `--physics-ablation` | λ₁ = 0 control run, tag `abl_physics` |

Quick check that the loop works before committing hours:

```bash
micromamba run -n metamaterials python -m src.train --max-iterations 20 --run-dir /tmp/smoke
```

A full baseline run is ~2.5 h for 30k iterations on an 8 GB GPU
(`batch_size: 32` fits; lower it on smaller cards).

### Never resume a diverged run

The first full training attempt diverged because its config lacked the gradient
clip. That run was preserved while the fix was being validated and has since been
**deleted as dead evidence**. The lesson stands: a checkpoint from a run whose
config lacked `optimizer.grad_clip: 1.0` restarts from an unstable state, so never
`--resume-from` one. Reproduce the stable baseline with the config in
`configs/train_baseline.yaml` instead.

---

## 3. Sample

```bash
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/baseline/ckpt.pt \
    --targets data/targets_test_50.csv \
    --out-dir outputs/samples
```

`--targets` must be a CSV with columns **`E, relative_density, nu`** in that order.
That ordering is locked project-wide (manifest, HDF5 `/conditioning_vector`, model
input); do not reorder or add `G`.

| flag | purpose |
|---|---|
| `--steps N` | DDIM steps, locked 50–100 (default from config) |
| `--n-per-target N` | draws per target; one seed per target, each draw its own noise row — all distinct |
| `--seed N` | override the sampling seed |
| `--weights {ema,live}` | `live` (default) or the EMA shadow |
| `--export-stl` | also write `.stl` surfaces next to the samples (quick look only — it does **not** run the watertightness check or ranking; use step 5 for the ROADMAP §4.7 deliverable) |

**Use the default `--weights live` for acceptance numbers.** The EMA arm scores far
worse (14% validity vs 92%) and has produced FEA solver failures — this is a
**documented decay/schedule artefact**, not an open bug: at `decay = 0.9999` the
shadow averages a 10k-step window centred ~step 25k of the 30k run, i.e. *before* the
cosine LR anneals to zero, so it samples a less-converged model and emits fragmented
grids. Root cause and the interpolation dose–response are in `PROJECT.md` §8. The
`--weights ema` path is retained for the documented A/B. Single-draw numbers are
noisy — for a rate with an error bar use several draws:

```bash
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/baseline/ckpt.pt --targets data/targets_test_50.csv \
    --out-dir outputs/samples_k5 --weights live --n-per-target 5 --seed 0
```

---

## 4. Evaluate (real FEA re-homogenization)

```bash
micromamba run -n metamaterials python -m src.evaluate \
    --config configs/evaluate.yaml \
    --samples-dir outputs/samples \
    --out-dir outputs/evaluate
```

Re-solves **every** sample with FEA (no surrogate) on a spawned process pool and
applies the synthesizability filter. Writes `report.json`, `per_sample.csv`,
`parity.png`, `relerr.png` to `--out-dir`.

Runtime scales with samples. Measured on 8 workers: 50 cells ≈ 5.5 min,
540 cells ≈ 65 min. Budget accordingly and prefer `tee` over `tail` (below).

`--samples-dir` / `--out-dir` override the config; `--no-filter` skips the filter.

Validity here uses **periodic** connectivity: components are merged across the
wrapped faces, because a unit cell that joins through its boundary is one
structure. Counting in the unwrapped box understates validity badly (it read
50.8% vs the correct 87.7% on the dataset).

**Set `TMPDIR`** for long runs: `mkdir -p runs/_pipeline/tmp` then
`export TMPDIR=$PWD/runs/_pipeline/tmp`. (`runs/_pipeline/` currently holds only
the two baseline training logs; the `tmp/` subdir is scratch and is created on
demand — it is not committed or kept.)

### What to keep vs. delete under `runs/` and `outputs/`

Both trees are gitignored, so this is purely about keeping the repo's
signal-to-noise high — a teammate or reviewer should be able to tell the real
results from the scratch. The rule used throughout: **keep an artifact if some
doc cites it or it is the sole record of a result; delete it if its finding is
already written up elsewhere and nothing references it.**

**Keep — the baseline's actual results:**

```
runs/baseline_clipfix_30k/            physics-on baseline (λ=0.05) ckpt + tfevents
runs/baseline_physics_off/            λ₁=0 ablation ckpt + config.json + tfevents
runs/_pipeline/logs/6_train_clipfix_30k.log      baseline training record
runs/_pipeline/logs/7_train_abl_physics_off.log  ablation training record

outputs/acceptance_30k/    on-manifold 92.0% (evaluate_live/), EMA A/B, watertight exports
outputs/ablation_30k/      λ₁=0: 94.0% on-manifold, 98.1% off-manifold
outputs/synthetic_30k/     off-manifold 97.2% (evaluate/), k5 error bar
outputs/evaluate/          ⚠️ STALE 0.0% — kept ONLY as documented bug evidence, never cite
```

**Delete on sight — scratch, no evidence value:**

| pattern | why |
|---|---|
| `runs/diag_*` | A/B/C + overfit diagnostic ladders; findings are in `PROJECT.md` |
| `runs/diverged_*_DO_NOT_RESUME/` | the pre-clip diverged run; superseded by the clipfix run |
| duplicate `evaluate_fixed`/`samples_fixed` dirs | same run re-exported; reports differ only in the 5th decimal |
| pre-void-padding `export/`, `export_live/` | 0/5 watertight; superseded by `export_final`/`export_tiled` |
| `outputs/diag/` | toy architecture probe, uncited |
| `runs/_pipeline/` scratch | eval-progress PNGs, one-off probe scripts/shells, non-canonical logs |
| `runs/_pipeline/tmp/`, `train.pid` | runtime debris (torch inductor cache, pytest temp dirs, stale PID) |

If you delete a run that a doc names, add a one-line "deleted as superseded
scratch" note in this file so the record stays
honest rather than silently broken.

### The off-manifold control (reproducing the headline numbers)

All 50 targets in `data/targets_test_50.csv` are exact dataset cells, so the
discriminating control samples from targets built *between* dataset points:

```bash
# 1. Rebuild the 108 off-manifold targets (deterministic, seed 0):
micromamba run -n metamaterials python -m src.data.build_synthetic_targets
#    writes data/targets_synthetic.csv (cols E, relative_density, nu, source_family)

# 2. Sample them with live weights (as in step 3), then evaluate (step 4):
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/baseline/ckpt.pt --targets data/targets_synthetic.csv \
    --out-dir outputs/synthetic_30k/samples --weights live --seed 0
```

This is how `outputs/synthetic_30k/` was produced: **97.2%** validity (105/108)
single-draw, and **92.2% [90.0–94.5%]** over 5 draws
(`--n-per-target 5 --seed 0` → `samples_k5/`, 540 FEA evaluations ≈ 65 min).

> **Note (2026-10-20):** the target set was rebuilt to fix a reachability bug —
> the builder previously drew `nu` uniformly across each family's band, independent
> of `E` and `rho`, which produced targets no material can satisfy and a spurious
> `nu` R² of 0.31. The single-draw 97.2% above is on the corrected set; the 5-draw
> 92.2% predates the fix (its validity is unaffected in direction, its `E`/`nu`
> errors are inflated). See `PROJECT.md` §8 "Off-manifold target-set bug".

### If it aborts

A large batch can die with `MPI_Abort(59)` and **zero output**. The cause we
actually hit was **running more than one `evaluate` at a time**: each spawns
`n_workers` (8) heavy PETSc processes, so two concurrent runs means 16+ FEA
workers fighting for memory and the solver aborts. Run one evaluation at a time
and check with `pgrep -af src.evaluate` before starting another.

Confirm the logic is fine by evaluating a small subset, then re-run. For long
runs, don't pipe through `tail` — it swallows the traceback:

```bash
micromamba run -n metamaterials python -m src.evaluate \
    --samples-dir outputs/samples --out-dir outputs/evaluate 2>&1 | tee eval.log
```

---

## 5. Export STL

```bash
micromamba run -n metamaterials python -m src.export.export_stl \
    --samples-dir outputs/evaluate --out-dir outputs/export --top-n 5
```

Ranks validity-passing candidates by reconstruction error and writes
`export_manifest.csv`, `export_report.json`, and one `.stl` per candidate.

A periodic cell is not closed on its own, so the surface is **padded with void**
before marching cubes; that closes it and keeps the winding outward, giving
watertight files (ROADMAP §4.7). `--tiles N` replicates the cell N times per axis:
`--tiles 2` writes a printable 2×2×2 block instead of a single cell.

If no FEA report is available, it falls back to `samples_manifest.csv`. When a
report sits beside multiple sample dirs, `samples_live` is preferred — exporting
the EMA grids by mistake is a silent failure mode that produces plausible-looking
but wrong candidates.

Inspect the result in a mesh viewer (ROADMAP §4.6 step 5 / §4.7); the manifest
reports `is_watertight` per file so you can see whether you're looking at a valid
solid.

---

## 6. Ablation (control run)

```bash
micromamba run -n metamaterials python -m src.train \
    --train-config configs/train_baseline.yaml \
    --model-config configs/model_baseline.yaml \
    --run-dir runs/baseline_physics_off \
    --physics-ablation
```

Identical to the baseline except λ₁ = 0, i.e. diffusion loss only. `--physics-ablation`
sets the tag so the two runs stay distinguishable in the logs. Compare with the
baseline at the same iteration count, then sample and evaluate both the same way.

When you report the result: ROADMAP §7.2 only requires the ablation be **documented**,
so the baseline obligation is met either way. Read the verdict with the **scoping**
framing in `PROJECT.md` §8 — stronger claims ("Option B is
broken", "λ is mis-tuned", "the premise is wrong") are research-track hypotheses that
need a λ sweep or a second seed, not baseline conclusions.

---

## Layout

| path | what |
|---|---|
| `src/data/generate_dataset.py` | dataset generation (resumable) |
| `src/geometry/lattice_families.py` | the three topology families |
| `src/train.py` / `sample.py` / `evaluate.py` | the pipeline |
| `src/validation/` | periodic connectivity + synthesizability filter |
| `src/export/export_stl.py` | STL export |
| `configs/` | all hyperparameters |

## If something looks wrong

- **Sampling seems to return duplicates** — `--n-per-target N` gives N distinct
  draws (one seed per target; each draw samples its own noise row); if they look
  identical, you are on a stale tree.
- **Evaluation finds no samples** — `--samples-dir` must contain either
  `per_sample.csv` or `samples_manifest.csv` alongside the `.npy` grids.
- **Validity looks implausibly low** — check connectivity is being counted
  periodically (see step 4); the unwrapped count is far lower.
- **Training loss climbs instead of falling** — confirm `optimizer.grad_clip` is
  present in your config and in the run's `config.json`.
