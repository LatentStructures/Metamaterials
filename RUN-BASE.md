# Run the baseline

End-to-end walkthrough for the physics-informed conditional diffusion baseline: generate the
dataset, train it, sample held-out targets, score them with real FEA, and export STL. Written for
anyone on the team working from a clean checkout.

```bash
# Run these commands from the repository root; no machine-specific checkout path is needed.
export MAMBA_ROOT_PREFIX="$PWD/.mamba"   # env is repo-local; omit this and micromamba looks in ~/.local/share
```

Every command is run from the repo root. All hyperparameters live in `configs/`; none are hardcoded
in the scripts.

---

## 0. One-time setup

```bash
micromamba env create -f environment.yml     # Python 3.11 + fenics-dolfinx 0.11
micromamba run -n metamaterials pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu130      # adjust cu130 to YOUR driver's CUDA major
```

`torch` is intentionally absent from `environment.yml`: CUDA wheels are machine-specific. See
`README.md` for the index URL matching your driver.

Confirm the GPU is actually visible before training:

```bash
micromamba run -n metamaterials python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Sanity check the environment (FEA install is the usual blocker):

```bash
micromamba run -n metamaterials python tests/smoke_test.py   # solid-cube homogenization, prints C1111
micromamba run -n metamaterials python -m pytest tests -q    # 170 tests (needs the full env — real FEA regressions run
#   real FEA benchmarks via src.fea; smoke_test.py contributes 0 of them and
#   only runs as a script, so execute it directly for the env DoD)
micromamba run -n metamaterials ruff check src tests scripts
```

---

## 1. Dataset

Resumable: progress is checkpointed to `data/checkpoints/ckpt_<config-stem>_s<seed>.parquet`, one
row per completed cell. Re-running after an interruption continues where it stopped.

```bash
micromamba run -n metamaterials python -m src.data.generate_dataset configs/dataset.yaml
```

Already complete in this repo (2,400 cells, 800/family), so you can skip to step 2.

**Do not use `--mock-fea`** except to test plumbing — it produces unphysical cells and any
checkpoint it writes looks valid.

---

## 2. Train

```bash
micromamba run -n metamaterials python -m src.train \
    --train-config configs/train_baseline.yaml \
    --model-config configs/model_baseline.yaml \
    --run-dir runs/baseline
```

Writes `runs/baseline/{ckpt.pt,config.json,events.out.tfevents.*}`. The resolved config is
snapshotted to `config.json` so a run is reproducible even after `configs/` changes.

> **Already-trained checkpoints in this repo** (skip straight to sampling):
> - `runs/baseline_clipfix_30k/ckpt.pt` — the legacy physics-on baseline (λ₁=0.05), 30k iters.
>   Historically the headline checkpoint. It predates `config.json` snapshotting, so its
>   `config.json` was reconstructed on 2026-10-21 from the config text embedded in the run's own
>   tfevents and carries an `_reconstructed` provenance key — `configs/` matches it field-for-field.
> - `runs/baseline_physics_off/ckpt.pt` — the legacy λ₁=0 ablation, 30k iters (has `config.json`).
> - **V1 (corrected proxy, dataset `data/`):** `runs/final_periodic_physics_on_gpu/ckpt.pt`
>   (physics-on) and `runs/final_periodic_physics_off/ckpt.pt` (λ₁=0) — 30k iters each, audited in
>   `outputs/audit/verify_periodic_physics_{on,off}.json`.
> - **V1f (corrected dataset `data/v1f/`):** `runs/final_v1f_physics_on/ckpt.pt` and
>   `runs/final_v1f_physics_off/ckpt.pt` — 30k iters each, audited in
>   `outputs/audit/verify_v1f_physics_{on,off}.json`.
>
> **V1/V1f are the current results; the clipfix pair is the historical baseline.** Results:
> `outputs/final_acceptance/` (V1), `outputs/final_acceptance_v1f/` (V1f), write-up in `PROJECT.md`
> §8 "Corrected-proxy matched results". Substitute any of these paths for `runs/baseline/ckpt.pt` in
> steps 3–5 below.
>
> **For V1f checkpoints, sampling and evaluation must pass `--train-config
> configs/train_baseline_v1f.yaml`** (see step 3): the two datasets have different conditioning
> normalization statistics, and the default config silently samples V1f with the wrong ones.

> **Historical result scope (2026-10-21):** the clipfix/ablation checkpoints were trained before the
> Option-B graph boundary correction. The old soft proxy used zero padding; the hard validity filter
> is periodic. The corrected-proxy comparison that supersedes them is above; do not treat the
> historical physics-on/off comparison as validation of the corrected proxy.

Watch it with `micromamba run -n metamaterials python -m tensorboard --logdir runs`.

`configs/train_baseline.yaml` carries `optimizer.grad_clip: 1.0`. **Do not remove it:** the Option-B
physics gradient produces unbounded stochastic bursts that dominate post-warmup training and
diverged the first full run (legacy zero-padded proxy: pre-clip global norms first pass 1,000 at
iter 9,250 and peak at 4,229 at iter 27,600; the corrected periodic proxy is ~100× milder — max 37.2
on the V1 physics-on arm vs 5.91 for its λ₁=0 control — but still bursts above every diffusion
gradient in the run). The clip bounds them in both cases; λ stays 0.05. Run-audit evidence:
`outputs/audit/verify_*.json`.

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
micromamba run -n metamaterials python -m src.train --max-iterations 20 --run-dir runs/smoke
```

A full baseline run is ~2.5 h for 30k iterations on an 8 GB GPU (`batch_size: 32` fits; lower it on
smaller cards).

### Never resume a diverged run

The first full training attempt diverged because its config lacked the gradient clip. The lesson
stands: a checkpoint from a run whose config lacked `optimizer.grad_clip: 1.0` restarts from an
unstable state, so never `--resume-from` one. Reproduce the stable baseline with the config in
`configs/train_baseline.yaml` instead.

---

## 3. Sample

```bash
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/baseline/ckpt.pt \
    --targets data/targets_test_50.csv \
    --out-dir outputs/samples
```

`--targets` must be a CSV with columns **`E, relative_density, nu`** in that order. That ordering is
locked project-wide (manifest, HDF5 `/conditioning_vector`, model input); do not reorder or add `G`.

| flag | purpose |
|---|---|
| `--steps N` | DDIM steps, locked 50–100 (default from config) |
| `--n-per-target N` | draws per target; one seed per target, each draw its own noise row — all distinct |
| `--seed N` | override the sampling seed |
| `--train-config PATH` | conditioning-statistics source; **required for V1f checkpoints** (see below) |
| `--weights {ema,live}` | `live` (default) or the EMA shadow |
| `--export-stl` | also write `.stl` surfaces next to the samples (quick look only — it does **not** run the watertightness check or ranking; use step 5 for the ROADMAP §4.7 deliverable) |

> **Dataset-specific normalization.** A V1f checkpoint (`runs/final_v1f_*`) was trained on
> `data/v1f/` whose conditioning mean/std differ from `data/`'s (`[0.1809, 0.4055, 0.2668]` vs
> `[0.1457, 0.3480, 0.2735]`). Pass `--train-config configs/train_baseline_v1f.yaml` to `src.sample`
> for those checkpoints (the normalization lives in the sampler); the default config normalizes with
> the wrong statistics and silently produces a materially worse model (caught once during the run: E
> R² 0.803 instead of 0.989). `src.evaluate` re-solves the stored grids with FEA and needs no train
> config. V1/legacy checkpoints use the default config.

> **Which weights to use.** Use the default `--weights live` to reproduce the historical baseline
> numbers. For the corrected-proxy runs, the sampler weights were selected on the 90-target
> validation set (rule: highest validation validity, ties → `live`, then the smallest decay) and are
> recorded per run: V1 → `ema-0.9999` both arms; V1f → `ema-0.9995` (physics-on) and `live`
> (physics-off). To reproduce a corrected headline number, pass that run's selected weight (e.g.
> `--weights ema-0.9999`); selections live in `outputs/ema_validation*/`*_weights.json`. The legacy
> `--weights ema` failure (14% vs 92% validity) is a documented decay/schedule artefact of the old
> single-decay shadow — root cause and dose–response in `PROJECT.md` §8.2. Single-draw numbers are
> noisy — for a rate with an error bar use several draws:

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

Re-solves **every** sample with FEA (no surrogate) on a spawned process pool and applies the
synthesizability filter. Writes `report.json`, `per_sample.csv`, `parity.png`, `relerr.png` to
`--out-dir`.

Runtime scales with samples. Measured on 8 workers: 50 cells ≈ 5.5 min, 540 cells ≈ 65 min. Budget
accordingly and prefer `tee` over `tail` (below).

`--samples-dir` / `--out-dir` override the config; `--no-filter` skips the filter.

Validity here uses **periodic** connectivity: components are merged across the wrapped faces,
because a unit cell that joins through its boundary is one structure. Counting in the unwrapped box
understates validity badly (it read 50.8% vs the correct 87.7% on the dataset).

**Set `TMPDIR`** for long runs: `mkdir -p runs/_pipeline/tmp` then `export
TMPDIR=$PWD/runs/_pipeline/tmp`. (`runs/_pipeline/` currently holds only the two baseline training
logs; the `tmp/` subdir is scratch and is created on demand — it is not committed or kept.)

### What to keep under `runs/` and `outputs/`

Both trees are gitignored. **Keep an artifact if some doc cites it or it is the sole record of a
result.**

**Keep — the baseline's actual results:**

```
runs/final_periodic_physics_on_gpu/   V1 corrected-proxy physics-on ckpt + config + tfevents
runs/final_periodic_physics_off/      V1 λ₁=0 control ckpt + config + tfevents
runs/final_v1f_physics_on/            V1f corrected-dataset physics-on ckpt + config + tfevents
runs/final_v1f_physics_off/           V1f λ₁=0 control ckpt + config + tfevents
runs/baseline_clipfix_30k/            legacy physics-on baseline (λ=0.05) ckpt + tfevents
runs/baseline_physics_off/            legacy λ₁=0 ablation ckpt + config.json + tfevents

outputs/final_acceptance/             V1: 88.0%/95.4% on, 94.0%/98.1% off (summary.md, satisfiable_*, family_density_validity)
outputs/final_acceptance_v1f/         V1f: 90.0%/95.4% on, 94.0%/96.3% off (same report set)
outputs/final_acceptance_v1f/errorbar_k5/          corrected-target 540-draw error bar (95.6% on / 96.5% off)
outputs/final_acceptance_v1f/*_synthetic_k5/       the 2x540 grids + their FEA evals behind the error bar
outputs/ema_validation/               V1 sampler-weight selection (per-decay validation validity)
outputs/ema_validation_v1f/           V1f sampler-weight selection
outputs/audit/verify_{periodic,v1f}_physics_{on,off}.json   run audits (30k iters, grad_clip, EMA, non-finite)
outputs/acceptance_30k/               legacy on-manifold 92.0% (evaluate_live/), EMA A/B, watertight exports, evidence pack
outputs/ablation_30k/                 legacy λ₁=0: 94.0% on-manifold, 98.1% off-manifold
outputs/synthetic_30k/                legacy off-manifold 97.2% (evaluate/), k5 error bar
outputs/evaluate/                     ⚠️ STALE 0.0% — kept ONLY as documented bug evidence, never cite
```

### The off-manifold control (reproducing the headline numbers)

All 50 targets in `data/targets_test_50.csv` are exact dataset cells, so the discriminating control
samples from targets built *between* dataset points:

```bash
# 1. Rebuild the 108 off-manifold targets (deterministic, seed 0):
micromamba run -n metamaterials python -m src.data.build_synthetic_targets
#    writes data/targets_synthetic.csv (cols E, relative_density, nu, source_family)

# 2. Sample them with live weights (as in step 3), then evaluate (step 4):
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/baseline/ckpt.pt --targets data/targets_synthetic.csv \
    --out-dir outputs/synthetic_30k/samples --weights live --seed 0
```

This is how `outputs/synthetic_30k/` was produced: **97.2%** validity (105/108) single-draw, and
**92.2% [90.0–94.5%]** over 5 draws (`--n-per-target 5 --seed 0` → `samples_k5/`, 540 FEA
evaluations ≈ 65 min).

> **Note (2026-10-20):** the target set was rebuilt to fix a reachability bug — the builder
> previously drew `nu` uniformly across each family's band, independent of `E` and `rho`, which
> produced targets no material can satisfy and a spurious `nu` R² of 0.31. The single-draw 97.2%
> above is on the corrected set; the 5-draw 92.2% predates the fix (its validity is unaffected in
> direction, its `E`/`nu` errors are inflated). See `PROJECT.md` §8.2 "Legacy results".

### If it aborts

A large batch can die with `MPI_Abort(59)` and **zero output**. The cause we actually hit was
**running more than one `evaluate` at a time**: each spawns `n_workers` (8) heavy PETSc processes,
so two concurrent runs means 16+ FEA workers fighting for memory and the solver aborts. Run one
evaluation at a time and check with `pgrep -af src.evaluate` before starting another.

Confirm the logic is fine by evaluating a small subset, then re-run. For long runs, don't pipe
through `tail` — it swallows the traceback:

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

Ranks validity-passing candidates by reconstruction error and writes `export_manifest.csv`,
`export_report.json`, and one `.stl` per candidate.

A periodic cell is not closed on its own, so the surface is **padded with void** before marching
cubes; that closes it and keeps the winding outward, giving watertight files (ROADMAP §4.7).
`--tiles N` replicates the cell N times per axis: `--tiles 2` writes a printable 2×2×2 block instead
of a single cell.

If no FEA report is available, it falls back to `samples_manifest.csv`. When a report sits beside
multiple sample dirs, `samples_live` is preferred — exporting the EMA grids by mistake is a silent
failure mode that produces plausible-looking but wrong candidates.

Inspect the result in a mesh viewer (ROADMAP §4.6 step 5 / §4.7); the manifest reports
`is_watertight` per file so you can see whether you're looking at a valid solid.

---

## 6. Ablation (control run)

```bash
micromamba run -n metamaterials python -m src.train \
    --train-config configs/train_baseline.yaml \
    --model-config configs/model_baseline.yaml \
    --run-dir runs/baseline_physics_off \
    --physics-ablation
```

Identical to the baseline except λ₁ = 0, i.e. diffusion loss only. `--physics-ablation` sets the tag
so the two runs stay distinguishable in the logs. Compare with the baseline at the same iteration
count, then sample and evaluate both the same way.

When you report the result: ROADMAP §7.2 only requires the ablation be **documented**, so the
baseline obligation is met either way. Read the verdict with the **scoping** framing in `PROJECT.md`
§8 — stronger claims ("Option B is broken", "λ is mis-tuned", "the premise is wrong") are
research-track hypotheses that need a λ sweep or a second seed, not baseline conclusions. The
measured matched corrected-proxy verdict (λ₁=0 leads on validity and on E/ν error in 7 of 8 property
cells, single seed) is in `PROJECT.md` §8.1.

---

## 6b. Reproducing the corrected-proxy runs (V1 / V1f)

The two matched pipelines that produced the current headline numbers are orchestrated end to end
(train both arms → verify runs → EMA validation → sample → evaluate → satisfiable-subset +
family/density reports → summary):

```bash
# V1 — corrected periodic proxy on the original dataset:
bash scripts/run_matched_arms.sh

# V1f — same protocol on the gate-only corrected dataset
# (drops the 296 filter-invalid rows; data/v1f/data_card.md):
bash scripts/run_v1f_after_v1.sh          # passes --train-config configs/train_baseline_v1f.yaml
```

Both scripts are resumable phase pipelines: every phase writes a marker under
`runs/_pipeline*/markers/`, so re-running skips finished phases. Each arm is ~2–3 h on an 8 GB GPU
plus ~6 min sampling per weight choice and ~5.5 min of FEA per 50-sample evaluation. Key individual
phases, if you want to re-run one:

```bash
# train one arm (V1f example):
micromamba run -n metamaterials python -m src.train \
    --train-config configs/train_baseline_v1f.yaml \
    --model-config configs/model_baseline.yaml \
    --run-dir runs/final_v1f_physics_on            # add --physics-ablation for the λ₁=0 arm

# verify a run reached 30k with grad_clip, EMA shadows, no non-finite:
micromamba run -n metamaterials python scripts/verify_training_run.py \
    --run-dir runs/final_v1f_physics_on --out outputs/audit/verify_v1f_physics_on

# EMA/decoy-weight selection on the 90-target validation set:
micromamba run -n metamaterials python scripts/select_ema_weights.py --help

# sample + evaluate a V1f arm (note --train-config on the sample step):
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/final_v1f_physics_on/ckpt.pt \
    --targets data/targets_test_50.csv \
    --train-config configs/train_baseline_v1f.yaml \
    --weights ema-0.9995 --out-dir outputs/final_acceptance_v1f/v1f_physics_on

# full-set vs satisfiable-subset validity (the verdicts use the ground-truth
# data/manifest.parquet for BOTH pipelines so V1 and V1f stay comparable):
micromamba run -n metamaterials python scripts/acceptance_satisfiable_subset.py \
    --targets data/targets_test_50.csv \
    --eval outputs/final_acceptance_v1f/evaluate_v1f_physics_on \
    --manifest data/manifest.parquet \
    --validity-cache outputs/audit/validity_cache_v1.json \
    --out outputs/final_acceptance_v1f/satisfiable_v1f_physics_on

# corrected-target 540-draw error bar (5 draws x 108 synthetic targets, both
# arms; off arm: runs/final_v1f_physics_off + --weights live, same flags):
micromamba run -n metamaterials python -m src.sample \
    --checkpoint runs/final_v1f_physics_on/ckpt.pt \
    --targets data/targets_synthetic.csv \
    --train-config configs/train_baseline_v1f.yaml \
    --weights ema-0.9995 --n-per-target 5 --seed 0 \
    --out-dir outputs/final_acceptance_v1f/v1f_physics_on_synthetic_k5
micromamba run -n metamaterials python -m src.evaluate \
    --samples-dir outputs/final_acceptance_v1f/v1f_physics_on_synthetic_k5 \
    --out-dir outputs/final_acceptance_v1f/evaluate_v1f_physics_on_synthetic_k5
micromamba run -n metamaterials python scripts/errorbar_k5_summary.py \
    --eval-on  outputs/final_acceptance_v1f/evaluate_v1f_physics_on_synthetic_k5 \
    --eval-off outputs/final_acceptance_v1f/evaluate_v1f_physics_off_synthetic_k5 \
    --label-on v1f_physics_on --label-off v1f_physics_off \
    --out-dir outputs/final_acceptance_v1f/errorbar_k5
```

Derived-dataset recipe (no FEA re-solve — labels are copied bit-for-bit):

```bash
micromamba run -n metamaterials python scripts/derive_corrected_dataset.py --help
```

Results land in `outputs/final_acceptance/` (V1) and `outputs/final_acceptance_v1f/` (V1f); the
cite-maps are `outputs/final_acceptance/README.md` and `outputs/final_acceptance_v1f/README.md`.
Full tables and caveats: `PROJECT.md` §8.

---

## 7. Evidence pack

```bash
micromamba run -n metamaterials python -m src.evidence
```

Builds the "target → verified candidate" display pack in `outputs/acceptance_30k/evidence/` **from
recorded artifacts only** — it reuses the evaluate/export outputs, no FEA re-run:

| file | contents |
|---|---|
| `reference_vs_generated.png` | target's nearest dataset cell vs the generated grid, for all 50 acceptance targets |
| `selected_pair.png` | the selected candidate: requested vs achieved properties, validity verdict, reference pairing |
| `selected_candidate_3d.png` | marching-cubes 3D render of that candidate |
| `export_montage.png` | STL export montage (per-file watertightness in the manifest) |
| `evidence_manifest.json` | claim → source artifact map, validity / FEA-failure stats, per-target match distances |

Defaults point at the citable acceptance dirs; flags: `--samples-dir`, `--eval-dir`, `--dataset`
(reference-cell split), `--export-dir`, `--out-dir`, `--selected <file>` (default: top-ranked valid
candidate), `--max-match-dist`, `--cols N`. Re-run it after steps 4–5 if you regenerate samples.

---

## Layout

| path | what |
|---|---|
| `src/data/generate_dataset.py` | dataset generation (resumable) |
| `src/geometry/lattice_families.py` | the three topology families |
| `src/train.py` / `sample.py` / `evaluate.py` | the pipeline |
| `src/validation/` | periodic connectivity + synthesizability filter |
| `src/export/export_stl.py` | STL export |
| `src/evidence.py` | evidence pack (step 7) |
| `configs/` | all hyperparameters |

## If something looks wrong

- **Sampling seems to return duplicates** — `--n-per-target N` gives N distinct draws (one seed per
  target; each draw samples its own noise row); if they look identical, you are on a stale tree.
- **Evaluation finds no samples** — `--samples-dir` must contain either `per_sample.csv` or
  `samples_manifest.csv` alongside the `.npy` grids.
- **Validity looks implausibly low** — check connectivity is being counted periodically (see step
  4); the unwrapped count is far lower.
- **Training loss climbs instead of falling** — confirm `optimizer.grad_clip` is present in your
  config and in the run's `config.json`.
