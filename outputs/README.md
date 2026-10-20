# outputs/ — results at a glance

Entry point for teammates and reviewers. Every directory here was produced by the
baseline pipeline; the ones below are the **citable results**. Read this first,
then follow the per-directory cite-maps for detail.

The four headline numbers, all on the final (post-fix) pipeline:

| # | result | set | number | directory |
|---|---|---|---|---|
| 1 | **On-manifold acceptance** | 50 held-out targets | **92.0%** valid (46/50), 0 FEA failures; E 3.6% / ρ 1.5% / ν 1.0%; R² 0.995/0.998/0.958 | `acceptance_30k/evaluate_live/` |
| 2 | **Off-manifold control** | 108 synthetic targets | **97.2%** valid (105/108), 0 FEA failures | `synthetic_30k/evaluate/` |
| 3 | **λ₁=0 ablation (on-manifold)** | 50 held-out targets | **94.0%** valid (47/50), 0 FEA failures | `ablation_30k/evaluate/` |
| 4 | **λ₁=0 ablation (off-manifold)** | 108 synthetic targets | **98.1%** valid (106/108), 0 FEA failures | `ablation_30k/evaluate_synthetic/` |

Supporting: the 5-draw error bar is 92.2%, 95% CI [90.0%, 94.5%] over 540 samples
(`synthetic_30k/evaluate_k5/`).

## Look at these files first

The numbers above are backed by rendered images and printable geometry. These are
the ones worth opening:

| file | what you're looking at |
|---|---|
| `acceptance_30k/samples_live/sample_grid.png` | generated metamaterial grids, physics-ON baseline |
| `acceptance_30k/evaluate_live/parity.png`, `relerr.png` | effective-property reconstruction: predicted vs FEA |
| `acceptance_30k/export_final/montage.png` | the top-5 designs, ready to print |
| `acceptance_30k/export_final/*.stl` (5 files) | **watertight 3D meshes** — open these in a viewer to see the structures |
| `ablation_30k/samples_live/paired_vs_physics_on.png` | physics-off vs physics-on, side by side |
| `ablation_30k/thin_wall_off_vs_on.png` | wall-thickness analysis (the ablation mechanism) |
| `ablation_30k/export/montage.png` | the ablation's top-5 printable designs |
| `synthetic_30k/evaluate/parity.png`, `relerr.png` | off-manifold reconstruction (no cell to copy) |

## What each top-level directory is

| dir | contents | cite? |
|---|---|---|
| `acceptance_30k/` | physics-ON baseline: on-manifold acceptance, EMA A/B, STL exports | ✅ results 1 |
| `ablation_30k/` | λ₁=0 ablation: on- and off-manifold, STL export | ✅ results 3, 4 |
| `synthetic_30k/` | physics-ON off-manifold control + 5-draw error bar | ✅ result 2 |
| `evaluate/` | ⚠️ **STALE 0.0%** report kept only as documented evidence of the pre-fix connectivity bug — **never cite** | ❌ see `evaluate/STALE_DO_NOT_CITE.md` |
| `samples/` | early pre-fix sample draw (superseded scratch; kept, not citable) | ❌ |

## Do not cite

- `acceptance_30k/evaluate/` — 54%; same grids as `evaluate_live/`, but judged
  **before** the periodic-connectivity/wall filter fix. Superseded by result 1.
- `acceptance_30k/evaluate_ema/` — 14%; the EMA arm, a documented decay/schedule
  artefact (see `PROJECT.md` §8). Only citable when discussing the EMA A/B.
- `outputs/evaluate/` — 0.0%; pre-fix. Kept deliberately as bug evidence.
- `outputs/samples/` — pre-fix scratch draw.

## How to read the results responsibly

- The **50 held-out targets are exact dataset cells** (nearest-cell distance 0.0),
  so result 1 alone could be consistent with copying. Result 2 uses targets built
  *between* dataset cells (distance ≥ 0.11 in dataset-std units) and is what rules
  out memorization.
- Result 4 (ablation matched/beat physics-ON everywhere) is the headline finding.
  Read `PROJECT.md` §8 / `SESSION_EXPORT.md` §6.2 **before** quoting it: the
  required ablation was to *document* the ablation, so the baseline obligation is
  met; "Option B is broken" is a research-track hypothesis, not a baseline result.
- Off-manifold **E R² = 0.808 misses the 0.85 bar by 0.04** (result 2). This is
  disclosed, not hidden: the E *error* gate passes (13.8% ≤ 15%) and the gap is
  target-spread. See `PROJECT.md` §8.

Detail and citations: `acceptance_30k/README.md`, `ablation_30k/README.md`,
`synthetic_30k/README.md`, plus `PROJECT.md` and `SESSION_EXPORT.md` at the repo root.
