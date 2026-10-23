# outputs/ — results at a glance

Entry point for reviewers. Each directory has a cite-map; read this first to know what to cite.

**Current evidence (2026-10-22):** Option-B proxy corrected to periodic boundaries (matching the hard filter). Two matched pipelines: **V1** (original `data/`, 87.7% valid) and **V1f** (gate-only corrected `data/v1f/`, 100% valid).

## Headline results — corrected proxy (cite these)

| # | result | set | number | details |
|---|---|---|---|---|
| A | **V1 physics-on** | 50 held-out / 108 off-manifold | **88.0%** / **95.4%** (0 failures) | [`final_acceptance/summary.md`](final_acceptance/summary.md) |
| B | **V1 λ₁=0** | 50 held-out / 108 off-manifold | **94.0%** / **98.1%** (0 failures) | [`final_acceptance/summary.md`](final_acceptance/summary.md) |
| C | **V1f physics-on** | 50 held-out / 108 off-manifold | **90.0%** / **95.4%** (0 failures) | [`final_acceptance_v1f/summary.md`](final_acceptance_v1f/summary.md) |
| D | **V1f λ₁=0** | 50 held-out / 108 off-manifold | **94.0%** / **96.3%** (0 failures) | [`final_acceptance_v1f/summary.md`](final_acceptance_v1f/summary.md) |

**Corrected-target multi-draw (V1f, 108 × 5 draws):** physics-on **95.6%** (516/540, [93.5%, 97.6%] target-level CI) / physics-off **96.5%** (521/540, [94.0%, 98.9%]). See [`final_acceptance_v1f/errorbar_k5/summary.md`](final_acceptance_v1f/errorbar_k5/summary.md).

For interpretation, tradeoffs, and caveats, see **[`PROJECT.md` §8](../PROJECT.md#8-current-status--whats-left)**.

## Historical results — legacy zero-padded proxy (2026-10-20/21)

| # | result | set | number | directory |
|---|---|---|---|---|
| 1 | **On-manifold acceptance** | 50 held-out targets | **92.0%** valid (46/50), 0 FEA failures; E 3.6% / ρ 1.5% / ν 1.0%; R² 0.995/0.998/0.958 | `acceptance_30k/evaluate_live/` |
| 2 | **Off-manifold control** | 108 synthetic targets | **97.2%** valid (105/108), 0 FEA failures | `synthetic_30k/evaluate/` |
| 3 | **λ₁=0 ablation (on-manifold)** | 50 held-out targets | **94.0%** valid (47/50), 0 FEA failures | `ablation_30k/evaluate/` |
| 4 | **λ₁=0 ablation (off-manifold)** | 108 synthetic targets | **98.1%** valid (106/108), 0 FEA failures | `ablation_30k/evaluate_synthetic/` |

Supporting: the legacy 5-draw result is 92.2%, sample-level 95% CI [90.0%,
94.5%] over 540 samples (`synthetic_30k/evaluate_k5/`). Corrected V1f also has
a 5-draw result on the fixed 108-target set (see above); its target-level
intervals are the more relevant uncertainty summary across targets.

## Look at these files first

The numbers above are backed by rendered images and printable geometry. These are
the ones worth opening:

| file | what you're looking at |
|---|---|
| `acceptance_30k/samples_live/sample_grid.png` | generated metamaterial grids, physics-ON baseline |
| `acceptance_30k/evaluate_live/parity.png`, `relerr.png` | effective-property reconstruction: predicted vs FEA |
| `acceptance_30k/export_final/montage.png` | the top-5 designs, ready to print |
| `acceptance_30k/evidence/reference_vs_generated.png` | **target's reference cell vs the generated grid**, all 50 targets |
| `acceptance_30k/evidence/selected_pair.png` | one candidate end-to-end: requested vs FEA-verified + validity + STL status |
| `acceptance_30k/evidence/selected_candidate_3d.png` | 3D render of the selected candidate |
| `acceptance_30k/evidence/export_montage.png` | reproducible 3D montage of the exported top-5 |
| `acceptance_30k/evidence/evidence_manifest.json` | machine-readable index: every displayed claim → its source artifact |
| `acceptance_30k/export_final/*.stl` (5 files) | **watertight 3D meshes** — open these in a viewer to see the structures |
| `final_acceptance/periodic_physics_on/sample_grid.png` | generated grids, corrected-proxy physics-on |
| `final_acceptance_v1f/v1f_physics_on/sample_grid.png` | generated grids, corrected-dataset physics-on |
| `final_acceptance*/evaluate_*/parity.png`, `relerr.png` | reconstruction parity for each corrected arm |
| `ablation_30k/samples_live/paired_vs_physics_on.png` | physics-off vs physics-on, side by side (legacy) |
| `ablation_30k/thin_wall_off_vs_on.png` | wall-thickness analysis (the legacy ablation mechanism) |
| `synthetic_30k/evaluate/parity.png`, `relerr.png` | off-manifold reconstruction (no cell to copy) |

## What each top-level directory is

| dir | contents | cite? |
|---|---|---|
| `final_acceptance/` | **V1 corrected-proxy pipeline**: both arms, 4 evaluations, satisfiable-subset + family/density reports, summary | ✅ current evidence (results A, B) |
| `final_acceptance_v1f/` | **V1f corrected-dataset pipeline**: matched evaluations plus corrected-target 5-draw follow-up | ✅ current evidence (results C, D) |
| `ema_validation/`, `ema_validation_v1f/` | sampler-weight selection on the 90-target validation set (V1 → ema-0.9999 both arms; V1f → ema-0.9995 on, live off) | ✅ when discussing EMA selection |
| `audit/` | run audits (`verify_*.json`: 30k iters, grad_clip, EMA shadows, non-finite), dataset integrity, validity cache, dataset-derivation report | ✅ provenance |
| `acceptance_30k/` | **legacy** physics-ON baseline: on-manifold acceptance, EMA A/B, STL exports, evidence pack | ✅ results 1 (historical) |
| `ablation_30k/` | **legacy** λ₁=0 ablation: on- and off-manifold, STL export | ✅ results 3, 4 (historical) |
| `synthetic_30k/` | **legacy** physics-ON off-manifold control + 5-draw error bar | ✅ result 2 (historical) |
| `evaluate/` | ⚠️ **STALE 0.0%** report kept only as documented evidence of the pre-fix connectivity bug — **never cite** | ❌ see `evaluate/STALE_DO_NOT_CITE.md` |

## Do not cite

- `acceptance_30k/evaluate/` — 54%; same grids as `evaluate_live/`, but judged
  **before** the periodic-connectivity/wall filter fix. Superseded by result 1.
- `acceptance_30k/evaluate_ema/` — 14%; the legacy single-decay EMA arm, a
  documented decay/schedule artefact (see `PROJECT.md` §8). Only citable when
  discussing the EMA A/B. The multi-decay selection in `ema_validation*/` is
  the current implementation.
- `outputs/evaluate/` — 0.0%; pre-fix. Kept deliberately as bug evidence.

## How to read the results responsibly

- The **50 held-out targets are exact dataset cells** (nearest-cell distance 0.0),
  so an on-manifold result alone could be consistent with copying. The 108-target
  off-manifold control uses targets built *between* dataset cells and is what
  rules out memorization — it passes in every corrected arm (95.4–98.1%).
- **Every corrected run is single-seed** (seed 0). The physics-off advantage is
  consistent across 2 pipelines × 2 target sets, but it is not seed-replicated.
  The V1f corrected-target multi-draw follow-up uses five draws per target, but
  does not measure between-seed variability.
- **Off-manifold E R² still misses 0.85 in every corrected arm** (0.719–0.792
  physics-on, 0.825–0.849 physics-off). Disclosed, not hidden: the E *error*
  gate passes everywhere (≤ 16.2%) and the gap is target-spread. See
  `PROJECT.md` §8.
- **V1f requires `--train-config configs/train_baseline_v1f.yaml` at sampling
  time** — different conditioning statistics from V1; the default config
  silently samples with the wrong ones.

Detail and citations: `final_acceptance/README.md`,
`final_acceptance_v1f/README.md`, `acceptance_30k/README.md`,
`ablation_30k/README.md`, `synthetic_30k/README.md`, plus `PROJECT.md`,
`FINDINGS.md`, and `data/*/data_card.md`.
