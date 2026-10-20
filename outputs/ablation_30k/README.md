# outputs/ablation_30k — which subdirectory to use

All numbers here are the **λ₁ = 0 ablation** (`runs/baseline_physics_off`),
sampled with `--weights live`, base seed 0. Compare against the physics-ON
counterparts in `outputs/acceptance_30k/` (50 held-out) and
`outputs/synthetic_30k/` (108 off-manifold).

| dir | what it is | cite? |
|---|---|---|
| `evaluate/` | **on-manifold ablation**: 94.0% (47/50) on the 50 held-out targets, 0 FEA failures, reads `samples_live/` | ✅ **YES** |
| `evaluate_synthetic/` | **off-manifold ablation**: 98.1% (106/108) on the 108 physical off-manifold targets, 0 FEA failures, reads `samples_synthetic/` — the discriminating control | ✅ **YES** |
| `export/` | top-5 STL export, `--tiles 2` — **5/5 watertight** | ✅ |
| `samples_live/` | the 50 held-out grids behind `evaluate/` (plus `sample_grid.png`, `paired_vs_physics_on.png`) | source data |
| `samples_synthetic/` | the 108 off-manifold grids behind `evaluate_synthetic/` | source data |
| `thin_wall_off_vs_on.png` | 3-panel wall-thickness analysis (the mechanism in `PROJECT.md` §8) | ✅ |

Headline: the ablation **matched or beat** the physics-ON baseline everywhere,
including off-manifold (validity 98.1% vs 97.2%, E 12.4% vs 13.8%, ν 1.2% vs 1.5%;
paired verdicts 0 ON-only / 1 OFF-only / 107 agree). See `PROJECT.md` §8 and
`SESSION_EXPORT.md` §6 for the framing this must be read with before it reaches a
reviewer. The off-manifold target set was rebuilt on 2026-10-20 (see `SESSION_EXPORT.md`
§6.3) — numbers dated before that used unreachable targets and are superseded.
