# outputs/acceptance_30k — which subdirectory to use

| dir | what it is | cite? |
|---|---|---|
| `evaluate_live/` | **canonical §7.1 acceptance**: 92.0% (46/50), 0 FEA failures, reads `samples_live/` | ✅ **YES** |
| `evaluate/` | same `samples_live/` grids evaluated **before** the periodic-connectivity/wall filter fix → 54%; all 50 FEA results identical to `evaluate_live`, only verdicts differ | ❌ no |
| `evaluate_ema/` | the **EMA arm**: 14.0% (7/50) + 1 FEA solver failure, reads `samples/` — a **documented decay/schedule artefact** (decay 0.9999 averages a less-converged region; see `PROJECT.md` §8 / `SESSION_EXPORT.md` §5.6). Sample with `--weights live` | only when discussing the EMA A/B |
| `export_final/` | top-5 STL export, `--tiles 1`, void padding — **5/5 watertight** | ✅ |
| `export_tiled/` | top-5 STL export, `--tiles 2` — **5/5 watertight** | ✅ |
| `samples_live/` | the live-weight samples (50 grids) behind `evaluate_live/` | source data |
| `samples/` | the EMA samples (50 grids) behind `evaluate_ema/` | source data |

See `SESSION_EXPORT.md` §5 and `PROJECT.md` §8 for the numbers and their caveats.
