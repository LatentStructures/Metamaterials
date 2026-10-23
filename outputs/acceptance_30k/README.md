# outputs/acceptance_30k — which subdirectory to use

| dir | what it is | cite? |
|---|---|---|
| `evaluate_live/` | **canonical §7.1 acceptance**: 92.0% (46/50), 0 FEA failures, reads `samples_live/` | ✅ **YES** |
| `evaluate/` | same `samples_live/` grids evaluated **before** the periodic-connectivity/wall filter fix → 54%; all 50 FEA results identical to `evaluate_live`, only verdicts differ | ❌ no |
| `evaluate_ema/` | the **legacy single-decay EMA arm**: 14.0% (7/50) + 1 FEA solver failure, reads `samples/` — decay 0.9999 averages a less-converged region. Multi-decay selection has since been implemented and measured (V1 → `ema-0.9999`; V1f → `ema-0.9995` on, `live` off; `outputs/ema_validation*/`); see `PROJECT.md` §8 | only when discussing the historical EMA A/B |
| `export_final/` | top-5 STL export, `--tiles 1`, void padding — **5/5 watertight** | ✅ |
| `export_tiled/` | top-5 STL export, `--tiles 2` — **5/5 watertight** | ✅ |
| `evidence/` | display pack from `python -m src.evidence`: reference-vs-generated pairs (all 50), selected-candidate hero panel + 3D render, export montage, `evidence_manifest.json` | ✅ |
| `samples_live/` | the live-weight samples (50 grids) behind `evaluate_live/` | source data |
| `samples/` | the EMA samples (50 grids) behind `evaluate_ema/` | source data |

All 4 acceptance failures are gyroid at ρ < 0.25 — quote 92.0% with its
conditioning (pooled validity ≥ ρ 0.25 is 98.7%; `FINDINGS.md` §2).
See `PROJECT.md` §8.2 and `FINDINGS.md` §2 for the numbers and their caveats.
# Historical checkpoint note (2026-10-22)

This acceptance artifact was generated from the legacy zero-padded physics-on
checkpoint. Its reported metrics remain tied to that checkpoint; it is not a
validation of the corrected periodic physics proxy. The matched corrected-proxy
runs are complete — V1 (`outputs/final_acceptance/`, 88.0% physics-on /
94.0% λ₁=0) and V1f (`outputs/final_acceptance_v1f/`, 90.0% / 94.0%). Cite
`PROJECT.md` §8 "Corrected-proxy matched results" for the current comparison;
cite this directory only for the historical 92.0% headline and the STL export /
evidence pack, which remain the citable artifacts for those deliverables.
