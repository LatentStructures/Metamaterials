# outputs/synthetic_30k — which subdirectory to use

The **off-manifold control** for the physics-ON baseline (`runs/baseline_clipfix_30k`),
sampled with `--weights live`, base seed 0, against the 108 `data/targets_synthetic.csv`
targets. Compare against the physics-OFF counterpart in
`outputs/ablation_30k/evaluate_synthetic/` and the on-manifold acceptance in
`outputs/acceptance_30k/`.

> **Rebuilt 2026-10-20 (target-set fix).** The 108-target set previously contained
> physically unreachable points (ν drawn independently of E and ρ); numbers dated
> before the fix (94.4% validity, ν R² 0.315, "variance compression") are superseded.
> See `PROJECT.md` §8.2.

| dir | what it is | cite? |
|---|---|---|
| `evaluate/` | **canonical off-manifold control**: 97.2% (105/108), 0 FEA failures, on the corrected physical target set; reads `samples/` | ✅ **YES** |
| `evaluate_k5/` | 5-draw error bar: 92.2% (498/540), 95% CI [90.0%, 94.5%]; reads `samples_k5/` — **predates the fix**, so its E/ν errors are inflated but validity is a conservative lower bound | ✅ for the interval only |
| `samples/` | the 108 grids behind `evaluate/` | source data |
| `samples_k5/` | the 540 grids behind `evaluate_k5/` | source data |

Headline: validity **97.2%** off-manifold with 0 FEA failures rules out memorization
(the 50 on-manifold acceptance targets are exact dataset cells). Off-manifold ν R²
**0.858** passes the ≥0.85 gate; E R² **0.808** misses by 0.04, which is target-spread
rather than model error (E's error gate passes at 13.75% ≤ 15%). See `PROJECT.md` §8.
# Historical checkpoint note (2026-10-22)

The physics-on samples here came from the legacy zero-padded Option-B proxy.
Their sample-level metrics remain valid for that checkpoint; they do not
validate the corrected periodic proxy. The corrected-proxy off-manifold
controls are 95.4% (V1 physics-on), 98.1% (V1 λ₁=0), 95.4% (V1f
physics-on) and 96.3% (V1f λ₁=0), all 0 FEA failures — see
`outputs/final_acceptance/`, `outputs/final_acceptance_v1f/`, and `PROJECT.md`
§8. This directory remains the citable source for the **legacy** 97.2%
headline and its pre-fix 5-draw result (92.2%, sample-level 95% CI
[90.0%, 94.5%]). The corrected V1f 5-draw follow-up on the fixed target set is
recorded separately in `outputs/final_acceptance_v1f/errorbar_k5/`; do not pool
the two target sets.
