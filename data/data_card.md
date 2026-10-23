# Dataset data card

- **resolution**: 32
- **families**: cubic_strut, octet_truss, gyroid
- **seed**: 0
- **generation date**: 2026-10-19 00:17:57
- **total records**: 2400 (failed/skipped: 0)
- **unique voxel grids**: 1023
- **manifest voxel paths**: repository-relative (portable across checkout locations)
- **wall time (FEA)**: 11251.4s

## FEA configuration
- base material: E_s=1.0, nu_s=0.3
- E_eff is in the same units as E_s; with E_s=1 it is the dimensionless ratio E_eff/E_s.
- void scale: 1e-06 (E_void = scale * E, nu_void = nu_base)
- validation tolerance: 0.05

## Conditioning targets (order ['E', 'relative_density', 'nu'])
| quantity | min | p50 | p95 | max |
|----------|-----|-----|-----|-----|
| E_eff | 0.0124015 | 0.130519 | 0.313049 | 0.38108 |
| relative_density | 0.0878906 | 0.354492 | 0.577393 | 0.614258 |
| nu_eff | 0.211983 | 0.266336 | 0.318684 | 0.347223 |

## Per-family counts
- cubic_strut: 800
- octet_truss: 800
- gyroid: 800

## FEA timing (s/cell)
- mean 85.515, median 83.071, p95 125.862, max 170.461

## Homogenization validation (closed-form preflight)

| benchmark | max rel err | ok |
|---|---|---|
| solid_cube | 4.61073e-12 | yes |
| laminate_x | 1.25371e-07 | yes |
| aligned_rods_x | 6.23112e-07 | yes |

## Periodic validity (synthesizability filter)
- overall: 87.7% (2104/2400)
- connectivity is counted **periodically** (components merged across
  wrapped faces); the unwrapped count is far lower and is wrong for
  unit cells.
- cubic_strut: 100.0% (800/800)
- gyroid: 75.4% (603/800)
- octet_truss: 87.6% (701/800)
- per-family breakdown recomputed 2026-10-21 with the committed filter
  (`evaluate_validity` as called by `_validity_summary`): all 296 failures
  are `min_wall_thickness`; zero connectivity and zero overhang failures.
  Overall total is unchanged.

## Effective diversity
- unique voxel grids: 1,023 of 2,400 rows (cubic 515/800, gyroid 368/800,
  octet 140/800). Duplicate rows are byte-identical grids with identical
  labels — loss re-weightings, not label noise — and none span more than one
  split. Full analysis: `FINDINGS.md` §3.1.

## Corrected copy: `data/v1f/`
A derived, gate-only-corrected copy of this dataset drops the 296
filter-invalid rows above and evens the per-(split × family) grid-multiplicity
skew, yielding a **100%-valid** dataset with the same 2,400 rows, the same
splits, and the same 800-per-family balance — at the cost of unique-grid
count (**935** vs 1,023 here). Labels were copied bit-for-bit; **no FEA was
re-run**. See `data/v1f/data_card.md` for the derivation, and note its
conditioning statistics differ, so sampling a V1f checkpoint requires
`--train-config configs/train_baseline_v1f.yaml`. This v1 copy remains the
ground truth for satisfiability verdicts and the historical pipeline.
