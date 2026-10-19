# Dataset data card

- **resolution**: 32
- **families**: cubic_strut, octet_truss, gyroid
- **seed**: 0
- **generation date**: 2026-10-19 00:17:57
- **total records**: 2400 (failed/skipped: 0)
- **unique voxel grids**: 1023
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
