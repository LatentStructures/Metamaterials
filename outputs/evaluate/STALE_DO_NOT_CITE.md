# STALE — do not cite these numbers

This directory holds the **first** §7.1 evaluation run, from 2026-10-19 12:08,
**before** the periodic-connectivity fix in `src/validation/`.

It reports `validity_rate: 0.0` (0/50). That number is **wrong** and is the reason
the baseline was briefly believed to have failed: the connectivity check was
counting solid components in the *unwrapped* voxel box, so periodic unit cells
whose structure joins through their own faces were miscounted as disconnected and
every sample was rejected as invalid.

The corrected run of the same 50 targets is
`outputs/acceptance_30k/evaluate_live/`: **92.0% validity**, 0 FEA failures.

Canonical results for the baseline live in `PROJECT.md` §8; the runbook is
`RuntheBaseline.md`. Kept only as evidence of the fixed defect — cite the
`acceptance_30k` numbers, never these.
