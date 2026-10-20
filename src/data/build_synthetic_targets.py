"""Build an OFF-MANIFOLD acceptance target set.

The original ``data/targets_test_50.csv`` was drawn from the held-out test split's
own measured properties, so every target coincides exactly with an existing cell
(distance 0.000000) and sampling degenerates into conditional reconstruction.

This script instead constructs targets that are *between* the data points:

* for each family, interpolate E and relative_density between random pairs of
  that family's cells (so the pair is achievable on-manifold in principle),
* draw nu from the family's own measured relation ``nu = f(E, rho)``, plus
  residual scatter, so the requested nu is physically consistent with the
  requested E and rho,
* reject any target that lands closer than ``--min-gap`` to an existing cell.

**Why nu must come from ``f(E, rho)`` and not from a uniform draw over the
family band.** Physical unit cells have ``nu`` almost fully determined by their
``(E, rho)`` pair: fitting ``nu ~ E + rho`` on each family gives R^2 =
0.87/0.99/0.93 (cubic/gyroid/octet) with residual std 0.0117/0.0012/0.0018.
A previous version of this script drew ``nu`` uniformly across the family band,
independent of ``E`` and ``rho``. That produces targets whose ``(E, rho)`` sits
on a dataset cell while the demanded ``nu`` is up to 0.11 away from that cell's
actual ``nu`` -- physically unreachable, since one ``(E, rho)`` has one ``nu``.
Worse, such a target passed the ``min_gap`` filter *because of* the nu
displacement, so the filter never caught it. Those unreachable targets made
off-manifold ``nu`` R^2 meaningless (the model was scored on refusing to emit
impossible values), which was originally mis-attributed to "variance
compression". Drawing nu from ``f(E, rho)`` removes the artifact.

Output columns match the sampler's expectation: ``E, relative_density, nu``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import REPO_ROOT


def _fit_nu_model(E: np.ndarray, rho: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit of ``nu ~ 1 + E + rho``, returning (beta, residual_std).

    This is the family's measured physical relation: the model must satisfy it to
    be a reachable target (see the module docstring).
    """
    X = np.column_stack([np.ones_like(E), E, rho])
    beta, *_ = np.linalg.lstsq(X, nu, rcond=None)
    resid = nu - X @ beta
    return beta, float(resid.std(ddof=3))


def build_targets(n_per_family: int, min_gap: float, seed: int,
                  nu_jitter: float = 1.0) -> pd.DataFrame:
    manifest = pd.read_parquet(REPO_ROOT / "data/manifest.parquet")
    props = manifest[["E_eff", "nu_eff", "achieved_density"]].to_numpy(float)
    tree = cKDTree(props)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for family in sorted(manifest.family.unique()):
        fam = manifest[manifest.family == family]
        idx = fam.index.to_numpy()
        vals = manifest.loc[idx, ["E_eff", "nu_eff", "achieved_density"]].to_numpy(float)
        # The family's physical nu = f(E, rho) relation, learned from its own cells.
        beta, resid_std = _fit_nu_model(vals[:, 0], vals[:, 2], vals[:, 1])
        made = 0
        attempts = 0
        while made < n_per_family:
            attempts += 1
            if attempts > 200 * n_per_family:
                raise RuntimeError(f"could not place {n_per_family} targets for {family}")
            a, b = rng.integers(0, len(vals), 2)
            # interior interpolation only: stay strictly between data points
            w = rng.uniform(0.2, 0.8)
            E = w * vals[a, 0] + (1 - w) * vals[b, 0]
            rho = w * vals[a, 2] + (1 - w) * vals[b, 2]
            # nu from the family's fitted relation + residual-scale scatter, so the
            # requested nu stays physically consistent with (E, rho).
            nu = float(beta[0] + beta[1] * E + beta[2] * rho
                       + rng.normal(0.0, nu_jitter * resid_std))
            nu_lo, nu_hi = vals[:, 1].min(), vals[:, 1].max()
            if not (nu_lo <= nu <= nu_hi):
                continue
            cand = np.array([E, nu, rho])
            if tree.query(cand)[0] < min_gap:
                continue
            rows.append({"E": E, "relative_density": rho, "nu": nu,
                         "source_family": family})
            made += 1
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-family", type=int, default=36)
    ap.add_argument("--min-gap", type=float, default=0.012,
                    help="min distance to the nearest existing cell in (E, nu, rho)")
    ap.add_argument("--nu-jitter", type=float, default=1.0,
                    help="multiplier on the nu=f(E,rho) residual std when drawing nu; "
                         "0.0 makes every target exactly on the physical manifold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/targets_synthetic.csv"))
    ap.add_argument("--with-family", action="store_true",
                    help="keep the source_family column (diagnostics only)")
    args = ap.parse_args()

    df = build_targets(args.n_per_family, args.min_gap, args.seed, args.nu_jitter)
    manifest = pd.read_parquet(REPO_ROOT / "data/manifest.parquet")
    tree = cKDTree(manifest[["E_eff", "nu_eff", "achieved_density"]].to_numpy(float))
    d, _ = tree.query(df[["E", "nu", "relative_density"]].to_numpy(float))
    print(f"built {len(df)} targets | gap to nearest cell: min {d.min():.5f} "
          f"median {np.median(d):.5f} | families {df.source_family.value_counts().to_dict()}")

    # Reachability guard: a target is only meaningful if some real cell with a
    # similar (E, rho) has a similar nu. Report the worst mismatch so an
    # unreachable set fails loudly here instead of silently poisoning nu R^2.
    worst = []
    for family, tg in df.groupby("source_family"):
        fam = manifest[manifest.family == family]
        ftree = cKDTree(fam[["E_eff", "achieved_density"]].to_numpy(float))
        _, j = ftree.query(tg[["E", "relative_density"]].to_numpy(float))
        worst.append(np.abs(tg.nu.to_numpy(float) - fam.nu_eff.to_numpy(float)[j]))
    worst = np.concatenate(worst)
    print(f"reachability: |nu - nu(nearest (E,rho) cell)| median {np.median(worst):.4f} "
          f"max {worst.max():.4f}  (want << family nu std; >0.02 means unreachable targets)")

    out = df if args.with_family else df[["E", "relative_density", "nu"]]
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
