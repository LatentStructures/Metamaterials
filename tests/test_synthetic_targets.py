"""Tests for the off-manifold target builder (src/data/build_synthetic_targets.py).

``data/targets_synthetic.csv`` is the source of the headline off-manifold
control (SESSION_EXPORT section 5.4), so the builder must be deterministic, keep
every target a safe distance from real cells, interpolate only inside a family's
realizable band **and keep nu physically consistent with (E, rho)** — see
``test_targets_are_physically_reachable`` for why that last property matters —
and emit the sampler's locked column order.
"""
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from src.config import CONDITIONING_ORDER, REPO_ROOT
from src.data import build_synthetic_targets as bst

N_PER_FAMILY = 4
MIN_GAP = 0.012


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / "data/manifest.parquet")


@pytest.fixture(scope="module")
def targets() -> pd.DataFrame:
    return bst.build_targets(n_per_family=N_PER_FAMILY, min_gap=MIN_GAP, seed=0)


def test_row_count_and_columns(targets: pd.DataFrame, manifest: pd.DataFrame) -> None:
    assert len(targets) == N_PER_FAMILY * manifest["family"].nunique()
    assert list(targets.columns) == [*CONDITIONING_ORDER, "source_family"]


def test_every_target_keeps_min_gap_to_any_dataset_cell(
    manifest: pd.DataFrame, targets: pd.DataFrame
) -> None:
    # The builder's tree is (E, nu, rho); the CSV/query order is (E, rho, nu).
    tree = cKDTree(manifest[["E_eff", "nu_eff", "achieved_density"]].to_numpy(float))
    query = targets[["E", "nu", "relative_density"]].to_numpy(float)
    dist, _ = tree.query(query)
    assert dist.min() >= MIN_GAP - 1e-12


def test_targets_stay_inside_their_family_band(
    manifest: pd.DataFrame, targets: pd.DataFrame
) -> None:
    # E/rho are convex combinations of two same-family cells; nu comes from the
    # family's fitted nu=f(E,rho) relation, so it must stay in the observed
    # band as well as be physically consistent (see the test below).
    for family, grp in targets.groupby("source_family"):
        fam = manifest[manifest["family"] == family]
        assert grp["E"].between(fam["E_eff"].min(), fam["E_eff"].max()).all()
        assert grp["relative_density"].between(
            fam["achieved_density"].min(), fam["achieved_density"].max()
        ).all()
        assert grp["nu"].between(fam["nu_eff"].min(), fam["nu_eff"].max()).all()


def _nu_resid_std(fam: pd.DataFrame) -> float:
    """Residual std of the family's own nu ~ 1 + E + rho fit (physical scatter)."""
    X = np.column_stack([np.ones(len(fam)), fam["E_eff"], fam["achieved_density"]])
    y = fam["nu_eff"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.std(y - X @ beta))


def test_targets_are_physically_reachable(
    manifest: pd.DataFrame, targets: pd.DataFrame
) -> None:
    """Every target's nu must be consistent with some real cell's nu at that (E, rho).

    Regression guard. An earlier version drew nu uniformly over the family band,
    independent of E and rho, which produced targets whose (E, rho) sat on a real
    cell while demanding a nu up to 0.11 away from that cell's actual nu. Those
    targets are unreachable (one (E, rho) has one nu) and they silently drove the
    off-manifold nu R^2 down to 0.31, mis-attributed to variance compression.
    Real cells at the same (E, rho) agree to ~0.000, so the tolerance here is
    tight on purpose.
    """
    for family, grp in targets.groupby("source_family"):
        fam = manifest[manifest["family"] == family]
        tree = cKDTree(fam[["E_eff", "achieved_density"]].to_numpy(float))
        _, idx = tree.query(grp[["E", "relative_density"]].to_numpy(float))
        mismatch = np.abs(grp["nu"].to_numpy(float) - fam["nu_eff"].to_numpy(float)[idx])
        # Tolerance scales with the family's own physical scatter; a floor keeps
        # it meaningful for the very tight families (octet nu std ~0.007).
        tol = max(4.0 * _nu_resid_std(fam), 0.012)
        assert mismatch.max() < tol, (
            f"{family}: max nu mismatch {mismatch.max():.4f} >= tolerance {tol:.4f}"
        )


def test_same_seed_reproduces_identical_targets() -> None:
    first = bst.build_targets(3, MIN_GAP, seed=0)
    second = bst.build_targets(3, MIN_GAP, seed=0)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_the_set() -> None:
    assert not bst.build_targets(3, MIN_GAP, seed=0).equals(
        bst.build_targets(3, MIN_GAP, seed=1)
    )


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory, *extra: str):
    out = tmp_path / "targets.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_synthetic_targets", "--n-per-family", "3", "--out", str(out), *extra],
    )
    bst.main()
    return out


def test_main_writes_the_sampler_column_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    out = _run_main(monkeypatch, tmp_path)
    df = pd.read_csv(out)
    assert list(df.columns) == CONDITIONING_ORDER
    assert len(df) == 9  # 3 locked families x --n-per-family 3


def test_main_with_family_keeps_the_diagnostic_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    out = _run_main(monkeypatch, tmp_path, "--with-family")
    df = pd.read_csv(out)
    assert list(df.columns) == [*CONDITIONING_ORDER, "source_family"]
