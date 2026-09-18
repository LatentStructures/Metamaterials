"""Batch dataset generation (ROADMAP section 3.3-3.4).

Reproducible from a single config + seed. Produces three HDF5 split files and a
Parquet manifest.  FEA homogenization is mandatory for the *real* run; a
``--mock-fea`` CLI flag is provided strictly for pipeline-testing (it prints a
loud warning and uses an unphysical placeholder stiffness).  Generation is
checkpointed every ``checkpoint_every`` cells and resumes automatically.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_yaml, set_seed, REPO_ROOT
from src.geometry import sample_for_density, voxels_to_tetra, periodic_pairing
from src.data.dataset import write_conditioned_hdf5, ConditioningStats

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEA interface
# ---------------------------------------------------------------------------

_HOMOGENIZER = None
_MOCK_FEA = False


def _load_homogenizer(validation_tolerance: float) -> None:
    global _HOMOGENIZER, _MOCK_FEA
    if _HOMOGENIZER is not None:
        return
    try:
        from src.fea.homogenization import homogenize_periodic_cell, validate_vs_analytical
        validate_vs_analytical(tolerance=validation_tolerance)
        _HOMOGENIZER = homogenize_periodic_cell
        LOG.info("FEA homogenization loaded; validation within %.1f%%",
                 100 * validation_tolerance)
    except ImportError:
        if _MOCK_FEA:
            _HOMOGENIZER = _mock_homogenizer()
            LOG.warning(
                "Using --mock-fea: stiffness is UNPHYSICAL. "
                "Do NOT commit mock data to production."
            )
        else:
            raise SystemExit(
                "src/fea/homogenization.py is not available.  "
                "Finish the FEA module (W3) before running the real dataset generation."
            )


def _mock_homogenizer():
    def homogenize_periodic_cell(voxels, mesh, pairings, res):
        rho = float(voxels.sum()) / voxels.size
        E_eff = rho ** 2.0
        nu = 0.3
        G = E_eff / (2 * (1 + nu))
        C6 = np.zeros((6, 6), dtype=np.float32)
        C6[0, 0] = C6[1, 1] = C6[2, 2] = E_eff
        C6[0, 1] = C6[0, 2] = C6[1, 2] = E_eff * nu / (1 - nu)
        C6[1, 0] = C6[2, 0] = C6[2, 1] = C6[0, 1]
        C6[3, 3] = C6[4, 4] = C6[5, 5] = G
        return C6
    return homogenize_periodic_cell


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_all(config: dict, rng: np.random.Generator) -> list[dict]:
    """Yield per-cell dicts with grid, achieved density, and target params."""
    res = config["resolution"]
    families = config["families"]
    density_lo, density_hi = config["relative_density_range"]
    num_total = config["num_cells"] * len(families)
    cells_per_family = [num_total // len(families)] * len(families)
    cells_per_family[-1] += num_total - sum(cells_per_family)

    samples = []
    for family, n in zip(families, cells_per_family):
        for _ in range(n):
            target_rho = rng.uniform(density_lo, density_hi)
            grid, achieved_rho = sample_for_density(family, target_rho, res)
            samples.append(dict(
                family=family,
                target_density=target_rho,
                achieved_density=achieved_rho,
                voxels=grid,
            ))
    return samples


def _stratify_splits(samples: list[dict], frac: dict[str, float],
                     density_range: tuple[float, float],
                     rng: np.random.Generator) -> dict[str, list[int]]:
    """Split sample indices 80/10/10 stratified by (family, density bucket)."""
    buckets: dict[tuple[str, int], list[int]] = {}
    n_buckets = 5
    density_lo, density_hi = density_range
    for i, s in enumerate(samples):
        b = min(int((s["achieved_density"] - density_lo) / (density_hi - density_lo) * n_buckets),
                n_buckets - 1)
        key = (s["family"], b)
        buckets.setdefault(key, []).append(i)

    result: dict[str, list[int]] = {k: [] for k in frac}
    for idxs in buckets.values():
        rng.shuffle(idxs)
        total = len(idxs)
        n_test = max(1, round(total * frac.get("test", 0.1)))
        n_val = max(1, round(total * frac.get("val", 0.1)))
        n_train = max(0, total - n_test - n_val)
        result["train"].extend(idxs[:n_train])
        result["val"].extend(idxs[n_train:n_train + n_val])
        result["test"].extend(idxs[n_train + n_val:])
    for k in result:
        rng.shuffle(result[k])
    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    c6_cols = [c for c in df.columns if c.startswith("C6_")]
    out = []
    for _, row in df.iterrows():
        C6 = np.zeros((6, 6), dtype=np.float32)
        if c6_cols:
            C6.flat[:] = row[c6_cols].to_numpy(dtype=np.float32)
        out.append({
            "family": row["family"],
            "achieved_density": float(row["achieved_density"]),
            "target_density": float(row["target_density"]),
            "voxel_path": Path(row["voxel_path"]),
            "cond": np.array(
                [row["cond_E"], row["cond_rho"], row["cond_nu"]], dtype=np.float32
            ),
            "C6": C6,
        })
    return out


def _save_checkpoint(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        c = r["cond"]
        row = {
            "family": r["family"],
            "achieved_density": r["achieved_density"],
            "target_density": r["target_density"],
            "voxel_path": str(r.get("voxel_path", "")),
            "cond_E": float(c[0]),
            "cond_rho": float(c[1]),
            "cond_nu": float(c[2]),
        }
        C6 = r.get("C6")
        if C6 is not None:
            for k in range(36):
                row[f"C6_{k}"] = float(C6.flat[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(config_path: str | Path, mock_fea: bool = False) -> None:
    global _MOCK_FEA
    _MOCK_FEA = mock_fea

    config = load_yaml(config_path)
    res = config["resolution"]
    seed = config["seed"]
    set_seed(seed)

    val_tol = config["homogenization"]["validation_tolerance"]
    _load_homogenizer(val_tol)

    checkpoint_every = config.get("checkpoint_every", 100)
    ckpt_dir = REPO_ROOT / "data" / "checkpoints"
    ckpt_file = ckpt_dir / f"ckpt_{Path(config_path).stem}_s{seed}.parquet"

    existing = _load_checkpoint(ckpt_file)
    voxdir = ckpt_dir / "vox"
    voxdir.mkdir(parents=True, exist_ok=True)

    samples = _sample_all(config, np.random.default_rng(seed))
    total = len(samples)
    per_family = ", ".join(
        f"{c} {f}" for f, c in zip(
            config["families"],
            [sum(1 for s in samples if s["family"] == f) for f in config["families"]],
        )
    )
    LOG.info("Prepared %d samples (%s)", total, per_family)

    records: list[dict] = []
    for i, s in enumerate(existing):
        voxels = np.load(s["voxel_path"])
        s["voxels"] = voxels
        records.append(s)
    skip = len(records)

    t0 = time.time()
    for i, s in enumerate(samples[skip:], start=skip):
        grid = s["voxels"]
        mesh = voxels_to_tetra(grid)
        pairings = periodic_pairing(mesh, res)
        C6 = _HOMOGENIZER(grid, mesh, pairings, res)
        rho = float(grid.sum()) / grid.size

        vox_path = voxdir / f"vox_{i:05d}.npy"
        np.save(vox_path, grid)

        s["C6"] = C6.astype(np.float32)
        s["achieved_density"] = rho
        s["voxel_path"] = vox_path
        s["cond"] = np.array(
            [config["base_material"]["E"], rho, config["base_material"]["nu"]],
            dtype=np.float32,
        )
        records.append(s)

        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(records, ckpt_file)
            elapsed = time.time() - t0
            LOG.info("checkpoint %d/%d (%.1f%%) at %.1fs", i + 1, total, 100 * (i + 1) / total, elapsed)

    _save_checkpoint(records, ckpt_file)

    splits = _stratify_splits(records, config["splits"],
                              config["relative_density_range"],
                              np.random.default_rng(seed + 1))

    all_conds = np.stack([r["cond"] for r in records])
    train_conds = all_conds[splits["train"]]
    stats = ConditioningStats(
        mean=train_conds.mean(axis=0),
        std=train_conds.std(axis=0) + 1e-8,
    )
    stats.mean = stats.mean.astype(np.float32)
    stats.std = stats.std.astype(np.float32)

    split_frac = config["splits"]
    for split_name in split_frac:
        idxs = splits[split_name]
        vox = np.stack([records[i]["voxels"] for i in idxs])
        cond = np.stack([records[i]["cond"] for i in idxs])
        cond_norm = stats.normalize(cond)
        stiff = np.stack([records[i]["C6"] for i in idxs]) if records[0].get("C6") is not None else None

        h5_path = REPO_ROOT / config["hdf5"].format(split=split_name)
        write_conditioned_hdf5(h5_path, vox, cond_norm, stiffness=stiff, stats=stats, chunk=64)
        LOG.info("wrote %s: %d samples to %s", split_name, len(idxs), h5_path)

    manifest_rows = []
    for i, r in enumerate(records):
        split_name = next(sn for sn, idxs in splits.items() if i in idxs)
        manifest_rows.append({
            "cell_id": i,
            "split": split_name,
            "family": r["family"],
            "achieved_density": float(r["achieved_density"]),
            "target_density": float(r["target_density"]),
            "voxel_path": str(r.get("voxel_path", "")),
            "hdf5_path": config["hdf5"].format(split=split_name),
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = REPO_ROOT / config["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    LOG.info("manifest written to %s", manifest_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to dataset.yaml")
    ap.add_argument("--mock-fea", action="store_true",
                    help="Use an UNPHYSICAL placeholder stiffness (pipeline testing only)")
    args = ap.parse_args()
    generate(args.config, mock_fea=args.mock_fea)


if __name__ == "__main__":
    main()