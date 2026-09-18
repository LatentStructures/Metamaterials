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
from src.geometry import (
    sample_for_density,
    voxels_to_tetra,
    periodic_pairing,
)
from src.data.dataset import write_conditioned_hdf5, ConditioningStats
from src.fea.property_extraction import effective_properties

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEA interface
# ---------------------------------------------------------------------------

_HOMOGENIZER = None
_MOCK_FEA = False
_VALIDATION_SUMMARY = None


def _write_validation_table(summary: dict) -> None:
    """Persist the closed-form benchmark preflight as ``data/validation_table.parquet``."""
    rows = []
    for name, info in summary["benchmarks"].items():
        rows.append({
            "benchmark": name,
            "max_rel_err": float(info["max_rel_err"]),
            "within_tolerance": bool(info["ok"]),
        })
    path = REPO_ROOT / "data" / "validation_table.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    LOG.info("validation table written to %s", path)


def _load_homogenizer(validation_tolerance: float, res: int,
                      base_E: float, base_nu: float, void_scale: float):
    global _HOMOGENIZER, _MOCK_FEA, _VALIDATION_SUMMARY
    if _VALIDATION_SUMMARY is not None:
        return _VALIDATION_SUMMARY
    try:
        from src.fea import set_base_material, set_void_scale
        from src.fea.homogenization import Homogenizer, validate_vs_analytical
        set_base_material(base_E, base_nu)
        set_void_scale(void_scale)
        summary = validate_vs_analytical(tolerance=validation_tolerance, res=res)
        _VALIDATION_SUMMARY = summary
        _write_validation_table(summary)
        _HOMOGENIZER = Homogenizer(res)
        LOG.info(
            "FEA homogenization loaded (res %d, E %.3g, nu %.3g, void scale %.3g); "
            "validation within %.1f%%",
            res, base_E, base_nu, void_scale, 100 * validation_tolerance,
        )
        return summary
    except ImportError:
        if _MOCK_FEA:
            _HOMOGENIZER = _MockHomogenizer()
            LOG.warning(
                "Using --mock-fea: stiffness is UNPHYSICAL. "
                "Do NOT commit mock data to production."
            )
        else:
            raise SystemExit(
                "src/fea/homogenization.py is not available.  "
                "Run inside the conda env where dolfinx is installed."
            )
        return None


class _MockHomogenizer:
    """Ersatz callable matching the ``Homogenizer`` interface (pipeline tests only)."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, voxels: np.ndarray, mesh, pairings, res: int) -> np.ndarray:
        t0 = time.time()
        rho = float(voxels.sum()) / voxels.size
        E_eff = rho ** 2.0
        nu = 0.3
        G = E_eff / (2 * (1 + nu))
        C6 = np.zeros((6, 6), dtype=np.float32)
        C6[0, 0] = C6[1, 1] = C6[2, 2] = E_eff
        C6[0, 1] = C6[0, 2] = C6[1, 2] = E_eff * nu / (1 - nu)
        C6[1, 0] = C6[2, 0] = C6[2, 1] = C6[0, 1]
        C6[3, 3] = C6[4, 4] = C6[5, 5] = G
        self.times.append(time.time() - t0)
        return C6


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
            "fea_time_s": float(row.get("fea_time_s", np.nan)),
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
            "fea_time_s": float(r.get("fea_time_s", np.nan)),
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

    base_E = float(config["base_material"]["E"])
    base_nu = float(config["base_material"]["nu"])
    void_scale = float(config["homogenization"].get("void_scale", 1.0e-6))
    val_tol = config["homogenization"]["validation_tolerance"]
    _VALIDATION_SUMMARY = _load_homogenizer(val_tol, res, base_E, base_nu, void_scale)

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

    max_attempts = int(config.get("fea_retries", 3))
    failed = 0
    t0 = time.time()
    for i, s in enumerate(samples[skip:], start=skip):
        grid = s["voxels"]
        target_rho = s["target_density"]

        C6 = None
        for attempt in range(max_attempts):
            try:
                mesh = voxels_to_tetra(grid)
                pairings = periodic_pairing(mesh, res)
                C6 = _HOMOGENIZER(grid, mesh, pairings, res)
                break
            except Exception:
                LOG.warning("sample %d (%s, target rho %.3f) FEA failed "
                            "(attempt %d/%d)", i, s["family"], target_rho,
                            attempt + 1, max_attempts)
                if attempt + 1 < max_attempts:
                    jitter = float(np.random.default_rng(seed + i + attempt)
                                   .uniform(-2e-2, 2e-2))
                    target = min(0.999, max(1e-3, target_rho + jitter))
                    grid, _ = sample_for_density(s["family"], target, res)

        if C6 is None:
            failed += 1
            LOG.error("sample %d (%s) permanently failed after %d attempts; skipping",
                      i, s["family"], max_attempts)
            continue

        try:
            ep = effective_properties(C6)
        except ValueError as exc:
            failed += 1
            LOG.error("sample %d (%s) produced an unphysical stiffness (%s); skipping",
                      i, s["family"], exc)
            continue

        rho = float(grid.sum()) / grid.size
        fea_time = _HOMOGENIZER.times[-1] if _HOMOGENIZER.times else float("nan")

        vox_path = voxdir / f"vox_{i:05d}.npy"
        np.save(vox_path, grid)

        s["C6"] = C6.astype(np.float32)
        s["achieved_density"] = rho
        s["fea_time_s"] = float(fea_time)
        s["voxel_path"] = vox_path
        s["cond"] = np.array([ep.E, rho, ep.nu], dtype=np.float32)
        records.append(s)

        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(records, ckpt_file)
            elapsed = time.time() - t0
            LOG.info("checkpoint %d/%d (%.1f%%) at %.1fs", i + 1, total, 100 * (i + 1) / total, elapsed)

    _save_checkpoint(records, ckpt_file)
    LOG.info("FEA complete: %d records (%d failed/skipped); "
             "mean %.3f s/cell, total %.1f s",
             len(records), failed,
             float(np.mean([r.get("fea_time_s", 0.0) for r in records])) if records else 0.0,
             time.time() - t0)

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
            "E_eff": float(r["cond"][0]),
            "nu_eff": float(r["cond"][2]),
            "fea_time_s": float(r.get("fea_time_s", np.nan)),
            "voxel_path": str(r.get("voxel_path", "")),
            "hdf5_path": config["hdf5"].format(split=split_name),
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = REPO_ROOT / config["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    LOG.info("manifest written to %s", manifest_path)

    _write_data_card(manifest, config, records, failed,
                     void_scale, float(time.time() - t0), _VALIDATION_SUMMARY)


def _write_data_card(manifest: pd.DataFrame, config: dict, records: list[dict],
                     failed: int, void_scale: float, wall_s: float,
                     validation: dict | None = None) -> None:
    """Write ``data/data_card.md`` summarising the generated dataset (ROADMAP 3.4)."""
    res = config["resolution"]
    e_eff = manifest["E_eff"].to_numpy()
    nu_eff = manifest["nu_eff"].to_numpy()
    rho_raw = manifest["achieved_density"].to_numpy()
    fea_times = manifest["fea_time_s"].dropna().to_numpy()

    def _f(v: float) -> str:
        return f"{v:.6g}"

    lines = [
        "# Dataset data card",
        "",
        f"- **resolution**: {res}",
        f"- **families**: {', '.join(str(f) for f in config['families'])}",
        f"- **seed**: {config['seed']}",
        f"- **generation date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **total records**: {len(records)} (failed/skipped: {failed})",
        f"- **wall time (FEA)**: {wall_s:.1f}s",
        "",
        "## FEA configuration",
        f"- base material: E={config['base_material']['E']}, "
        f"nu={config['base_material']['nu']}",
        f"- void scale: {_f(void_scale)} (E_void = scale * E, nu_void = nu_base)",
        f"- validation tolerance: {config['homogenization']['validation_tolerance']}",
        "",
        "## Conditioning targets (order [E, relative_density, nu])",
        "| quantity | min | p50 | p95 | max |",
        "|----------|-----|-----|-----|-----|",
        f"| E_eff | {_f(e_eff.min())} | {_f(np.median(e_eff))} | "
        f"{_f(np.percentile(e_eff, 95))} | {_f(e_eff.max())} |",
        f"| relative_density | {_f(rho_raw.min())} | {_f(np.median(rho_raw))} | "
        f"{_f(np.percentile(rho_raw, 95))} | {_f(rho_raw.max())} |",
        f"| nu_eff | {_f(nu_eff.min())} | {_f(np.median(nu_eff))} | "
        f"{_f(np.percentile(nu_eff, 95))} | {_f(nu_eff.max())} |",
        "",
        "## Per-family counts",
    ]
    for family, n in manifest["family"].value_counts().items():
        lines.append(f"- {family}: {n}")
    if len(fea_times):
        lines += [
            "",
            "## FEA timing (s/cell)",
            f"- mean {fea_times.mean():.3f}, median {np.median(fea_times):.3f}, "
            f"p95 {np.percentile(fea_times, 95):.3f}, max {fea_times.max():.3f}",
        ]
    if validation is not None:
        lines += [
            "",
            "## Homogenization validation (closed-form preflight)",
            "",
            "| benchmark | max rel err | ok |",
            "|---|---|---|",
        ]
        for name, info in validation["benchmarks"].items():
            lines.append(
                f"| {name} | {_f(info['max_rel_err'])} | "
                f"{'yes' if info['ok'] else 'no'} |"
            )

    path = REPO_ROOT / "data" / "data_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    LOG.info("data card written to %s", path)


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