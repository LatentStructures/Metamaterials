"""Batch dataset generation (ROADMAP section 3.3-3.4).

Reproducible from a single config + seed. Produces three HDF5 split files and a
Parquet manifest.  FEA homogenization is mandatory for the *real* run; a
``--mock-fea`` CLI flag is provided strictly for pipeline-testing (it prints a
loud warning and uses an unphysical placeholder stiffness).  Generation is
checkpointed every ``checkpoint_every`` cells and resumes automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CONDITIONING_ORDER,
    REPO_ROOT,
    check_conditioning_order,
    load_yaml,
    set_seed,
)
from src.data.dataset import ConditioningStats, write_conditioned_hdf5
from src.fea.property_extraction import effective_properties
from src.geometry import (
    periodic_pairing,
    sample_for_density,
    voxels_to_tetra,
)

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# FEA interface
# --------------------------------------------------------------------------- #


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


def _load_homogenizer(mock_fea: bool, validation_tolerance: float, res: int,
                      base_E: float, base_nu: float, void_scale: float):
    """Return ``(homogenizer, validation_summary)`` after the analytical preflight.

    ``validation_summary`` is ``None`` for the mock homogenizer.  Raises
    ``SystemExit`` when dolfinx is unavailable and ``mock_fea`` is not set.
    """
    try:
        from src.fea import set_base_material, set_void_scale
        from src.fea.homogenization import Homogenizer, validate_vs_analytical
        set_base_material(base_E, base_nu)
        set_void_scale(void_scale)
        summary = validate_vs_analytical(tolerance=validation_tolerance, res=res)
        _write_validation_table(summary)
        LOG.info(
            "FEA homogenization loaded (res %d, E %.3g, nu %.3g, void scale %.3g); "
            "validation within %.1f%%",
            res, base_E, base_nu, void_scale, 100 * validation_tolerance,
        )
        return Homogenizer(res), summary
    except ImportError:
        if not mock_fea:
            raise SystemExit(
                "src/fea/homogenization.py is not available.  "
                "Run inside the conda env where dolfinx is installed."
            ) from None
        LOG.warning(
            "Using --mock-fea: stiffness is UNPHYSICAL. "
            "Do NOT commit mock data to production."
        )
        return _MockHomogenizer(), None


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


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #

def _sample_all(config: dict, rng: np.random.Generator) -> list[dict]:
    """Per-cell target specs (family + target density), in canonical order.

    Voxel grids are NOT materialized here: geometry generation is deterministic
    (``sample_for_density`` is pure) but ``octet_truss`` rasterization is costly
    (~seconds), so real runs render each grid inside the FEA worker that is
    about to solve it.  The parent only draws the cheap target densities in the
    exact same order a serial run would.
    """
    families = config["families"]
    density_lo, density_hi = config["relative_density_range"]
    num_total = config["num_cells"] * len(families)
    cells_per_family = [num_total // len(families)] * len(families)
    cells_per_family[-1] += num_total - sum(cells_per_family)

    samples = []
    shape_ranges = config.get("shape_parameter_ranges", {})
    for family, n in zip(families, cells_per_family, strict=True):
        for _ in range(n):
            target_rho = rng.uniform(density_lo, density_hi)
            sample = {
                "family": family,
                "target_density": target_rho,
            }
            if family in shape_ranges:
                low, high = shape_ranges[family]
                sample["shape_parameter"] = float(rng.uniform(low, high))
            samples.append(sample)
    return samples


def _stratify_splits(samples: list[dict], frac: dict[str, float],
                     density_range: tuple[float, float],
                     rng: np.random.Generator) -> dict[str, list[int]]:
    """Stratify by family/density while keeping identical voxel grids together."""
    groups: dict[bytes, list[int]] = {}
    for i, sample in enumerate(samples):
        voxels = sample.get("voxels")
        # Before FEA, samples have no grids yet. In that case each requested
        # design remains its own group; generation calls this after grids exist.
        key = (np.ascontiguousarray(voxels).tobytes()
               if voxels is not None else f"sample:{i}".encode())
        groups.setdefault(key, []).append(i)

    buckets: dict[tuple[str, int], list[bytes]] = {}
    n_buckets = 5
    density_lo, density_hi = density_range
    for key, indexes in groups.items():
        # Identical grids can arise across parameter samples (and occasionally
        # across family labels). Use a deterministic representative for the
        # stratum, but assign every member of the group to the same split.
        s = samples[indexes[0]]
        rho = float(s.get("achieved_density", s.get("target_density")))
        b = min(int((rho - density_lo) / (density_hi - density_lo) * n_buckets),
                n_buckets - 1)
        buckets.setdefault((s["family"], b), []).append(key)

    result: dict[str, list[int]] = {k: [] for k in frac}
    for group_keys in buckets.values():
        rng.shuffle(group_keys)
        if len(group_keys) < len(frac):
            # Too few independent designs to populate every split without
            # leakage; keep this stratum in train and let the target-count gate
            # below reject an underpowered held-out set.
            result["train"].extend(i for key in group_keys for i in groups[key])
            continue
        row_total = sum(len(groups[key]) for key in group_keys)
        targets = {split: row_total * weight for split, weight in frac.items()}
        assigned = dict.fromkeys(frac, 0)
        # Place large duplicate groups first, balancing by number of records
        # while keeping each exact geometry wholly within one split.
        ordered = sorted(group_keys, key=lambda key: len(groups[key]), reverse=True)
        for pos, key in enumerate(ordered):
            empty = [split for split in frac if assigned[split] == 0]
            remaining = len(ordered) - pos
            if empty and remaining <= len(empty):
                split = max(empty, key=lambda name: targets[name])
            else:
                split = max(
                    frac,
                    key=lambda name: targets[name] - assigned[name],
                )
            result[split].extend(groups[key])
            assigned[split] += len(groups[key])
    for idxs in result.values():
        rng.shuffle(idxs)
    return result


def _check_cross_split_leakage(records: list[dict], splits: dict[str, list[int]]) -> int:
    """Check that identical voxel grids do not leak across splits (ROADMAP 3.4).

    Returns the number of cross-split duplicate pairs found.
    """
    hashes: dict[bytes, tuple[str, int]] = {}
    leakages = 0
    for split_name, idxs in splits.items():
        for i in idxs:
            v_bytes = np.ascontiguousarray(records[i]["voxels"]).tobytes()
            if v_bytes in hashes:
                prev_split, prev_i = hashes[v_bytes]
                if prev_split != split_name:
                    leakages += 1
                    LOG.warning(
                        "cross-split identical voxel grid: %s cell %d matches %s cell %d",
                        prev_split, prev_i, split_name, i
                    )
            else:
                hashes[v_bytes] = (split_name, i)
    return leakages


def _unique_test_conditions(records: list[dict], test_indices: list[int],
                            count: int, seed: int) -> np.ndarray:
    """Choose distinct raw [E, relative_density, nu] targets from the test split."""
    by_condition: dict[tuple[float, ...], np.ndarray] = {}
    for i in test_indices:
        c = np.asarray(records[i]["cond"], dtype=np.float32)
        by_condition.setdefault(tuple(float(v) for v in c), c)
    if len(by_condition) < count:
        raise RuntimeError(
            f"test split has only {len(by_condition)} distinct conditioning vectors; "
            f"need {count} for the configured held-out acceptance targets"
        )
    values = list(by_condition.values())
    selected = np.random.default_rng(seed).choice(len(values), size=count, replace=False)
    return np.stack([values[int(i)] for i in selected])


# --------------------------------------------------------------------------- #
# Worker functions (multiprocess FEA path)
# --------------------------------------------------------------------------- #

_WKOPTS: dict = {}


def _init_worker(opts: dict) -> None:
    """Per-process setup: independent Homogenizer + cached res-locked mesh.

    Also caps the intra-process BLAS/PETSc thread count so ``n_workers``
    processes never *multiply* the core count (on a 20-core box, 12 workers
    with unlimited OpenBLAS threads thrash instead of scale).
    """
    threads = max(1, int(opts.get("omp_threads", 1)))
    # NOTE: spawned children re-import this module (and numpy/torch) before the
    # initializer runs, so these vars only bind to libraries loaded *after* this
    # point -- dolfinx/PETSc/scipy, imported just below, which are the heavy
    # consumers anyway. OpenBLAS/torch already loaded in the child are unaffected.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    from src.fea import set_base_material, set_void_scale
    from src.fea.homogenization import Homogenizer
    _WKOPTS.clear()
    _WKOPTS.update(opts)
    set_base_material(opts["base_E"], opts["base_nu"])
    set_void_scale(opts["void_scale"])
    _WKOPTS["homogenizer"] = _MockHomogenizer() if opts["mock_fea"] else Homogenizer(opts["res"])
    _WKOPTS["mesh"], _WKOPTS["pairings"] = _build_mesh_and_pairings(opts["res"])
    LOG.info("worker %s ready (pid %d, %d BLAS thread%s)",
             "mock" if opts["mock_fea"] else "FEA", os.getpid(),
             threads, "s" if threads != 1 else "")


def _build_mesh_and_pairings(res: int) -> tuple:
    """The structured lattice mesh depends only on ``res``, not on the voxels.

    The void flags in the meshio mesh are never read by the homogenizer (it
    derives per-cell material from the voxel grid at solve time), so building
    the mesh and PBC tables once per resolution is exact -- and avoids
    rebuilding 196,608 tets for every one of ~3,000 cells.
    """
    dummy = np.zeros((res, res, res), dtype=np.uint8)
    mesh = voxels_to_tetra(dummy)
    return mesh, periodic_pairing(mesh, res)


def _homogenize_with_retry(family: str, target_rho: float, res: int,
                           homogenizer, mesh, pairings: dict,
                           max_attempts: int, seed: int, index: int,
                           shape_parameter: float = 1.0
                           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Rasterize + homogenize one cell, retrying with a density jitter on failure.

    Returns ``(C6, final_grid, fea_time_s)`` where ``final_grid`` is the grid
    actually solved (a jitter retry may have regenerated the geometry). Raises
    ``RuntimeError`` after ``max_attempts``.
    """
    grid, _ = sample_for_density(family, target_rho, res,
                                 shape_parameter=shape_parameter)
    for attempt in range(max_attempts):
        try:
            C6 = homogenizer(grid, mesh, pairings, res)
            fea_time = float(homogenizer.times[-1]) if homogenizer.times else float("nan")
            return C6.astype(np.float32), grid, fea_time
        except Exception:
            LOG.warning("sample %d (%s, target rho %.3f) FEA failed "
                        "(attempt %d/%d)", index, family, target_rho,
                        attempt + 1, max_attempts, exc_info=True)
            if attempt + 1 == max_attempts:
                break
            jitter = float(np.random.default_rng(
                seed + index + attempt).uniform(-2e-2, 2e-2))
            target = min(0.999, max(1e-3, target_rho + jitter))
            grid, _ = sample_for_density(family, target, res,
                                         shape_parameter=shape_parameter)
    raise RuntimeError(
        f"sample {index} ({family}) permanently failed after {max_attempts} attempts"
    )


def _process_cell(i: int, s: dict) -> dict:
    """Homogenize one cell inside a worker; retries with jitter on failure.

    Returns ``{"C6", "fea_time_s", "voxels"}`` where ``voxels`` is the FINAL
    grid actually solved (a retried jitter target may have regenerated the
    geometry).  Raises after ``max_attempts`` per-cell retries.
    """
    opts = _WKOPTS
    C6, grid, t = _homogenize_with_retry(
        s["family"], float(s["target_density"]), opts["res"],
        opts["homogenizer"], opts["mesh"], opts["pairings"],
        opts["max_attempts"], opts["seed"], i,
        float(s.get("shape_parameter", 1.0)))
    return {"C6": C6, "fea_time_s": t, "voxels": grid}


def _assemble_record(i: int, s: dict, C6: np.ndarray, fea_time_s: float,
                     voxdir: Path, grid: np.ndarray) -> dict:
    """Build the record dict stored in the checkpoint (shared by both paths)."""
    ep = effective_properties(C6)
    rho = float(grid.sum()) / grid.size
    vox_path = voxdir / f"vox_{i:05d}.npy"
    np.save(vox_path, grid)
    properties = {"E": float(ep.E), "relative_density": rho, "nu": float(ep.nu)}
    rec = dict(s)
    rec["voxels"] = grid
    rec.update(
        C6=C6.astype(np.float32),
        achieved_density=rho,
        E_eff=float(ep.E),
        nu_eff=float(ep.nu),
        fea_time_s=float(fea_time_s),
        voxel_path=vox_path.name,  # relative: checkpoints stay portable across clones
        sample_index=i,
        cond=np.array([properties[k] for k in CONDITIONING_ORDER], dtype=np.float32),
    )
    return rec


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #

def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    # Sort C6_<k> columns numerically: parquet round-trips need not preserve
    # column order, and a lexical sort would mis-map C6_10..C6_35.
    c6_cols = sorted((c for c in df.columns if c.startswith("C6_")),
                     key=lambda c: int(c.rsplit("_", 1)[1]))
    # Legacy checkpoints wrote cond_rho; canonical ones use the locked names.
    legacy = {"E": "cond_E", "relative_density": "cond_rho", "nu": "cond_nu"}
    def _cond_col(name: str) -> str:
        col = f"cond_{name}"
        return col if col in df.columns else legacy[name]
    def _row_cond(row) -> np.ndarray:
        return np.array([float(row[_cond_col(name)]) for name in CONDITIONING_ORDER],
                        dtype=np.float32)
    out = []
    for _, row in df.iterrows():
        C6 = np.zeros((6, 6), dtype=np.float32)
        if c6_cols:
            C6.flat[:] = row[c6_cols].to_numpy(dtype=np.float32)
        # None for legacy checkpoints (which predate the column); the resume
        # path in main() backfills it from the voxel file name before saving.
        sample_index = int(row["sample_index"]) if "sample_index" in df.columns else None
        out.append({
            "family": row["family"],
            "achieved_density": float(row["achieved_density"]),
            "target_density": float(row["target_density"]),
            "shape_parameter": float(row.get("shape_parameter", 1.0)),
            "E_eff": float(row["E_eff"]) if "E_eff" in df.columns else float(row.get("cond_E", np.nan)),
            "nu_eff": float(row["nu_eff"]) if "nu_eff" in df.columns else float(row.get("cond_nu", np.nan)),
            "voxel_path": Path(str(row["voxel_path"])),
            "cond": _row_cond(row),
            "fea_time_s": float(row.get("fea_time_s", np.nan)),
            "sample_index": sample_index,
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
            "shape_parameter": float(r.get("shape_parameter", 1.0)),
            "E_eff": float(r.get("E_eff", c[CONDITIONING_ORDER.index("E")])),
            "nu_eff": float(r.get("nu_eff", c[CONDITIONING_ORDER.index("nu")])),
            "voxel_path": str(r.get("voxel_path", "")),
            "fea_time_s": float(r.get("fea_time_s", np.nan)),
            "sample_index": int(r.get("sample_index", -1)),
        }
        for k, name in enumerate(CONDITIONING_ORDER):
            row[f"cond_{name}"] = float(c[k])
        C6 = r.get("C6")
        if C6 is not None:
            for k in range(36):
                row[f"C6_{k}"] = float(C6.flat[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _run_parallel(records: list[dict], pending: list[tuple[int, dict]],
                  n_workers: int, max_attempts: int, seed: int, res: int,
                  base_E: float, base_nu: float, void_scale: float,
                  ckpt_file: Path, checkpoint_every: int, voxdir: Path,
                  total: int, t0: float) -> tuple[list[dict], int]:
    """Homogenize many cells across independent dolfinx worker processes."""
    n_workers = max(1, min(n_workers, os.cpu_count() or 1))
    omp_threads = max(1, (os.cpu_count() or 1) // n_workers)
    opts = {
        "seed": seed,
        "res": res,
        "base_E": base_E,
        "base_nu": base_nu,
        "void_scale": void_scale,
        "max_attempts": max_attempts,
        "mock_fea": False,
        "omp_threads": omp_threads,
    }
    failed = 0
    n_completed = 0
    LOG.info("running %d cells on %d worker processes (%d already in checkpoint)",
             len(pending), n_workers, len(records))
    with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(opts,),
            mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(_process_cell, i, s): (i, s) for i, s in pending}
        for fut in as_completed(futures):
            i, s = futures[fut]
            try:
                out = fut.result()
            except Exception:
                failed += 1
                LOG.exception("sample %d (%s) failed", i, s["family"])
                continue
            s["voxels"] = out["voxels"]  # final grid (may be a jitter retry)
            try:
                rec = _assemble_record(i, s, out["C6"], out["fea_time_s"],
                                       voxdir, out["voxels"])
            except ValueError as exc:
                failed += 1
                LOG.error("sample %d (%s) produced an unphysical stiffness (%s); skipping",
                          i, s["family"], exc)
                continue
            records.append(rec)
            n_completed += 1
            if n_completed % checkpoint_every == 0:
                _save_checkpoint(records, ckpt_file)
                LOG.info("checkpoint %d/%d (%.1f%%) at %.1fs",
                         len(records), total,
                         100 * len(records) / total, time.time() - t0)

    _save_checkpoint(records, ckpt_file)
    return records, failed


def generate(config_path: str | Path, mock_fea: bool = False,
             n_workers: int | None = None) -> None:
    config = load_yaml(config_path)
    check_conditioning_order(config)
    res = config["resolution"]
    seed = config["seed"]
    set_seed(seed)
    if n_workers is None:
        n_workers = int(config.get("n_workers", 0))

    base_E = float(config["base_material"]["E"])
    base_nu = float(config["base_material"]["nu"])
    void_scale = float(config["homogenization"].get("void_scale", 1.0e-6))
    val_tol = config["homogenization"]["validation_tolerance"]
    homogenizer, validation = _load_homogenizer(
        mock_fea, val_tol, res, base_E, base_nu, void_scale)

    checkpoint_every = config.get("checkpoint_every", 100)
    ckpt_dir = REPO_ROOT / "data" / "checkpoints"
    checkpoint_name = str(config.get("checkpoint_name", Path(config_path).stem))
    ckpt_file = ckpt_dir / f"ckpt_{checkpoint_name}_s{seed}.parquet"

    existing = _load_checkpoint(ckpt_file)
    voxdir = ckpt_dir / "vox"
    voxdir.mkdir(parents=True, exist_ok=True)

    samples = _sample_all(config, np.random.default_rng(seed))
    total = len(samples)
    counts = Counter(s["family"] for s in samples)
    per_family = ", ".join(f"{counts[f]} {f}" for f in config["families"])
    LOG.info("Prepared %d samples (%s)", total, per_family)

    records: list[dict] = []
    done: set[int] = set()
    for s in existing:
        vp = Path(s["voxel_path"])
        # Legacy checkpoints stored absolute paths; resolve relative names
        # against the voxel directory so clones stay resumable.
        voxels = np.load(vp if vp.is_absolute() else voxdir / vp)
        s["voxels"] = voxels
        records.append(s)
        idx = s.get("sample_index")
        if idx is None:  # legacy checkpoint: recover the sample index from the file name
            idx = int(Path(s["voxel_path"]).stem.rsplit("_", 1)[1])
            # Backfill: without this the key stays None and _save_checkpoint
            # (int(None)) / the records.sort below both raise TypeError.
            s["sample_index"] = idx
        done.add(idx)

    max_attempts = int(config.get("fea_retries", 3))
    failed = 0
    t0 = time.time()

    pending = [(i, s) for i, s in enumerate(samples) if i not in done]
    if not pending:
        LOG.info("all %d samples already in checkpoint; skipping FEA", total)
    elif n_workers is not None and n_workers > 1 and not mock_fea:
        records, failed = _run_parallel(
            records, pending, n_workers, max_attempts, seed, res, base_E, base_nu,
            void_scale, ckpt_file, checkpoint_every, voxdir, total, t0)
    else:
        if mock_fea and (n_workers or 1) > 1:
            LOG.warning("n_workers>1 ignored for --mock-fea (serial); "
                        "multiprocessing only applies to the real FEA path")
        mesh, pairings = _build_mesh_and_pairings(res)
        for i, s in pending:
            try:
                C6, grid, fea_time = _homogenize_with_retry(
                    s["family"], float(s["target_density"]), res, homogenizer,
                    mesh, pairings, max_attempts, seed, i,
                    float(s.get("shape_parameter", 1.0)))
                rec = _assemble_record(i, s, C6, fea_time, voxdir, grid)
            except (RuntimeError, ValueError) as exc:
                failed += 1
                LOG.error("sample %d (%s) failed and was skipped: %s",
                          i, s["family"], exc)
                continue
            records.append(rec)

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

    # Resume with failures may leave records out of sample order; restore the
    # canonical 0..N-1 ordering so cell_id == sample index throughout.
    records.sort(key=lambda r: r.get("sample_index", -1))

    if not records:
        raise RuntimeError(
            "no cells were successfully homogenized; refusing to write empty "
            "HDF5 splits and a manifest"
        )
    min_total = int(config.get("min_total_cells", 0))
    if len(records) < min_total:
        raise RuntimeError(
            f"only {len(records)} cells passed FEA; config requires at least "
            f"{min_total} before publishing the dataset"
        )

    splits = _stratify_splits(records, config["splits"],
                              config["relative_density_range"],
                              np.random.default_rng(seed + 1))
    leakages = _check_cross_split_leakage(records, splits)
    if leakages == 0:
        LOG.info("cross-split duplicate check: 0 duplicate pairs across splits")
    else:
        raise RuntimeError(
            f"found {leakages} exact duplicate voxel grids across dataset splits; "
            "refusing to write a leaked train/validation/test dataset"
        )

    target_count = int(config.get("num_test_targets", 0))
    test_conditions = None
    if target_count:
        test_conditions = _unique_test_conditions(
            records, splits.get("test", []), target_count, seed + 2)

    all_conds = np.stack([r["cond"] for r in records])
    train_conds = all_conds[splits["train"]]
    stats = ConditioningStats(
        mean=train_conds.mean(axis=0),
        std=train_conds.std(axis=0) + 1e-8,
    )
    stats.mean = stats.mean.astype(np.float32)
    stats.std = stats.std.astype(np.float32)

    has_stiffness = records[0].get("C6") is not None
    meta_dtype = np.dtype([
        ("family", "S16"),
        ("target_density", "f4"),
        ("shape_parameter", "f4"),
        ("achieved_density", "f4"),
        ("E_eff", "f4"),
        ("nu_eff", "f4"),
        ("valid", "u1"),
    ])
    split_frac = config["splits"]
    for split_name in split_frac:
        idxs = splits[split_name]
        if not idxs:
            LOG.warning("split '%s' empty; skipping HDF5 write", split_name)
            continue
        vox = np.stack([records[i]["voxels"] for i in idxs])
        cond = np.stack([records[i]["cond"] for i in idxs])
        cond_norm = stats.normalize(cond)
        stiff = np.stack([records[i]["C6"] for i in idxs]) if has_stiffness else None
        meta = np.zeros(len(idxs), dtype=meta_dtype)
        for k, i in enumerate(idxs):
            r = records[i]
            meta[k]["family"] = r["family"].encode("utf-8")
            meta[k]["target_density"] = float(r.get("target_density", np.nan))
            meta[k]["shape_parameter"] = float(r.get("shape_parameter", 1.0))
            meta[k]["achieved_density"] = float(r["achieved_density"])
            meta[k]["E_eff"] = float(r.get("E_eff", r["cond"][CONDITIONING_ORDER.index("E")]))
            meta[k]["nu_eff"] = float(r.get("nu_eff", r["cond"][CONDITIONING_ORDER.index("nu")]))
            meta[k]["valid"] = 1

        h5_path = REPO_ROOT / config["hdf5"].format(split=split_name)
        write_conditioned_hdf5(h5_path, vox, cond_norm, stiffness=stiff,
                               metadata=meta, stats=stats, chunk=64)
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
            "shape_parameter": float(r.get("shape_parameter", 1.0)),
            "E_eff": float(r.get("E_eff", r["cond"][CONDITIONING_ORDER.index("E")])),
            "nu_eff": float(r.get("nu_eff", r["cond"][CONDITIONING_ORDER.index("nu")])),
            "fea_time_s": float(r.get("fea_time_s", np.nan)),
            "voxel_path": str(voxdir / Path(r.get("voxel_path", "")))
                           if r.get("voxel_path") else "",
            "hdf5_path": config["hdf5"].format(split=split_name),
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = REPO_ROOT / config["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    LOG.info("manifest written to %s", manifest_path)

    if test_conditions is not None:
        target_path = REPO_ROOT / config["test_targets"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_conditions, columns=CONDITIONING_ORDER).to_csv(
            target_path, index=False)
        LOG.info("wrote %d distinct held-out target vectors to %s",
                 target_count, target_path)

    validity_report = _validity_summary(manifest, records, config)
    _write_data_card(manifest, config, records, failed,
                     void_scale, float(time.time() - t0), validation,
                     validity_report)


def _validity_summary(manifest: pd.DataFrame, records: list[dict],
                      config: dict) -> dict | None:
    """Periodic synthesizability rate over the dataset, for the data card.

    Uses the same filter thresholds as the evaluation pipeline
    (``configs/evaluate.yaml``) so the card cannot drift from the real filter.
    Returns ``None`` if the voxels are unavailable (e.g. a resumed run whose
    arrays are no longer in memory), in which case the card just omits the
    section.
    """
    if not records or "voxels" not in records[0]:
        return None
    try:
        from src.config import load_yaml
        from src.validation import evaluate_validity

        eval_cfg_path = REPO_ROOT / "configs" / "evaluate.yaml"
        filter_cfg = load_yaml(eval_cfg_path) if eval_cfg_path.exists() else {}
        rates: dict[str, dict] = {}
        valid = 0
        for r in records:
            ok, _ = evaluate_validity(np.asarray(r["voxels"], dtype=bool), filter_cfg)
            valid += int(ok)
            fam = r["family"]
            rates.setdefault(fam, {"valid": 0, "total": 0})
            rates[fam]["valid"] += int(ok)
            rates[fam]["total"] += 1
        total = len(records)
        return {
            "rate": valid / total if total else 0.0,
            "valid": valid,
            "total": total,
            "by_family": {
                fam: {"rate": v["valid"] / v["total"] if v["total"] else 0.0,
                      "valid": v["valid"], "total": v["total"]}
                for fam, v in rates.items()
            },
        }
    except Exception:  # card generation must never sink a completed dataset
        LOG.warning("periodic validity summary unavailable", exc_info=True)
        return None


def _write_data_card(manifest: pd.DataFrame, config: dict, records: list[dict],
                     failed: int, void_scale: float, wall_s: float,
                     validation: dict | None = None,
                     validity_report: dict | None = None) -> None:
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
        f"- **unique voxel grids**: {len({hashlib.sha256(np.ascontiguousarray(r['voxels']).tobytes()).digest() for r in records})}",
        f"- **wall time (FEA)**: {wall_s:.1f}s",
        "",
        "## FEA configuration",
        f"- base material: E_s={config['base_material']['E']}, nu_s={config['base_material']['nu']}",
        "- E_eff is in the same units as E_s; with E_s=1 it is the dimensionless ratio E_eff/E_s.",
        f"- void scale: {_f(void_scale)} (E_void = scale * E, nu_void = nu_base)",
        f"- validation tolerance: {config['homogenization']['validation_tolerance']}",
        "",
        f"## Conditioning targets (order {CONDITIONING_ORDER})",
        "| quantity | min | p50 | p95 | max |",
        "|----------|-----|-----|-----|-----|",
        (f"| E_eff | {_f(e_eff.min())} | {_f(np.median(e_eff))} |"
            f" {_f(np.percentile(e_eff, 95))} | {_f(e_eff.max())} |"),
        (f"| relative_density | {_f(rho_raw.min())} | {_f(np.median(rho_raw))} |"
            f" {_f(np.percentile(rho_raw, 95))} | {_f(rho_raw.max())} |"),
        (f"| nu_eff | {_f(nu_eff.min())} | {_f(np.median(nu_eff))} |"
            f" {_f(np.percentile(nu_eff, 95))} | {_f(nu_eff.max())} |"),
        "",
        "## Per-family counts",
    ]
    for family, n in manifest["family"].value_counts().items():
        lines.append(f"- {family}: {n}")
    if len(fea_times):
        lines += [
            "",
            "## FEA timing (s/cell)",
            (f"- mean {fea_times.mean():.3f}, median {np.median(fea_times):.3f},"
            f" p95 {np.percentile(fea_times, 95):.3f}, max {fea_times.max():.3f}"),
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

    # Periodic validity is a property of the *data*, not of one downstream run,
    # so record it here. It is the fraction of cells that pass the
    # synthesizability filter, and it must be computed with periodic
    # connectivity: counting components in the unwrapped box undercounts badly
    # (it read 50.8% on this dataset vs 87.7% correctly measured).
    if validity_report is not None:
        lines += [
            "",
            "## Periodic validity (synthesizability filter)",
            (f"- overall: {validity_report['rate'] * 100:.1f}%"
             f" ({validity_report['valid']}/{validity_report['total']})"),
            "- connectivity is counted **periodically** (components merged across",
            "  wrapped faces); the unwrapped count is far lower and is wrong for",
            "  unit cells.",
        ]
        for family, info in sorted(validity_report["by_family"].items()):
            lines.append(
                f"- {family}: {info['rate'] * 100:.1f}%"
                f" ({info['valid']}/{info['total']})"
            )

    path = REPO_ROOT / "data" / "data_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    LOG.info("data card written to %s", path)


def main() -> None:
    if multiprocessing.current_process().name != "MainProcess":
        # Spawned worker children re-import this module as __main__; never
        # let them re-enter the CLI.
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to dataset.yaml")
    ap.add_argument("--mock-fea", action="store_true",
                    help="Use an UNPHYSICAL placeholder stiffness (pipeline testing only)")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="Number of FEA worker processes (overrides config n_workers)")
    args = ap.parse_args()
    generate(args.config, mock_fea=args.mock_fea, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
