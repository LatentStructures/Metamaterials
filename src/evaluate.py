"""FEA-verified evaluation + synthesizability filter (ROADMAP 4.6).

Consumes the outputs of ``sample.py`` (``*.npy`` voxel grids plus
``samples_manifest.csv`` carrying the requested ``E, relative_density, nu``
targets), re-homogenizes every generated grid with the **real** Phase 1 FEA
solver, applies the synthesizability filter, and reports property
reconstruction error on the validity-passing samples.

Synthesizability filter (owned by ``src.validation``, ROADMAP 4.6, step 4):

* connectivity -- a single connected solid component, evaluated under
  **periodic (wrap-around) boundaries** because a unit cell is tessellated:
  solid leaving one face re-enters the opposite one. Checking the unwrapped box
  would count one continuous network as several islands (it scored the real
  gyroid cells at 9%), so the wrap is part of the check,
* minimum wall thickness -- morphological opening (also periodic) rejects
  1-voxel-thin walls, without penalising members merely clipped by a face,
* overhang angle estimate -- approximate fraction of voxels lacking downward
  support (full support-structure analysis is out of baseline scope).

Define ``filter_generated`` in ``src.validation`` to substitute a custom filter.

``--no-filter`` reports on every sample regardless of validity (the report is
labelled accordingly).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONDITIONING_ORDER, REPO_ROOT, load_yaml
from src.fea.property_extraction import effective_properties
from src.validation import evaluate_validity, safe_sample_name

LOG = logging.getLogger(__name__)

COND_NAMES = list(CONDITIONING_ORDER)


# --------------------------------------------------------------------------- #
# Synthesizability filter (ROADMAP 4.6, step 4)
# --------------------------------------------------------------------------- #
# The connectivity / wall-thickness / overhang checks live in ``src.validation``
# (imported above).

def _plugged_filter() -> object | None:
    """Optional external filter; returns ``None`` when absent."""
    try:
        from src.validation import filter_generated  # type: ignore
    except ImportError:
        return None
    return filter_generated


# --------------------------------------------------------------------------- #
# Metrics (ROADMAP 4.6, step 3)
# --------------------------------------------------------------------------- #

def reconstruction_metrics(pred: np.ndarray, target: np.ndarray,
                           names: list[str]) -> dict:
    """Per-target-kind rel. error / MAE / RMSE / R2/pred-vs-target and bias."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != len(names):
        raise ValueError(
            f"pred/target must share shape (N, {len(names)}); "
            f"got {pred.shape} and {target.shape}"
        )
    denom = np.maximum(1e-9, np.abs(target))
    rel = np.abs(pred - target) / denom
    out: dict = {}
    for k, name in enumerate(names):
        ss_res = float(((pred[:, k] - target[:, k]) ** 2).sum())
        ss_tot = float(((target[:, k] - target[:, k].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out[name] = {
            "mean_abs_rel_err": float(rel[:, k].mean()),
            "median_abs_rel_err": float(np.median(rel[:, k])),
            "p95_abs_rel_err": float(np.percentile(rel[:, k], 95)),
            "mae": float(np.mean(np.abs(pred[:, k] - target[:, k]))),
            "rmse": float(np.sqrt(np.mean((pred[:, k] - target[:, k]) ** 2))),
            # R² is undefined when the target has no variance (including a
            # one-sample smoke evaluation). Use JSON null rather than NaN.
            "r2": float(r2) if np.isfinite(r2) else None,
            "bias": float(np.mean(pred[:, k] - target[:, k])),
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

_EVAL_WORKER: dict = {}


def _init_fea_worker(opts: dict) -> None:
    """Per-process FEA setup: one locked Homogenizer + cached res mesh.

    Caps BLAS/PETSc threads so N workers never multiply the core count.
    """
    threads = max(1, int(opts.get("omp_threads", 1)))
    # NOTE: spawned children import this module (and numpy) before the
    # initializer runs, so these vars only bind to libraries loaded *after*
    # this point -- dolfinx/PETSc/scipy, imported just below (the heavy
    # consumers); already-loaded OpenBLAS in the child is unaffected.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    from src.fea import set_base_material, set_void_scale
    from src.fea.homogenization import Homogenizer
    from src.geometry import periodic_pairing, voxels_to_tetra

    set_base_material(float(opts["E"]), float(opts["nu"]))
    set_void_scale(float(opts["void_scale"]))
    res = int(opts["res"])
    _EVAL_WORKER["homogenizer"] = Homogenizer(res)
    dummy = np.zeros((res, res, res), dtype=np.uint8)
    _EVAL_WORKER["mesh"] = voxels_to_tetra(dummy)
    _EVAL_WORKER["pairings"] = periodic_pairing(_EVAL_WORKER["mesh"], res)


def _worker_homogenize(voxels: np.ndarray) -> np.ndarray:
    """Run one FEA solve inside a worker (module-level for pickling)."""
    h = _EVAL_WORKER["homogenizer"]
    return h(voxels, _EVAL_WORKER["mesh"], _EVAL_WORKER["pairings"], h.res)


def _load_samples(samples_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(samples_dir / "samples_manifest.csv")
    required = {"file", "target_E", "target_rho", "target_nu"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f"--samples-dir manifest must contain {required}; "
            f"got {set(manifest.columns)}"
        )
    return manifest


def run_evaluate(cfg: dict, samples_dir: Path, out_dir: Path,
                 no_filter: bool = False, fea_fn=None) -> dict:
    """Evaluate generated samples; returns the report dict and writes artifacts."""
    manifest = _load_samples(samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_E = float(cfg["base_material"]["E"])
    base_nu = float(cfg["base_material"]["nu"])
    void_scale = float(cfg.get("void_scale", 1.0e-6))
    res_expected = int(cfg.get("resolution", 32))

    # Load every sample up front; grids whose resolution differs from the
    # locked config are excluded with a distinct reason (not an FEA failure).
    jobs: list[tuple[int, object, np.ndarray, bool]] = []
    for idx, (_, row) in enumerate(manifest.iterrows()):
        fname = safe_sample_name(str(row["file"]), "--samples-dir manifest")
        fpath = samples_dir / fname
        if not fpath.exists():
            LOG.warning("missing sample %s", fpath)
            continue
        voxels = np.load(fpath)
        # Check every axis, not just the first: a non-cubic (32, 16, 16) grid
        # would otherwise pass and only fail downstream inside FEA.
        res_ok = voxels.shape == (res_expected, res_expected, res_expected)
        if not res_ok:
            LOG.error("sample %s has shape %s != config %dx%dx%d; excluded",
                      fname, voxels.shape, res_expected, res_expected, res_expected)
        jobs.append((idx, row, voxels, res_ok))

    pluggable = None if no_filter else _plugged_filter()
    validity: dict[int, tuple[bool, str]] = {}
    for idx, _row, voxels, res_ok in jobs:
        if not res_ok:
            validity[idx] = (False, "resolution_mismatch")
        elif no_filter:
            validity[idx] = (True, "ok")
        elif pluggable is not None:
            passed = bool(pluggable(voxels))
            validity[idx] = (passed, "plugged" if passed else "plugged_filter")
        else:
            validity[idx] = evaluate_validity(voxels, cfg)

    c6s: dict[int, np.ndarray] = {}
    if fea_fn is not None:
        # Injected stub (tests): evaluate in-process, exactly as before.
        for idx, _row, voxels, res_ok in jobs:
            if not res_ok or not validity[idx][0]:
                continue
            try:
                c6s[idx] = fea_fn(voxels, voxels.shape[0])
            except Exception:
                LOG.exception("FEA failed for %s; excluded from metrics", _row["file"])
    else:
        fea_jobs = [(idx, voxels) for idx, _r, voxels, ok in jobs
                    if ok and validity[idx][0]]
        n_workers = max(1, min(int(cfg.get("n_workers", os.cpu_count() or 1)),
                               os.cpu_count() or 1))
        omp_threads = max(1, (os.cpu_count() or 1) // n_workers)
        opts = {"E": base_E, "nu": base_nu, "void_scale": void_scale,
                "res": res_expected, "omp_threads": omp_threads}
        if fea_jobs:
            LOG.info("re-homogenizing %d samples on %d worker processes",
                     len(fea_jobs), n_workers)
            with ProcessPoolExecutor(
                    max_workers=n_workers, initializer=_init_fea_worker,
                    initargs=(opts,), mp_context=multiprocessing.get_context("spawn")) as ex:
                futures = {ex.submit(_worker_homogenize, voxels): idx
                           for idx, voxels in fea_jobs}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        c6s[idx] = fut.result()
                    except Exception:
                        LOG.exception("FEA failed for sample %d; excluded from metrics", idx)

    rows = []
    fea_fail = 0
    for idx, row, voxels, res_ok in jobs:
        achieved_rho = float(voxels.mean())
        valid, valid_reason = validity[idx]
        if not valid:
            E_eff = nu_eff = float("nan")
            reason = valid_reason
        elif res_ok and idx in c6s:
            try:
                ep = effective_properties(c6s[idx])
                E_eff, nu_eff = float(ep.E), float(ep.nu)
                reason = "ok"
            except Exception:  # any FEA/property-extraction error counts as a failure
                fea_fail += 1
                E_eff = nu_eff = float("nan")
                reason = "fea_failed"
        elif valid and res_ok:
            fea_fail += 1
            E_eff = nu_eff = float("nan")
            reason = "fea_failed"
        else:
            E_eff = nu_eff = float("nan")
            reason = "resolution_mismatch"

        valid = valid and reason == "ok"

        rows.append({
            "file": row["file"],
            "target_E": float(row["target_E"]),
            "target_rho": float(row["target_rho"]),
            "target_nu": float(row["target_nu"]),
            "achieved_rho": achieved_rho,
            "E_eff": E_eff,
            "nu_eff": nu_eff,
            "valid": valid,
            "reason": reason if valid else (valid_reason if reason != "fea_failed" else reason),
        })

    df = pd.DataFrame(rows, columns=[
        "file", "target_E", "target_rho", "target_nu", "achieved_rho",
        "E_eff", "nu_eff", "valid", "reason",
    ])
    df.to_csv(out_dir / "per_sample.csv", index=False)

    total = len(df)
    n_valid = int(df["valid"].fillna(False).astype(bool).sum())
    valid_df = df[df["valid"]]

    report: dict = {
        "no_filter": no_filter,
        "total_samples": total,
        "fea_failures": fea_fail,
        "validity_rate": float(n_valid / total) if total else 0.0,
        "filter_used": "none" if no_filter else ("plugged" if pluggable is not None else "own_core"),
    }
    if len(valid_df):
        pred = valid_df[["E_eff", "achieved_rho", "nu_eff"]].to_numpy()
        target = valid_df[["target_E", "target_rho", "target_nu"]].to_numpy()
        report["reconstruction"] = reconstruction_metrics(pred, target, COND_NAMES)
    else:
        report["reconstruction"] = {}

    with (out_dir / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float, allow_nan=False)

    if len(valid_df):
        _write_plots(valid_df, out_dir)

    LOG.info("evaluation complete: %d/%d valid (%.1f%%), %d FEA failures",
             n_valid, total, 100 * (n_valid / total) if total else 0.0, fea_fail)
    return report


def _write_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Write parity plots and rel-error histograms (values above 1 are capped)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_col = {"E": "E_eff", "relative_density": "achieved_rho", "nu": "nu_eff"}
    tgt_col = {"E": "target_E", "relative_density": "target_rho", "nu": "target_nu"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, COND_NAMES, strict=True):
        pc, tc = pred_col[name], tgt_col[name]
        ax.scatter(df[tc], df[pc], s=6, alpha=0.5)
        lo = min(df[tc].min(), df[pc].min())
        hi = max(df[tc].max(), df[pc].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"target {name}")
        ax.set_ylabel(f"FEA-verified {name}")
        ax.set_title(f"{name} parity")
    fig.tight_layout()
    fig.savefig(out_dir / "parity.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, COND_NAMES, strict=True):
        pc, tc = pred_col[name], tgt_col[name]
        rel = np.abs(df[pc] - df[tc]) / np.maximum(1e-9, np.abs(df[tc]))
        ax.hist(rel.clip(upper=1.0), bins=40)
        ax.set_title(f"{name} |rel err|")
        ax.set_xlabel("abs rel err")
    fig.tight_layout()
    fig.savefig(out_dir / "relerr.png", dpi=130)
    plt.close(fig)


def main() -> None:
    if multiprocessing.current_process().name != "MainProcess":
        # Spawned FEA workers re-import this module as __main__; never let
        # them re-enter the CLI.
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/evaluate.yaml"))
    ap.add_argument("--samples-dir", default=None, help="override config samples_dir")
    ap.add_argument("--out-dir", default=None, help="override config out_dir")
    ap.add_argument("--no-filter", action="store_true",
                    help="report metrics on every sample, valid or not")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    samples_dir = Path(args.samples_dir) if args.samples_dir else REPO_ROOT / cfg["samples_dir"]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["out_dir"]
    run_evaluate(cfg, samples_dir, out_dir, no_filter=args.no_filter)


if __name__ == "__main__":
    main()
