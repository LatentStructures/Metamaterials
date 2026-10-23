"""Visual evidence pack for the "target properties -> verified candidate" loop.

Renders the display artifacts the acceptance narrative needs, reusing what the
pipeline already recorded -- nothing is re-simulated (the FEA numbers shown are
the recorded ones from ``src.evaluate``):

* ``reference_vs_generated.png``  -- one cell per target: the dataset reference
  cell (the nearest [E, rho, nu] match in the held-out split, i.e. the cell the
  target was drawn from) beside the generated 32^3 grid, captioned with
  requested vs FEA-verified properties and the structural-validity verdict.
* ``selected_pair.png``           -- the selected candidate at large: three
  orthographic views of both grids plus a full annotation block (requested /
  FEA-verified / errors / validity / reference distance / STL watertightness).
* ``selected_candidate_3d.png``   -- marching-cubes 3D render of that candidate.
* ``export_montage.png``          -- 3D montage of the exported top-N
  candidates; a reproducible stand-in for the hand-made ``montage.png`` files
  (whose generating script was cleaned up as scratch).
* ``evidence_manifest.json``      -- machine-readable index mapping every
  displayed claim to the artifact it came from.

Labeling honesty: a target vector does not define a unique voxel grid, so the
reference panel always says "nearest dataset cell" and prints its match
distance in dataset-std units. An off-manifold target therefore reads as
``nearest, d > tol`` instead of pretending an exact ground truth exists.

Inputs are the outputs of ``src.sample``, ``src.evaluate`` and
``src.export.export_stl``; the FEA numbers shown are the recorded ones.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONDITIONING_ORDER, REPO_ROOT
from src.data.dataset import MetamaterialDataset
from src.export.export_stl import rank_candidates
from src.geometry import voxels_to_surface
from src.validation import safe_sample_name

LOG = logging.getLogger(__name__)

DEFAULT_SAMPLES_DIR = "outputs/acceptance_30k/samples_live"
DEFAULT_EVAL_DIR = "outputs/acceptance_30k/evaluate_live"
DEFAULT_DATASET = "data/dataset_test.h5"
DEFAULT_EXPORT_DIR = "outputs/acceptance_30k/export_final"
DEFAULT_OUT_DIR = "outputs/acceptance_30k/evidence"
DEFAULT_MAX_MATCH_DIST = 1.0e-2
DEFAULT_COLS = 5

# per_sample.csv column names for the requested / FEA-verified values, keyed by
# the locked conditioning names (same mapping src.evaluate uses for its plots).
TGT_COLS = {"E": "target_E", "relative_density": "target_rho", "nu": "target_nu"}
ACH_COLS = {"E": "E_eff", "relative_density": "achieved_rho", "nu": "nu_eff"}
SHORT = {"E": "E", "relative_density": "rho", "nu": "nu"}
EVAL_COLUMNS = {
    "file", "target_E", "target_rho", "target_nu", "achieved_rho",
    "E_eff", "nu_eff", "valid", "reason",
}

REF_RGB = (0.55, 0.66, 0.80)   # dataset reference cell
GEN_RGB = (1.00, 0.55, 0.15)   # generated grid (matches the orange montages)
MISSING_RGB = (0.30, 0.10, 0.10)


def _plt():
    """Agg-backed pyplot (headless-safe; idempotent if already imported)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rel(path: Path | str | None) -> str | None:
    """Repo-relative path when possible, so the manifest survives moves."""
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _finite_or_none(val) -> float | None:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int_or_none(val) -> int | None:
    v = _finite_or_none(val)
    return None if v is None else int(v)


def _bool_or_none(val) -> bool | None:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes"}
    return bool(val)


def _fmt(val) -> str:
    """3-decimal value, or an explicit FAILED marker for NaN/None."""
    v = _finite_or_none(val)
    return "FAILED" if v is None else f"{v:.3f}"


def load_eval(eval_dir: Path) -> pd.DataFrame:
    """FEA-verified per-sample table (the evidence pack refuses to run without it)."""
    path = Path(eval_dir) / "per_sample.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; the evidence pack reuses the FEA-verified "
            "evaluation -- run `python -m src.evaluate` first"
        )
    df = pd.read_csv(path)
    missing = EVAL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path} has no rows")
    # read_csv yields bool, object (with NaN) or string dtypes depending on the
    # writer; parsing explicitly avoids the classic astype(bool) trap where the
    # *string* "False" would count as valid.
    valid = df["valid"]
    if valid.dtype == object:
        valid = valid.map(_bool_or_none)
    df["valid"] = valid.fillna(False).astype(bool)
    df["file"] = [safe_sample_name(str(f), "per_sample.csv") for f in df["file"]]
    return df


def nearest_reference(raw_targets: np.ndarray,
                      dataset: MetamaterialDataset) -> tuple[np.ndarray, np.ndarray]:
    """Nearest conditioning cell per target: (index, distance) pairs.

    Distance is the L2 norm in dataset-std units. Targets are normalized with
    the split's own conditioning stats before comparison, so it is the same
    metric the off-manifold control reports (>= 0.012 there; ~0 for the
    on-manifold acceptance set).
    """
    raw = np.asarray(raw_targets, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(CONDITIONING_ORDER):
        raise ValueError(
            f"targets must have shape (N, {len(CONDITIONING_ORDER)}) in order "
            f"{CONDITIONING_ORDER}; got {raw.shape}"
        )
    if len(dataset) == 0:
        raise ValueError(f"{dataset.path}: reference split is empty")
    norm = dataset.conditioning_mean_std().normalize(raw)
    d = np.linalg.norm(norm[:, None, :] - dataset.cond[None, :, :], axis=-1)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(d)), idx]


def _proj(vol):
    """Max-intensity projection along axis 0 (None-safe for missing grids)."""
    return None if vol is None else (np.asarray(vol) > 0).max(axis=0)


def _pair_rgb(ref, gen, gap: int = 2) -> np.ndarray:
    """RGB composite of two 3D grids: [reference | gap | generated]."""
    if ref is None and gen is None:
        raise ValueError("nothing to render: reference and generated grid both missing")
    ref, gen = _proj(ref), _proj(gen)
    if ref is not None and gen is not None and ref.shape != gen.shape:
        raise ValueError(
            f"reference {ref.shape} and generated {gen.shape} projections "
            "differ; cannot pair them in one panel"
        )
    d = int(max(np.shape(g)[0] for g in (ref, gen) if g is not None))
    img = np.zeros((d, 2 * d + gap, 3), dtype=float)
    left, right = img[:, :d], img[:, d + gap:]
    if ref is None:
        left[...] = MISSING_RGB
    else:
        ref = np.asarray(ref)
        left[: ref.shape[0], : ref.shape[1]][ref > 0] = REF_RGB
    if gen is None:
        right[...] = MISSING_RGB
    else:
        gen = np.asarray(gen)
        right[: gen.shape[0], : gen.shape[1]][gen > 0] = GEN_RGB
    return img


def _view_rgb(view: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    img = np.zeros((*np.asarray(view).shape, 3), dtype=float)
    img[np.asarray(view)] = color
    return img


def _views(vol: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Three orthographic max-intensity projections of one grid."""
    v = np.asarray(vol) > 0
    return [(v.max(axis=k), f"max || {'xyz'[k]}") for k in range(3)]


def _caption(row: pd.Series, dist: float, exact: bool) -> str:
    parts = [
        f"{SHORT[name]} {_fmt(row[TGT_COLS[name]])}->{_fmt(row[ACH_COLS[name]])}"
        for name in CONDITIONING_ORDER
    ]
    tail = f"ref d={dist:.3f}" + ("" if exact else " (nearest cell)")
    return "  ".join(parts) + "  |  " + tail


def figure_pairs(df: pd.DataFrame, grids: dict, ref_vox: np.ndarray,
                 ref_dist: np.ndarray, max_match_dist: float,
                 cols: int = int(DEFAULT_COLS)):
    """Contact sheet: reference cell vs generated grid, one cell per target."""
    plt = _plt()
    n = len(df)
    if not (len(ref_vox) == len(ref_dist) == n):
        raise ValueError(
            f"row/reference count mismatch: {n} rows, {len(ref_vox)} reference "
            f"grids, {len(ref_dist)} distances"
        )
    n_cols = max(1, min(int(cols), n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.7 * n_cols, 2.15 * n_rows), squeeze=False)
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i // n_cols][i % n_cols]
        name = str(row["file"])
        dist = float(ref_dist[i])
        exact = dist <= max_match_dist
        ax.imshow(_pair_rgb(ref_vox[i], grids.get(name)))
        verdict = "PASS" if row["valid"] else f"FAIL {row['reason']}"
        ax.set_title(f"{Path(name).stem}  {verdict}", fontsize=8)
        ax.set_xlabel(_caption(row, dist, exact), fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(
        f"{n} held-out targets: reference cell vs generated grid "
        "(left blue = dataset reference, right orange = generated; "
        "caption: requested -> FEA-verified, ref d = match distance in "
        "dataset-std units)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _selected_summary(row: pd.Series, ref_index: int, dist: float,
                      exact: bool, stl: dict | None) -> dict:
    requested = {c: _finite_or_none(row[TGT_COLS[c]]) for c in CONDITIONING_ORDER}
    achieved = {c: _finite_or_none(row[ACH_COLS[c]]) for c in CONDITIONING_ORDER}
    rel: dict[str, float | None] = {}
    for c in CONDITIONING_ORDER:
        t = _finite_or_none(row[TGT_COLS[c]])
        a = achieved[c]
        rel[c] = None if (t is None or a is None) else abs(a - t) / max(abs(t), 1e-9)
    return {
        "file": str(row["file"]),
        "requested": requested,
        "fea_verified": achieved,
        "abs_rel_err": rel,
        "valid": bool(row["valid"]),
        "reason": str(row["reason"]),
        "reference_cell": {"index": int(ref_index), "distance": float(dist),
                           "exact": bool(exact)},
        "stl": stl,
    }


def _summary_lines(summary: dict) -> list[str]:
    tgt = summary["requested"]
    ach = summary["fea_verified"]
    rel = summary["abs_rel_err"]
    req = "  ".join(f"{SHORT[c]} {_fmt(tgt[c])}" for c in CONDITIONING_ORDER)
    fea = "  ".join(
        f"{SHORT[c]} {_fmt(ach[c])}"
        + (f" ({100 * rel[c]:.1f}%)" if rel[c] is not None else " (FAILED)")
        for c in CONDITIONING_ORDER
    )
    ref = summary["reference_cell"]
    ref_txt = (f"split idx {ref['index']}, d={ref['distance']:.4f} "
               f"({'exact dataset cell' if ref['exact'] else 'nearest cell, OFF-MANIFOLD'})")
    stl = summary["stl"]
    if not stl or not stl.get("stl"):
        stl_txt = "not in export manifest"
    else:
        wt = stl.get("is_watertight")
        stl_txt = (f"{Path(str(stl['stl'])).name}  watertight="
                   f"{'unknown' if wt is None else wt}  euler={stl.get('euler_characteristic')}")
    verdict = ("PASS" if summary["valid"] else f"FAIL ({summary['reason']})")
    return [
        f"candidate        {summary['file']}",
        f"requested        {req}",
        f"FEA-verified     {fea}",
        f"structural check {verdict}",
        f"reference cell   {ref_txt}",
        f"STL export       {stl_txt}",
    ]


def figure_selected(ref_vol: np.ndarray, gen_vol: np.ndarray, summary: dict):
    """Hero panel: three views of reference + generated, annotation block right."""
    plt = _plt()
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.0, 1.0, 1.5), wspace=0.05)
    rows = (
        (ref_vol, "dataset reference cell (nearest conditioning match)", REF_RGB),
        (gen_vol, "generated grid (conditional DDIM)", GEN_RGB),
    )
    for r, (vol, label, color) in enumerate(rows):
        for c, (view, vlabel) in enumerate(_views(vol)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(_view_rgb(view, color))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel(vlabel, fontsize=7)
    ax_txt = fig.add_subplot(gs[:, 3])
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(_summary_lines(summary)),
                family="monospace", fontsize=10, va="top", transform=ax_txt.transAxes)
    fig.suptitle(f"requested vs FEA-verified properties -- "
                 f"{Path(summary['file']).stem}", fontsize=12)
    return fig


def _add_surface(ax, vol: np.ndarray, color: tuple[float, float, float]) -> None:
    """Marching-cubes surface of one grid onto a 3D axes (unit-cell box)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    surf = voxels_to_surface(np.asarray(vol) > 0, smooth=True, tiles=1)
    tris = surf.points[np.asarray(surf.cells_dict["triangle"])]
    ax.add_collection3d(Poly3DCollection(tris, facecolor=color,
                                         edgecolor=(0.0, 0.0, 0.0, 0.06),
                                         linewidths=0.15))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=38)
    ax.set_axis_off()


def figure_surface(vol: np.ndarray, summary: dict, color=GEN_RGB):
    """Single-candidate 3D render with the annotation block underneath."""
    plt = _plt()
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(projection="3d")
    _add_surface(ax, vol, color)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.13)
    fig.text(0.02, 0.015, "\n".join(_summary_lines(summary)),
             family="monospace", fontsize=8.5, va="bottom")
    fig.suptitle(f"selected candidate (marching-cubes surface) -- "
                 f"{Path(summary['file']).stem}", fontsize=12)
    return fig


def figure_montage(entries: list[tuple[str, np.ndarray, bool | None]]):
    """3D render of the exported candidates in a row (reproducible montage)."""
    plt = _plt()
    n = len(entries)
    fig = plt.figure(figsize=(3.6 * n, 4.4))
    for i, (name, vol, watertight) in enumerate(entries):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        _add_surface(ax, vol, GEN_RGB)
        wt = "unknown" if watertight is None else str(watertight)
        ax.set_title(f"{Path(name).stem}\nwatertight: {wt}", fontsize=9)
    fig.suptitle("exported top-N candidates (marching-cubes surfaces; counts "
                 "in export_report.json)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _load_export(export_dir: Path | None) -> dict[str, dict]:
    """``file`` -> watertightness record from the STL export manifest."""
    if export_dir is None:
        return {}
    path = Path(export_dir) / "export_manifest.csv"
    if not path.is_file():
        LOG.warning("no export manifest at %s; STL status omitted", path)
        return {}
    df = pd.read_csv(path)
    if "exported" in df.columns:
        keep = df["exported"].fillna(False).map(_bool_or_none).astype(bool)
        df = df[keep]
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        stl_val = row.get("stl")
        # pd.notna guards the NaN cell: str(nan) == "nan" is truthy and would
        # resolve to a bogus relative path (same trap as export_stl's source_dir).
        stl_path = _rel(str(stl_val)) if pd.notna(stl_val) and str(stl_val) else None
        out[str(row["file"])] = {
            "stl": stl_path,
            "is_watertight": _bool_or_none(row.get("is_watertight")),
            "euler_characteristic": _int_or_none(row.get("euler_characteristic")),
            "vertices": _int_or_none(row.get("vertices")),
            "faces": _int_or_none(row.get("faces")),
        }
    return out


def _maybe_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_selected(df: pd.DataFrame, selected: str) -> str:
    names = list(df["file"])
    if selected in names:
        return selected
    stem = Path(selected).stem
    hits = [n for n in names if Path(n).stem == stem]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(
        f"--selected {selected!r} matches none of the {len(names)} evaluated samples"
    )


def run_evidence(samples_dir: str | Path, eval_dir: str | Path,
                 dataset_path: str | Path, out_dir: str | Path,
                 export_dir: str | Path | None = None,
                 selected: str | None = None,
                 max_match_dist: float = float(DEFAULT_MAX_MATCH_DIST),
                 cols: int = int(DEFAULT_COLS)) -> dict:
    """Build the evidence pack; returns the manifest dict written to disk."""
    samples_dir, eval_dir = Path(samples_dir), Path(eval_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval(eval_dir)
    dataset = MetamaterialDataset(dataset_path)
    targets = df[[TGT_COLS[c] for c in CONDITIONING_ORDER]].to_numpy(np.float32)
    ref_idx, ref_dist = nearest_reference(targets, dataset)
    ref_vox = dataset.voxels[ref_idx]

    grids: dict[str, np.ndarray | None] = {}
    missing: list[str] = []
    for name in df["file"]:
        path = samples_dir / name
        if path.is_file():
            grids[name] = np.load(path)
        else:
            grids[name] = None
            missing.append(name)
            LOG.warning("sample grid %s missing from %s", name, samples_dir)

    export_rows = _load_export(Path(export_dir) if export_dir else None)
    eval_report = _maybe_json(eval_dir / "report.json")
    export_report = (_maybe_json(Path(export_dir) / "export_report.json")
                     if export_dir else None)

    if selected is None:
        ranked = rank_candidates(df)
        if ranked.empty:
            raise SystemExit(
                "no validity-passing candidate to feature; pass --selected <file>"
            )
        sel_file = str(ranked.iloc[0]["file"])
    else:
        sel_file = _resolve_selected(df, str(selected))
    sel_pos = int(np.flatnonzero(df["file"].to_numpy() == sel_file)[0])
    sel_row = df.iloc[sel_pos]
    if grids.get(sel_file) is None:
        raise SystemExit(f"selected grid {sel_file} not found in {samples_dir}")
    exact = bool(ref_dist[sel_pos] <= max_match_dist)
    summary = _selected_summary(sel_row, int(ref_idx[sel_pos]),
                                float(ref_dist[sel_pos]), exact,
                                export_rows.get(sel_file))

    plt = _plt()
    fig = figure_pairs(df, grids, ref_vox, ref_dist, max_match_dist=max_match_dist,
                       cols=cols)
    fig.savefig(out_dir / "reference_vs_generated.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    ref_grid = ref_vox[sel_pos]
    fig = figure_selected(ref_grid, grids[sel_file], summary)
    fig.savefig(out_dir / "selected_pair.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig = figure_surface(grids[sel_file], summary)
    fig.savefig(out_dir / "selected_candidate_3d.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    images: dict[str, str | None] = {
        "reference_vs_generated": "reference_vs_generated.png",
        "selected_pair": "selected_pair.png",
        "selected_candidate_3d": "selected_candidate_3d.png",
        "export_montage": None,
    }
    montage_entries: list[tuple[str, np.ndarray, bool | None]] = []
    for fname, record in export_rows.items():
        vol = grids.get(fname)
        if vol is not None:
            montage_entries.append(
                (fname, vol, record.get("is_watertight")))
        else:
            LOG.warning("exported candidate %s has no resolvable grid", fname)
    if montage_entries:
        fig = figure_montage(montage_entries)
        fig.savefig(out_dir / "export_montage.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        images["export_montage"] = "export_montage.png"
    else:
        LOG.warning("no exported candidates with resolvable grids; montage skipped")

    n = len(df)
    n_valid = int(df["valid"].sum())
    manifest = {
        "generated_by": "python -m src.evidence",
        "inputs": {
            "samples_dir": _rel(samples_dir),
            "eval_dir": _rel(eval_dir),
            "dataset": _rel(dataset_path),
            "export_dir": _rel(export_dir) if export_dir else None,
        },
        "n_targets": n,
        "validity": {
            "valid": n_valid,
            "rate": n_valid / n,
            "source": _rel(eval_dir / "per_sample.csv"),
            "report": _rel(eval_dir / "report.json"),
        },
        "fea_failures": None if eval_report is None else eval_report.get("fea_failures"),
        "reference_match": {
            "dataset": _rel(dataset_path),
            "n_exact": int((ref_dist <= max_match_dist).sum()),
            "max_match_dist": float(max_match_dist),
            "mean_distance": float(ref_dist.mean()),
            "max_distance": float(ref_dist.max()),
        },
        "selected": summary,
        "stl_export": export_report,
        "missing_grids": missing,
        "images": {k: (str(v) if v else None) for k, v in images.items()},
    }
    with (out_dir / "evidence_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)

    LOG.info("evidence pack: %d images + manifest in %s (selected %s)",
             sum(1 for v in images.values() if v), out_dir, sel_file)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", default=str(REPO_ROOT / DEFAULT_SAMPLES_DIR),
                    help="directory with the generated *.npy grids")
    ap.add_argument("--eval-dir", default=str(REPO_ROOT / DEFAULT_EVAL_DIR),
                    help="FEA-verified evaluation dir (per_sample.csv, report.json)")
    ap.add_argument("--dataset", default=str(REPO_ROOT / DEFAULT_DATASET),
                    help="split HDF5 holding the reference cells (default: test split)")
    ap.add_argument("--export-dir", default=str(REPO_ROOT / DEFAULT_EXPORT_DIR),
                    help="STL export dir (export_manifest.csv) for watertight status")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / DEFAULT_OUT_DIR))
    ap.add_argument("--selected", default=None,
                    help="sample file to feature; default: top-ranked "
                         "validity-passing candidate (same ranking as export_stl)")
    ap.add_argument("--max-match-dist", type=float, default=DEFAULT_MAX_MATCH_DIST,
                    help="dataset-std distance below which a target counts as an "
                         "exact dataset cell (default %(default)s)")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS,
                    help="contact-sheet columns (default %(default)s)")
    args = ap.parse_args()
    run_evidence(args.samples_dir, args.eval_dir, args.dataset, args.out_dir,
                 export_dir=args.export_dir, selected=args.selected,
                 max_match_dist=args.max_match_dist, cols=args.cols)


if __name__ == "__main__":
    main()
