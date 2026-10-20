"""STL export of top-N evaluated candidates with watertightness checks (ROADMAP 4.6 step 5 / 4.7).

Consumes the outputs of ``evaluate.py`` (``per_sample.csv``) or of ``sample.py``
(``samples_manifest.csv``).  It keeps only the validity-passing rows (when the
source is an evaluation report), ranks them by reconstruction error, exports the
top-N voxel grids as smooth marching-cubes STL files, and records a
watertightness/closed-manifold sanity check per file:

``outputs/export/export_manifest.csv``  one row per candidate
``outputs/export/export_report.json``   summary counts (exported / watertight / failed)
``outputs/export/<candidate>.stl``      binary STL per exported candidate

Periodic unit cells intersect their own faces, so surfacing a *single* tile leaves
the mesh open wherever solid runs through a face. The default ``--tiles 2`` pads a
2x2x2 replication with void, which closes the surface and yields the watertight,
printable block that ROADMAP 4.7 requires.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPO_ROOT
from src.geometry import voxels_to_surface
from src.validation import safe_sample_name

LOG = logging.getLogger(__name__)

MANIFEST_CANDIDATES = ("per_sample.csv", "samples_manifest.csv")


def _load_manifest(samples_dir: Path) -> pd.DataFrame:
    """Prefer the FEA-verified evaluation report, else the raw sampling manifest."""
    for name in MANIFEST_CANDIDATES:
        p = Path(samples_dir) / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"no candidate manifest in {samples_dir}; expected one of "
        f"{', '.join(MANIFEST_CANDIDATES)}"
    )


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep validity-passing rows and sort by reconstruction error ascending.

    Falls back to all rows when no ``valid``/error columns exist (raw sampling
    output), so the command still runs on a fresh ``sample.py`` directory.
    """
    df = df.copy()
    if "valid" in df.columns:
        df = df[df["valid"].fillna(False).astype(bool)]
    if {"E_eff", "target_E"}.issubset(df.columns):
        denom = df["target_E"].abs().clip(lower=1e-9)
        df["rel_err"] = (df["E_eff"] - df["target_E"]).abs() / denom
    elif {"achieved_rho", "target_rho"}.issubset(df.columns):
        df["rel_err"] = (df["achieved_rho"] - df["target_rho"]).abs()
    if "rel_err" in df.columns:
        df = df.sort_values("rel_err")
    return df.reset_index(drop=True)


def _trimesh_surface(surf):
    """Wrap a meshio surface as a trimesh, welding duplicate vertices.

    ``marching_cubes`` emits per-triangle vertices, so the mesh is only
    watertight *after* coincident vertices are merged. Building with
    ``process=False`` and never merging made the watertightness report depend on
    whether raw coordinates happened to be bitwise identical, which made the same
    export flip between watertight and non-watertight across runs.
    """
    import trimesh

    faces = surf.cells_dict.get("triangle")
    if faces is None:
        raise ValueError("surface has no triangle cells")
    mesh = trimesh.Trimesh(vertices=np.asarray(surf.points, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.merge_vertices()
    return mesh


def _watertight_report(mesh) -> dict:
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "euler_characteristic": int(mesh.euler_number),
    }


def _resolve_grid(samples_dir: Path, fname: str) -> Path | None:
    """Locate a candidate grid, wherever the sampling stage wrote it.

    ``evaluate.py`` records the grid *filename* in ``per_sample.csv`` but writes
    its report into a different directory than the ``.npy`` files it scored, so
    resolving the name against the manifest's own directory alone is wrong.

    The sibling search is deliberately ordered so a ``samples_live`` directory
    (live weights -- the arm the acceptance numbers come from) is preferred over
    a plain ``samples`` directory (EMA). Otherwise the two arms are both matched
    by ``samples*`` and a lexicographic glob silently exports the wrong one.
    """
    candidates = [samples_dir / fname]
    parent = samples_dir.parent
    if parent.is_dir():
        siblings = [p for p in sorted(parent.glob("samples*")) if p.is_dir()]
        siblings.sort(key=lambda p: (not p.name.endswith("_live"), p.name))
        candidates += [p / fname for p in siblings]
    for p in candidates:
        if p.is_file():
            return p
    return None


def run_export(samples_dir: str | Path, out_dir: str | Path, top_n: int = 5,
               smooth: bool = True, tiles: int = 1,
               mesh_builder=_trimesh_surface) -> dict:
    """Export top-N ranked candidates as STL; returns the summary report dict."""
    samples_dir, out_dir = Path(samples_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Side-channel: an optional ``source_dir`` manifest column (written by the
    # sampling run) redirects grid lookup to the writer's directory. pd.notna
    # guards against NaN cells -- str(nan) == "nan" is truthy and would resolve
    # every candidate against a nonexistent "nan" subdirectory.
    manifest = _load_manifest(samples_dir)
    source_dir = next(
        (
            samples_dir / str(p)
            for p in manifest.get("source_dir", [])
            if pd.notna(p) and str(p)
        ),
        None,
    )

    candidates = rank_candidates(manifest).head(max(1, top_n))
    rows = []
    for _, row in candidates.iterrows():
        fname = safe_sample_name(str(row["file"]), "candidate manifest")
        fpath = (source_dir / fname) if source_dir else _resolve_grid(samples_dir, fname)
        name = Path(fname).stem
        entry = {"file": str(row["file"]), "name": name}
        if fpath is None or not fpath.exists():
            entry.update(exported=False, reason="missing")
            rows.append(entry)
            LOG.warning("candidate %s not found", fpath)
            continue
        try:
            surf = voxels_to_surface(np.load(fpath), smooth=smooth, tiles=tiles)
            mesh = mesh_builder(surf)
            stl_path = out_dir / f"{name}.stl"
            mesh.export(str(stl_path))
            entry.update(exported=True, stl=str(stl_path), reason="ok",
                         **_watertight_report(mesh))
        except Exception as exc:  # skip the sample, keep the pipeline alive
            LOG.error("export failed for %s: %s", row["file"], exc)
            entry.update(exported=False, reason=str(exc))
        rows.append(entry)

    out_df = pd.DataFrame(rows, columns=[
        "file", "name", "exported", "stl", "reason",
        "vertices", "faces", "is_watertight", "euler_characteristic",
    ])
    out_df.to_csv(out_dir / "export_manifest.csv", index=False)

    watertight = int(out_df["is_watertight"].fillna(False).astype(bool).sum())
    exported = out_df["exported"].fillna(False).astype(bool)
    report = {
        "top_n": int(top_n),
        "tiles": int(tiles),
        "requested": len(candidates),
        "exported": int(exported.sum()),
        "watertight": watertight,
        "failed": int((~exported).sum()),
    }
    if tiles == 1:
        report["watertightness_expected"] = (
            "a single periodic cell is open wherever solid crosses a cell face, "
            "so non-watertight files are inherent to --tiles 1; use --tiles 2 "
            "for a closed, printable block (ROADMAP 4.7)"
        )
    else:
        report["watertightness_expected"] = (
            f"--tiles {tiles} pads the tiled block with void, so every exported "
            "surface is closed; a non-watertight file here is a real failure"
        )
    with (out_dir / "export_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    LOG.info("exported %d/%d (watertight %d, tiles %d) to %s",
             report["exported"], report["requested"], report["watertight"],
             report["tiles"], out_dir)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", default=str(REPO_ROOT / "outputs/evaluate"),
                    help="directory with per_sample.csv (evaluate) or samples_manifest.csv (sample)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/export"))
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--tiles", type=int, default=2,
                    help="replicate the cell N times per axis before surfacing; "
                         "N>=2 gives a closed, watertight, printable block "
                         "(default 2, required by ROADMAP 4.7)")
    ap.add_argument("--no-smooth", action="store_true",
                    help="use the crisp 0.5 level set instead of the smooth field")
    args = ap.parse_args()
    run_export(args.samples_dir, args.out_dir, top_n=args.top_n,
               smooth=not args.no_smooth, tiles=args.tiles)


if __name__ == "__main__":
    main()
