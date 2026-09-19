"""STL export of top-N evaluated candidates with watertightness checks (ROADMAP 4.6 step 5 / 4.7).

Consumes the outputs of ``evaluate.py`` (``per_sample.csv``) or of ``sample.py``
(``samples_manifest.csv``).  It keeps only the validity-passing rows (when the
source is an evaluation report), ranks them by reconstruction error, exports the
top-N voxel grids as smooth marching-cubes STL files, and records a
watertightness/closed-manifold sanity check per file:

``outputs/export/export_manifest.csv``  one row per candidate
``outputs/export/export_report.json``   summary counts (exported / watertight / failed)
``outputs/export/<candidate>.stl``      binary STL per exported candidate

Periodic unit cells are not closed standalone bodies, so some candidates may be
reported non-watertight; the goal here is to *surface* that, not to hide it.
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


def _safe_sample_name(name: str, source: str) -> str:
    """Reject manifest entries that escape ``samples_dir`` (path traversal)."""
    p = Path(name)
    if p.is_absolute() or len(p.parts) != 1 or p.name in ("", ".", ".."):
        raise ValueError(
            f"{source} entry {name!r} must be a bare filename inside the samples directory"
        )
    return str(p)


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
    import trimesh

    faces = surf.cells_dict.get("triangle")
    if faces is None:
        raise ValueError("surface has no triangle cells")
    return trimesh.Trimesh(vertices=np.asarray(surf.points, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=False)


def _watertight_report(mesh) -> dict:
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "euler_characteristic": int(mesh.euler_number),
    }


def run_export(samples_dir: str | Path, out_dir: str | Path, top_n: int = 5,
               smooth: bool = True,
               mesh_builder=_trimesh_surface) -> dict:
    """Export top-N ranked candidates as STL; returns the summary report dict."""
    samples_dir, out_dir = Path(samples_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = rank_candidates(_load_manifest(samples_dir)).head(max(1, top_n))
    rows = []
    for _, row in candidates.iterrows():
        fname = _safe_sample_name(str(row["file"]), "candidate manifest")
        fpath = samples_dir / fname
        name = Path(fname).stem
        entry = {"file": str(row["file"]), "name": name}
        if not fpath.exists():
            entry.update(exported=False, reason="missing")
            rows.append(entry)
            LOG.warning("candidate %s not found", fpath)
            continue
        try:
            surf = voxels_to_surface(np.load(fpath), smooth=smooth)
            mesh = mesh_builder(surf)
            stl_path = out_dir / f"{name}.stl"
            mesh.export(str(stl_path))
            entry.update(exported=True, stl=str(stl_path), reason="ok",
                         **_watertight_report(mesh))
        except Exception as exc:  # noqa: BLE001 -- skip the sample, keep the pipeline alive
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
        "requested": len(candidates),
        "exported": int(exported.sum()),
        "watertight": watertight,
        "failed": int((~exported).sum()),
        "watertightness_expected": (
            "unit cells that intersect their cell faces are inherently open; "
            "non-watertight files are expected for those candidates"
        ),
    }
    with open(out_dir / "export_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    LOG.info("exported %d/%d (watertight %d) to %s",
             report["exported"], report["requested"], report["watertight"], out_dir)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", default=str(REPO_ROOT / "outputs/evaluate"),
                    help="directory with per_sample.csv (evaluate) or samples_manifest.csv (sample)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/export"))
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--no-smooth", action="store_true",
                    help="use the crisp 0.5 level set instead of the smooth field")
    args = ap.parse_args()
    run_export(args.samples_dir, args.out_dir, top_n=args.top_n,
               smooth=not args.no_smooth)


if __name__ == "__main__":
    main()