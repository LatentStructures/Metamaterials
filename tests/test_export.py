"""Tests for STL export + watertightness check (ROADMAP 4.6 step 5 / 4.7)."""
import numpy as np
import pandas as pd

from src.export import export_stl as ex


def _sphere(n: int = 16, r: float = 6.0) -> np.ndarray:
    g = np.arange(n)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    c = 0.5 * (n - 1)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    return (d <= r).astype(np.uint8)


def _make_samples(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    vol = _sphere()
    assert vol.min() == 0 and vol.max() == 1 and 0 < vol.mean() < 1
    np.save(samples / "c00.npy", vol)
    np.save(samples / "c01.npy", vol)
    pd.DataFrame([
        {"file": "c00.npy", "target_E": 0.3, "target_rho": vol.mean(),
         "target_nu": 0.3,
         "E_eff": 0.31, "achieved_rho": vol.mean(), "valid": True},
        {"file": "c01.npy", "target_E": 0.9, "target_rho": vol.mean(),
         "target_nu": 0.3,
         "E_eff": 0.2, "achieved_rho": vol.mean(), "valid": True},
    ]).to_csv(samples / "per_sample.csv", index=False)
    return samples


def test_sphere_is_watertight():
    from src.geometry import voxels_to_surface
    m = ex._trimesh_surface(voxels_to_surface(_sphere()))
    assert m.is_watertight


def test_rank_keeps_valid_and_sorts_by_error():
    df = pd.DataFrame({
        "file": ["b", "a"],
        "valid": [True, True],
        "E_eff": [0.2, 0.5],
        "target_E": [0.9, 0.5],
    })
    ranked = ex.rank_candidates(df)
    assert ranked.iloc[0]["file"] == "a"  # smaller E error first


def test_rank_falls_back_without_valid_column():
    df = pd.DataFrame({"file": ["a", "b"],
                       "achieved_rho": [0.9, 0.2], "target_rho": [0.3, 0.3]})
    ranked = ex.rank_candidates(df)
    assert ranked.iloc[0]["file"] == "b"


def test_run_export_writes_stl_and_report(tmp_path):
    samples = _make_samples(tmp_path)
    out = tmp_path / "out"
    report = ex.run_export(samples, out, top_n=1, smooth=True)
    assert report["exported"] == 1
    assert report["watertight"] == 1
    assert report["failed"] == 0
    assert (out / "c00.stl").exists() and (out / "c00.stl").stat().st_size > 0
    assert (out / "export_manifest.csv").exists()
    assert (out / "export_report.json").exists()


def test_run_export_reports_missing_file(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    pd.DataFrame([{"file": "ghost.npy", "valid": True}]
                 ).to_csv(samples / "per_sample.csv", index=False)
    out = tmp_path / "out"
    report = ex.run_export(samples, out, top_n=1)
    assert report["exported"] == 0
    assert report["failed"] == 1