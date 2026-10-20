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


def test_surface_winding_is_not_inverted():
    """Watertightness alone does not catch a sign/winding error: an inside-out
    mesh is still closed. Assert positive volume against the known sphere
    volume for both the smoothed and the binary branch.

    Regression for the surfacing sign convention: (+dist_in, -dist_out) gives a
    geometrically correct but inverted mesh whose signed volume is negated.
    """
    from src.geometry import voxels_to_surface

    vol = _sphere(n=32, r=10.0)
    true_volume = (4.0 / 3.0) * np.pi * 10.0**3
    for smooth in (True, False):
        mesh = voxels_to_surface(vol, smooth=smooth)
        m = ex._trimesh_surface(mesh)
        measured = m.volume * vol.shape[0] ** 3  # mesh is in normalized cell units
        assert measured > 0, f"smooth={smooth}: inverted winding (volume {measured})"
        assert abs(measured - true_volume) / true_volume < 0.05


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


def test_run_export_finds_grids_in_sibling_samples_dir(tmp_path):
    """evaluate.py writes per_sample.csv into its out_dir while the grids stay in
    the sampling dir; export must still resolve the candidates."""
    samples = _make_samples(tmp_path)          # grids + manifest in tmp_path/samples
    report_dir = tmp_path / "evaluate"
    report_dir.mkdir()
    pd.read_csv(samples / "per_sample.csv").to_csv(
        report_dir / "per_sample.csv", index=False
    )
    out = tmp_path / "out"
    report = ex.run_export(report_dir, out, top_n=1, smooth=True)
    assert report["exported"] == 1, report
    assert report["failed"] == 0
    assert (out / "c00.stl").stat().st_size > 0


def test_run_export_reports_missing_file(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    pd.DataFrame([{"file": "ghost.npy", "valid": True}]
                 ).to_csv(samples / "per_sample.csv", index=False)
    out = tmp_path / "out"
    report = ex.run_export(samples, out, top_n=1)
    assert report["exported"] == 0
    assert report["failed"] == 1


def test_run_export_prefers_live_arm_over_ema(tmp_path):
    """Regression: ``samples*`` matches both ``samples`` (EMA) and ``samples_live``.
    A lexicographic glob picked ``samples`` and silently exported the EMA arm,
    which is not the arm the acceptance numbers come from."""
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples_live").mkdir()
    report_dir = tmp_path / "evaluate"
    report_dir.mkdir()
    vol = _sphere()
    np.save(tmp_path / "samples" / "c00.npy", np.zeros_like(vol))       # ema: empty
    np.save(tmp_path / "samples_live" / "c00.npy", vol)                 # live: real
    pd.DataFrame([{"file": "c00.npy", "valid": True}]
                 ).to_csv(report_dir / "per_sample.csv", index=False)
    resolved = ex._resolve_grid(report_dir, "c00.npy")
    assert resolved == tmp_path / "samples_live" / "c00.npy", resolved


def test_tiled_export_is_watertight_and_scales_volume():
    """ROADMAP 4.7 wants watertight STLs. Void padding closes the surface; tiling
    then determines what the artifact *is* -- a 2x2x2 block. Assert both are
    closed and that the tiled block encloses the solid fraction, i.e. the surface
    is a faithful boundary rather than a self-overlapping shell."""
    from src.geometry import voxels_to_surface

    strut = np.zeros((16, 16, 16), np.uint8)
    strut[:, 6:10, 6:10] = 1                      # runs straight through the cell
    solid_fraction = strut.mean()

    single = ex._trimesh_surface(voxels_to_surface(strut, tiles=1))
    tiled = ex._trimesh_surface(voxels_to_surface(strut, tiles=2))
    assert single.is_watertight and tiled.is_watertight
    assert single.volume > 0
    # The mesh volume is expressed as a fraction of the output box, which is
    # solid_fraction whether one cell or a 2x2x2 block is surfaced.
    assert abs(single.volume - solid_fraction) < 0.01
    assert abs(tiled.volume - solid_fraction) < 0.01
