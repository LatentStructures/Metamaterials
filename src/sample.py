"""Conditional DDIM sampling + export (ROADMAP section 4.5, section 3.5).

Given a trained checkpoint, sample voxel grids for target conditioning vectors,
save them as .npy, compute achieved densities, and optionally export a smooth
closed STL per sample via the smooth marching-cubes surface. Targets are loaded
from a CSV/parquet with columns E, relative_density, nu (un-normalized).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_yaml, REPO_ROOT
from src.data.dataset import read_conditioning_stats
from src.geometry import voxels_to_surface
from src.models.diffusion import build_diffusion
from src.models.unet3d import UNet3D

LOG = logging.getLogger(__name__)


def _parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-config", default=str(REPO_ROOT / "configs/train_baseline.yaml"))
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs/model_baseline.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--targets", required=True, help="csv with columns E, relative_density, nu")
    ap.add_argument("--out-dir", default="outputs/samples")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps (50-100)")
    ap.add_argument("--n-per-target", type=int, default=1)
    ap.add_argument("--export-stl", action="store_true")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse()

    tcfg = load_yaml(args.train_config)
    mcfg = load_yaml(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_h5 = REPO_ROOT / tcfg["data"]["hdf5"]
    stats = read_conditioning_stats(dataset_h5)

    model = UNet3D.from_config(mcfg["unet3d"]).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    diffusion = build_diffusion(mcfg["diffusion"], device)
    LOG.info("loaded checkpoint iter=%s device=%s", ck.get("iter", "?"), device)

    targets = pd.read_csv(args.targets)
    required_cols = {"E", "relative_density", "nu"}
    if not required_cols.issubset(targets.columns):
        raise ValueError(
            f"--targets CSV must contain columns {required_cols}; "
            f"got {set(targets.columns)}"
        )
    conds = targets[["E", "relative_density", "nu"]].to_numpy(dtype=np.float32)
    conds = stats.normalize(conds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []
    model.eval()
    with torch.no_grad():
        for ti, c in enumerate(conds):
            c_b = torch.as_tensor(c[None], dtype=torch.float32, device=device).repeat(args.n_per_target, 1)
            gen = diffusion.sample_ddim(model, c_b, (args.n_per_target, 1, 32, 32, 32),
                                        steps=args.steps, seed=0)
            for si in range(args.n_per_target):
                v = (gen[si, 0].cpu().numpy() > 0.5).astype(np.uint8)
                name = f"sample_t{ti:02d}_s{si:02d}"
                np.save(out_dir / f"{name}.npy", v)
                if args.export_stl:
                    surf = voxels_to_surface(v)
                    surf.write(str(out_dir / f"{name}.stl"))
                    LOG.info("wrote %s.stl", name)
                grid_rows.append({
                    "file": f"{name}.npy",
                    "target_E": float(targets.iloc[ti]["E"]),
                    "target_rho": float(targets.iloc[ti]["relative_density"]),
                    "target_nu": float(targets.iloc[ti]["nu"]),
                    "achieved_rho": float(v.mean()),
                })
    pd.DataFrame(grid_rows).to_csv(out_dir / "samples_manifest.csv", index=False)
    LOG.info("wrote %d samples to %s", len(grid_rows), out_dir)


if __name__ == "__main__":
    main()