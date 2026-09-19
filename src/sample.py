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

from src.config import CONDITIONING_ORDER, REPO_ROOT, load_yaml
from src.data.dataset import read_conditioning_stats
from src.geometry import voxels_to_surface
from src.models.diffusion import build_diffusion
from src.models.unet3d import UNet3D

LOG = logging.getLogger(__name__)

# Locked range (ROADMAP 4.3): DDIM inference uses 50-100 steps.
DDIM_MIN_STEPS = 50
DDIM_MAX_STEPS = 100


def resolve_ddim_steps(cli_steps: int | None, mcfg: dict) -> int:
    """Effective DDIM step count: CLI override else ``sampling.steps``, clamped.

    ``--steps`` outside the locked 50-100 range is clamped (not rejected) so a
    typo never silently picks a pathological trajectory length.
    """
    steps = int(cli_steps if cli_steps is not None
                else mcfg.get("sampling", {}).get("steps", DDIM_MIN_STEPS))
    if not DDIM_MIN_STEPS <= steps <= DDIM_MAX_STEPS:
        LOG.warning("DDIM steps %d outside the locked [%d, %d] range; clamped",
                    steps, DDIM_MIN_STEPS, DDIM_MAX_STEPS)
        return max(DDIM_MIN_STEPS, min(steps, DDIM_MAX_STEPS))
    return steps


def _parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-config", default=str(REPO_ROOT / "configs/train_baseline.yaml"))
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs/model_baseline.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--targets", required=True, help="csv with columns E, relative_density, nu")
    ap.add_argument("--out-dir", default="outputs/samples")
    ap.add_argument("--steps", type=int, default=None,
                    help="DDIM steps (locked 50-100); default from model config sampling.steps")
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
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    diffusion = build_diffusion(mcfg["diffusion"], device)
    LOG.info("loaded checkpoint iter=%s device=%s", ck.get("iter", "?"), device)

    steps = resolve_ddim_steps(args.steps, mcfg)

    targets = pd.read_csv(args.targets)
    if targets.empty:
        raise ValueError(f"--targets {args.targets} contains no rows")
    required_cols = set(CONDITIONING_ORDER)
    if not required_cols.issubset(targets.columns):
        raise ValueError(
            f"--targets CSV must contain columns {required_cols}; "
            f"got {set(targets.columns)}"
        )
    conds = targets[CONDITIONING_ORDER].to_numpy(dtype=np.float32)
    conds = stats.normalize(conds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []
    model.eval()
    with torch.no_grad():
        for ti, c in enumerate(conds):
            c_b = torch.as_tensor(c[None], dtype=torch.float32, device=device).repeat(args.n_per_target, 1)
            gen = diffusion.sample_ddim(model, c_b, (args.n_per_target, 1, 32, 32, 32),
                                        steps=steps, seed=0)
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