"""HDF5-backed PyTorch Dataset for the voxel+conditioning baseline (ROADMAP section 3.4).

HDF5 schema (one file per split):
    /voxels              (N, 32, 32, 32)  uint8  binary occupancy
    /stiffness_tensor    (N, 6, 6)        float32  Voigt C_ijkl (engineering shear)
    /conditioning_vector (N, 3)           float32  [E, rho, nu] NORMALIZED
    /metadata            (N,) records     family, generating parameters
    /conditioning_vector attributes:
        mean, std (3,)  -- dataset-wide stats used for normalization

The loader returns (float32 tensor (1,32,32,32) voxels, float32 tensor (3,) cond).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CONDITIONING_ORDER

# A voxel transform augments the grid in place and may be sample-index-aware so
# that augmentation stays reproducible even when DataLoader workers consume
# samples out of order (each index always maps to the same rotation).
VoxelTransform = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class ConditioningStats:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class MetamaterialDataset(Dataset):
    """Loads one split entirely into memory (<=5,000 * 32^3 uint8 ~ 160 MB)."""

    def __init__(self,
                 h5_path: str | Path,
                 transform: VoxelTransform | None = None,
                 return_stiffness: bool = False,
                 dtype: torch.dtype = torch.float32):
        self.path = Path(h5_path)
        self.transform = transform
        self.return_stiffness = return_stiffness
        self.dtype = dtype
        with h5py.File(self.path, "r") as h5:
            self.voxels = h5["/voxels"][...]
            self.cond = h5["/conditioning_vector"][...].astype(np.float32)
            self.has_stiffness = "/stiffness_tensor" in h5
            self.stiffness = (h5["/stiffness_tensor"][...].astype(np.float32)
                              if self.has_stiffness else None)
            self.has_metadata = "/metadata" in h5
            stats_group = h5["/conditioning_vector"].attrs
            if "mean" in stats_group and "std" in stats_group:
                self.stats = ConditioningStats(
                    np.asarray(stats_group["mean"], dtype=np.float32),
                    np.asarray(stats_group["std"], dtype=np.float32),
                )
            else:
                self.stats = None
        if (self.voxels.ndim != 4 or self.voxels.shape[1] != self.voxels.shape[2]
                or self.voxels.shape[2] != self.voxels.shape[3]):
            raise ValueError(
                f"bad voxel shape at {self.path}: {self.voxels.shape}; "
                f"expected a cubic (N, D, D, D) grid"
            )
        if self.cond.ndim != 2 or self.cond.shape[1] != len(CONDITIONING_ORDER):
            raise ValueError(
                f"bad conditioning shape at {self.path}: {self.cond.shape}; "
                f"expected (N, {len(CONDITIONING_ORDER)}) in order {CONDITIONING_ORDER}"
            )
        if len(self.voxels) != len(self.cond):
            raise ValueError(
                f"length mismatch at {self.path}: voxels={len(self.voxels)} "
                f"cond={len(self.cond)}"
            )
        if self.stats is not None and np.any(self.stats.std <= 0.0):
            raise ValueError(
                f"{self.path}: conditioning stats carry a zero std column "
                f"({self.stats.std}); it would divide by zero on normalize"
            )

    def __len__(self) -> int:
        return len(self.voxels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.voxels[idx]
        if v.dtype == np.uint8 and v.max() > 1:
            v = v.astype(np.float32) / 255.0
        else:
            v = v.astype(np.float32)
        if self.transform is not None:
            v = self.transform(v, idx)
        x = torch.from_numpy(v).unsqueeze(0).to(self.dtype)
        c = torch.from_numpy(self.cond[idx]).to(self.dtype)
        if not self.return_stiffness:
            return x, c
        C = torch.from_numpy(self.stiffness[idx]).to(self.dtype) if self.has_stiffness else None
        return x, c, C

    def conditioning_mean_std(self) -> ConditioningStats:
        if self.stats is None:
            raise ValueError(f"{self.path}: /conditioning_vector has no mean/std attrs")
        return self.stats


def read_conditioning_stats(h5_path: str | Path) -> ConditioningStats:
    with h5py.File(h5_path, "r") as h5:
        cvec = h5["/conditioning_vector"]
        if cvec.ndim != 2 or cvec.shape[1] != len(CONDITIONING_ORDER):
            raise ValueError(
                f"{h5_path}: /conditioning_vector has shape {cvec.shape}; "
                f"expected (N, {len(CONDITIONING_ORDER)}) per {CONDITIONING_ORDER}"
            )
        if isinstance(h5.get("/conditioning_order"), h5py.Dataset):
            stored = [str(s, "utf-8") for s in h5["/conditioning_order"][...]]
            if stored != CONDITIONING_ORDER:
                raise ValueError(
                    f"{h5_path}: /conditioning_order {stored} != locked {CONDITIONING_ORDER}"
                )
        attrs = cvec.attrs
        try:
            mean = np.asarray(attrs["mean"], dtype=np.float32)
            std = np.asarray(attrs["std"], dtype=np.float32)
        except KeyError:
            raise ValueError(
                f"{h5_path}: /conditioning_vector is missing mean/std attrs "
                "(not produced by write_conditioned_hdf5?)"
            ) from None
        if mean.shape != (len(CONDITIONING_ORDER),) or std.shape != (len(CONDITIONING_ORDER),):
            raise ValueError(
                f"{h5_path}: conditioning stats have shape {mean.shape}/{std.shape}; "
                f"expected ({len(CONDITIONING_ORDER)},)"
            )
        if np.any(std <= 0.0):
            raise ValueError(f"{h5_path}: conditioning std contains a non-positive entry {std}")
        return ConditioningStats(mean, std)


def write_conditioned_hdf5(path: str | Path,
                           voxels: np.ndarray,
                           cond: np.ndarray,
                           stiffness: np.ndarray | None = None,
                           metadata: np.ndarray | None = None,
                           stats: ConditioningStats | None = None,
                           chunk: int = 64) -> None:
    """Write one split file per the ROADMAP section 3.4 schema (chunked, uint8/float32)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vox = np.asarray(voxels)
    if vox.dtype != np.uint8:
        vox = (vox > 0).astype(np.uint8)
    cond = np.asarray(cond, dtype=np.float32)
    if (vox.ndim != 4 or vox.shape[1] != vox.shape[2] or vox.shape[2] != vox.shape[3]):
        raise ValueError(
            f"bad voxel shape: {vox.shape}; expected a cubic (N, D, D, D) grid"
        )
    if cond.ndim != 2 or cond.shape[1] != len(CONDITIONING_ORDER):
        raise ValueError(
            f"bad conditioning shape: {cond.shape}; "
            f"expected (N, {len(CONDITIONING_ORDER)}) in order {CONDITIONING_ORDER}"
        )
    if len(vox) != len(cond):
        raise ValueError(f"length mismatch: voxels={len(vox)} cond={len(cond)}")
    if stats is not None and (stats.mean.shape != (len(CONDITIONING_ORDER),)
                              or stats.std.shape != (len(CONDITIONING_ORDER),)):
        raise ValueError(
            f"stats shape {stats.mean.shape}/{stats.std.shape} != "
            f"({len(CONDITIONING_ORDER)},) conditioning order"
        )
    if stats is not None and np.any(stats.std <= 0.0):
        raise ValueError(f"stats.std contains a non-positive entry {stats.std}")
    if metadata is not None and len(metadata) != len(vox):
        raise ValueError(f"metadata length {len(metadata)} != voxel count {len(vox)}")
    chunk_v = min(chunk, len(vox))
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/voxels", data=vox, chunks=(chunk_v, *vox.shape[1:]),
                          compression="gzip")
        h5.create_dataset("/conditioning_vector", data=cond, chunks=(chunk_v, cond.shape[1]),
                          compression="gzip")
        # Locked column order, persisted so the manifest and model input can be
        # checked against the file itself (ROADMAP 3.4: match silently exactly).
        h5.create_dataset(
            "/conditioning_order",
            data=np.asarray([n.encode("utf-8") for n in CONDITIONING_ORDER]),
        )
        cds = h5["/conditioning_vector"]
        if stats is not None:
            cds.attrs["mean"] = stats.mean.astype(np.float32)
            cds.attrs["std"] = stats.std.astype(np.float32)
        if stiffness is not None:
            h5.create_dataset("/stiffness_tensor", data=np.asarray(stiffness, dtype=np.float32),
                              chunks=(chunk_v, 6, 6), compression="gzip")
        if metadata is not None:
            h5.create_dataset("/metadata", data=np.asarray(metadata))