"""Config loading and fixed-seed reproducibility helpers (ROADMAP section 5).

Every run is launched from a single YAML file under ``configs/`` with a fixed
seed. No hyperparameters are defined here.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

CONDITIONING_ORDER = ["E", "relative_density", "nu"]
"""Locked conditioning order (ROADMAP Phase 1 section 3.4 / Phase 2 section 4.1).

The manifest columns, the HDF5 ``/conditioning_vector`` columns, and the model
input order must all follow this exact order. The shear modulus G is never part
of the conditioning vector.
"""


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def check_conditioning_order(config: dict) -> None:
    """Fail loudly if a config's ``conditioning_order`` contradicts the lock.

    Acceppts the order either at the top level (``dataset.yaml``) or nested
    under ``data`` (the training configs). The two YAML keys exist purely as
    documentation of the locked invariant; this check turns a silent order swap
    into a hard error at launch time.
    """
    declared = config.get("conditioning_order") or config.get("data", {}).get("conditioning_order")
    if declared is not None and list(declared) != CONDITIONING_ORDER:
        raise ValueError(
            f"config conditioning_order {list(declared)} != locked {CONDITIONING_ORDER}; "
            "the conditioning order is a locked invariant (ROADMAP Phase 1 3.4)"
        )


def set_seed(seed: int) -> None:
    """Seed the whole pipeline for reproducibility (ROADMAP section 5)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # geometry/data scripts may run without torch
        pass