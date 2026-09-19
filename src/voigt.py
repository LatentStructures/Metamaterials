"""Shared Voigt index convention (ROADMAP sections 3.3 and 3.4).

Engineering-shear strain order ``[e11, e22, e33, g23, g13, g12]`` with
``g = 2e``.  This single definition is imported by the augmentation
(``src/data/augment.py``) and homogenization (``src/fea/property_extraction.py``)
layers so the two can never drift apart.
"""
from __future__ import annotations

VOIGT_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1),
)
