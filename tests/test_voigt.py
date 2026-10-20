"""The shared Voigt convention must never drift between consumers (ROADMAP 3.3/3.4).

Engineering-shear order ``[e11, e22, e33, g23, g13, g12]`` with ``g = 2e``:
the augmentation, the homogenization, and the property-extraction layers all
contract stiffness tensors with this single tuple, so a change in one place
would silently mislabel every generated label.
"""
from src.data import augment
from src.fea import property_extraction
from src.voigt import VOIGT_PAIRS


def test_pairs_encode_the_engineering_shear_order() -> None:
    assert VOIGT_PAIRS == ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    # three normal entries first, then shear (yz, xz, xy)
    assert [ab for ab in VOIGT_PAIRS if ab[0] == ab[1]] == list(VOIGT_PAIRS[:3])
    assert len(set(VOIGT_PAIRS)) == 6


def test_every_consumer_imports_the_single_definition() -> None:
    assert augment.VOIGT_PAIRS is VOIGT_PAIRS
    assert property_extraction.VOIGT_PAIRS is VOIGT_PAIRS
