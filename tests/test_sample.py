"""Tests for DDIM sampling CLI helpers (ROADMAP section 4.3: locked 50-100)."""
import numpy as np
import pandas as pd
import pytest

from src.sample import resolve_ddim_steps, validate_target_values


def test_steps_from_model_config_default():
    assert resolve_ddim_steps(None, {}) == 50


def test_steps_from_model_config_sampling_key():
    assert resolve_ddim_steps(None, {"sampling": {"steps": 75}}) == 75


def test_cli_steps_override_config():
    assert resolve_ddim_steps(80, {"sampling": {"steps": 50}}) == 80


def test_cli_steps_clamped_to_locked_range():
    assert resolve_ddim_steps(0, {}) == 50
    assert resolve_ddim_steps(-3, {}) == 50
    assert resolve_ddim_steps(1000, {}) == 100
    assert resolve_ddim_steps(None, {"sampling": {"steps": 500}}) == 100


def test_boundary_steps_kept():
    assert resolve_ddim_steps(50, {}) == 50
    assert resolve_ddim_steps(100, {}) == 100


def test_target_values_are_unique_finite_and_in_physical_ranges():
    frame = pd.DataFrame({"E": [0.1, 0.2],
                          "relative_density": [0.2, 0.4],
                          "nu": [0.25, 0.3]})
    assert np.array_equal(validate_target_values(frame), frame.to_numpy(dtype=np.float32))


@pytest.mark.parametrize("values", [
    {"E": [0.1, 0.1], "relative_density": [0.2, 0.2], "nu": [0.3, 0.3]},
    {"E": [np.nan], "relative_density": [0.2], "nu": [0.3]},
    {"E": [-1.0], "relative_density": [0.2], "nu": [0.3]},
])
def test_target_values_reject_duplicate_or_unphysical_rows(values):
    with pytest.raises(ValueError):
        validate_target_values(pd.DataFrame(values))
