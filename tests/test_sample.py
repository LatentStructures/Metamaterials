"""Tests for DDIM sampling CLI helpers (ROADMAP section 4.3: locked 50-100)."""
from src.sample import resolve_ddim_steps


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