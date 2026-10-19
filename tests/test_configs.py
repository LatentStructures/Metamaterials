"""Regression checks for baseline config values and cross-config consistency."""
import pytest

from src.config import CONDITIONING_ORDER, REPO_ROOT, load_yaml


def _cfg(name: str) -> dict:
    return load_yaml(REPO_ROOT / "configs" / name)


def test_dataset_config_matches_locked_scope_and_minimum_size():
    cfg = _cfg("dataset.yaml")
    assert cfg["resolution"] == 32
    assert cfg["families"] == ["cubic_strut", "octet_truss", "gyroid"]
    assert cfg["num_cells"] * len(cfg["families"]) >= cfg["min_total_cells"] >= 2000
    assert cfg["conditioning_order"] == CONDITIONING_ORDER
    assert sum(cfg["splits"].values()) == pytest.approx(1.0)
    assert cfg["relative_density_range"] == [0.1, 0.6]
    assert cfg["num_test_targets"] == 50
    assert cfg["checkpoint_name"] == "dataset_v2"
    assert set(cfg["shape_parameter_ranges"]) == set(cfg["families"])
    assert cfg["shape_parameter_ranges"]["cubic_strut"][0] < cfg["shape_parameter_ranges"]["cubic_strut"][1]
    assert cfg["shape_parameter_ranges"]["octet_truss"][0] < cfg["shape_parameter_ranges"]["octet_truss"][1]
    assert cfg["shape_parameter_ranges"]["gyroid"] == [0.0, 0.08]


def test_model_config_has_locked_diffusion_and_sampling_values():
    cfg = _cfg("model_baseline.yaml")
    assert cfg["unet3d"]["channels"] == [32, 64, 128, 256]
    assert cfg["unet3d"]["attention_resolutions"] == [4]
    assert cfg["unet3d"]["attention_heads"] == 4
    assert cfg["diffusion"] == {
        "T": 1000, "beta_schedule": "linear",
        "beta_start": 0.0001, "beta_end": 0.02,
    }
    assert cfg["sampling"]["schedule"] == "ddim"
    assert 50 <= cfg["sampling"]["steps"] <= 100
    assert cfg["sampling"]["eta"] == 0.0


@pytest.mark.parametrize("name", [
    "train_baseline.yaml", "train_4060.yaml",
])
def test_training_profiles_use_option_b_and_production_data(name):
    cfg = _cfg(name)
    assert cfg["data"]["conditioning_order"] == CONDITIONING_ORDER
    assert cfg["data"]["min_samples"] >= 1600
    assert cfg["physics_loss"]["option"] == "B"
    assert cfg["physics_loss"]["lambda_1"] == 0.05
    assert cfg["physics_loss"]["ablation"]["lambda_1"] == 0.0
    assert cfg["optimizer"]["lr"] == 1e-4
    assert cfg["optimizer"]["weight_decay"] == 0.01
    assert cfg["optimizer"]["betas"] == [0.9, 0.999]
    assert cfg["ema"]["decay"] == 0.9999


def test_evaluation_material_and_resolution_match_dataset():
    dataset = _cfg("dataset.yaml")
    evaluation = _cfg("evaluate.yaml")
    assert evaluation["resolution"] == dataset["resolution"]
    assert evaluation["base_material"] == dataset["base_material"]
    assert evaluation["void_scale"] == dataset["homogenization"]["void_scale"]
