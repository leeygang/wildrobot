from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from training.configs.training_config import load_training_config_from_dict
from training.configs.zmp_reference import zmp_walk_config_from_env
from training.scripts.run_morphology_size_factorial import (
    PROJECT_ROOT,
    SOURCE_SCENE,
    FactorialArm,
    _factorial_effects,
    build_arm_config,
    inspect_model,
    validate_scaled_model,
    write_scaled_assets,
)


def test_zmp_reference_scales_lengths_and_froude_time() -> None:
    baseline = zmp_walk_config_from_env({})
    scaled = zmp_walk_config_from_env(
        {"loc_ref_morphology_length_scale": 0.64}
    )

    assert scaled.upper_leg_m == pytest.approx(baseline.upper_leg_m * 0.64)
    assert scaled.lower_leg_m == pytest.approx(baseline.lower_leg_m * 0.64)
    assert scaled.foot_step_height_m == pytest.approx(
        baseline.foot_step_height_m * 0.64
    )
    assert scaled.max_step_length_m == pytest.approx(
        baseline.max_step_length_m * 0.64
    )
    assert scaled.cycle_time_s == pytest.approx(
        baseline.cycle_time_s * math.sqrt(0.64)
    )
    assert scaled.dt_s == baseline.dt_s


def test_zmp_reference_rejects_invalid_morphology_scale() -> None:
    with pytest.raises(ValueError, match="morphology_length_scale"):
        zmp_walk_config_from_env({"loc_ref_morphology_length_scale": 0.0})

    with pytest.raises(ValueError, match="already frozen its geometry"):
        zmp_walk_config_from_env(
            {"loc_ref_morphology_length_scale": 0.8},
            offline_library_path="frozen.npz",
        )


def test_scaled_mjcf_preserves_topology_actuator_and_expected_physics(
    tmp_path: Path,
) -> None:
    geometry_scale = 0.8
    mass_scale = 0.9
    baseline = inspect_model(SOURCE_SCENE)
    scene = write_scaled_assets(
        tmp_path / "scaled",
        geometry_scale=geometry_scale,
        mass_scale=mass_scale,
    )
    candidate = inspect_model(scene)

    validate_scaled_model(
        baseline,
        candidate,
        geometry_scale=geometry_scale,
        mass_scale=mass_scale,
    )
    assert candidate["total_mass_kg"] == pytest.approx(
        baseline["total_mass_kg"] * mass_scale,
        rel=2e-5,
    )
    assert candidate["leg_joint_chain_length_m"] == pytest.approx(
        baseline["leg_joint_chain_length_m"] * geometry_scale,
        rel=2e-5,
    )
    assert candidate["actuator_force_limit_max_nm"] == pytest.approx(
        baseline["actuator_force_limit_max_nm"]
    )


def test_arm_config_scales_dimensioned_contract_and_saves_each_iteration() -> None:
    base_path = (
        PROJECT_ROOT
        / "training/configs/ppo_walking_v0210_tb11_reference_geometry_com_lever.yaml"
    )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    arm = FactorialArm(
        name="dimension_and_mass",
        version="0.21.0-test",
        geometry_scale=0.8,
        mass_scale=0.9,
    )
    config = build_arm_config(
        base,
        arm=arm,
        scene_path=SOURCE_SCENE,
        seed=7,
        iterations=20,
    )
    parsed = load_training_config_from_dict(config)

    assert parsed.env.loc_ref_morphology_length_scale == pytest.approx(0.8)
    assert parsed.env.target_height == pytest.approx(0.45 * 0.8)
    assert parsed.env.loc_ref_default_stance_width_m == pytest.approx(0.0782 * 0.8)
    assert parsed.env.loc_ref_hip_lateral_offset_m == pytest.approx(0.0892 * 0.8)
    assert parsed.env.close_feet_threshold == pytest.approx(0.146 * 0.8)
    assert parsed.env.single_support_com_lateral_normalization_m == pytest.approx(
        0.10 * 0.8
    )
    assert parsed.env.contact_threshold_force == pytest.approx(1.0 * 0.9)
    assert parsed.reward_weights.feet_phase_swing_height == pytest.approx(0.05 * 0.8)
    assert parsed.reward_weights.feet_phase_alpha == pytest.approx(
        914.304 / (0.8 * 0.8)
    )
    assert parsed.ppo.iterations == 20
    assert parsed.checkpoints.interval == 1
    assert config["env"]["eval_velocity_cmd_probes"] == [[0.066667, 0.0, 0.0]]


def test_factorial_effects_use_primary_matched_command() -> None:
    jobs = [
        {
            "arm": arm,
            "reference_metrics": [{"max_support_ratio_p95": value}],
        }
        for arm, value in (
            ("baseline", 0.70),
            ("dimension_only", 0.40),
            ("mass_only", 0.65),
            ("dimension_and_mass", 0.37),
        )
    ]

    effects = _factorial_effects(jobs)

    assert effects["dimension_effect_absolute"] == pytest.approx(-0.30)
    assert effects["mass_effect_absolute"] == pytest.approx(-0.05)
    assert effects["combined_effect_absolute"] == pytest.approx(-0.33)
    assert effects["interaction_absolute"] == pytest.approx(0.02)
