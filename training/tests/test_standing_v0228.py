from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import (
    _load_float_range,
    load_training_config,
)
from training.envs.domain_randomize import (
    apply_persistent_calibration_to_target,
    remove_persistent_calibration_from_observation,
    sample_persistent_torso_pitch_calibration,
)
from training.envs.wildrobot_env import WildRobotEnv
from training.policy_spec_utils import build_policy_spec_from_training_config


_CFG = Path(
    "training/configs/ppo_standing_stabilizer_v0228_persistent_bias.yaml"
)


def test_v0228_preserves_actor_contract_and_pins_finetune_recipe() -> None:
    cfg = load_training_config(str(_CFG))
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.22.8"
    assert spec.observation.layout_id == "wr_obs_v9_standing"
    assert spec.model.obs_dim == 59
    assert spec.model.action_dim == 17
    assert cfg.ppo.iterations == 100
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-5)
    assert cfg.env.domain_rand_persistent_torso_pitch_error_range == pytest.approx(
        (-0.218166, 0.218166)
    )
    assert cfg.env.joint_feedback_sample_hold_enabled is True
    assert cfg.env.joint_feedback_leg_period_steps_range == (4, 7)
    assert cfg.env.joint_feedback_upper_period_steps_range == (12, 24)
    assert cfg.env.penalty_pose_weights_per_joint["left_ankle_pitch"] == 1.0
    assert cfg.env.penalty_pose_weights_per_joint["right_ankle_pitch"] == 1.0
    assert cfg.ppo.eval.post_training_num_envs == 128


def test_persistent_pitch_calibration_is_bilateral_and_hidden() -> None:
    indices = jnp.asarray([0, 2, 4, 7, 9, 11], dtype=jnp.int32)
    signs = jnp.asarray([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    pitch, offsets = sample_persistent_torso_pitch_calibration(
        jax.random.PRNGKey(7),
        num_actuators=17,
        leg_pitch_actuator_indices=indices,
        leg_pitch_joint_signs=signs,
        pitch_error_range=(0.12, 0.12),
    )

    assert float(pitch) == pytest.approx(0.12)
    selected = np.asarray(offsets)[np.asarray(indices)]
    magnitudes = selected * np.asarray(signs)
    np.testing.assert_allclose(magnitudes[:3], magnitudes[3:], atol=1e-7)
    assert float(np.sum(magnitudes[:3])) == pytest.approx(0.12, abs=1e-6)
    untouched = np.delete(np.asarray(offsets), np.asarray(indices))
    np.testing.assert_array_equal(untouched, np.zeros(11, dtype=np.float32))

    requested = jnp.zeros(17, dtype=jnp.float32)
    physical = apply_persistent_calibration_to_target(
        requested,
        offsets,
        -jnp.ones(17, dtype=jnp.float32),
        jnp.ones(17, dtype=jnp.float32),
    )
    observed = remove_persistent_calibration_from_observation(physical, offsets)
    np.testing.assert_allclose(np.asarray(observed), np.asarray(requested), atol=1e-7)


def test_persistent_pitch_calibration_default_is_exact_noop() -> None:
    pitch, offsets = sample_persistent_torso_pitch_calibration(
        jax.random.PRNGKey(9),
        num_actuators=17,
        leg_pitch_actuator_indices=jnp.arange(6),
        leg_pitch_joint_signs=jnp.ones(6),
        pitch_error_range=(0.0, 0.0),
    )
    assert float(pitch) == 0.0
    np.testing.assert_array_equal(offsets, np.zeros(17, dtype=np.float32))


@pytest.mark.parametrize(
    "bad",
    ([0.0], [1.0, -1.0], [0.0, float("inf")], "-1,1"),
)
def test_persistent_pitch_range_rejects_invalid_values(bad) -> None:
    with pytest.raises(ValueError):
        _load_float_range(
            bad,
            default=(0.0, 0.0),
            field_name="env.domain_rand_persistent_torso_pitch_error_range",
        )


def test_continuous_gate_bypasses_timeout_but_not_physical_failure() -> None:
    stable_pose = SimpleNamespace(
        height=jnp.float32(0.46),
        euler_angles=lambda: (
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
        ),
    )
    fallen_pose = SimpleNamespace(
        height=jnp.float32(0.46),
        euler_angles=lambda: (
            jnp.float32(0.0),
            jnp.float32(0.5),
            jnp.float32(0.0),
        ),
    )
    dummy = SimpleNamespace(
        _cal=SimpleNamespace(get_root_pose=lambda data: data),
        _config=SimpleNamespace(
            env=SimpleNamespace(
                min_height=0.36,
                max_height=0.70,
                max_pitch=0.436332,
                max_roll=0.436332,
                use_relaxed_termination=False,
                max_episode_steps=1000,
            )
        ),
    )

    done, terminated, truncated, _ = WildRobotEnv._get_termination(
        dummy, stable_pose, jnp.int32(1000)
    )
    assert float(done) == 1.0
    assert float(terminated) == 0.0
    assert float(truncated) == 1.0

    done, terminated, truncated, _ = WildRobotEnv._get_termination(
        dummy,
        stable_pose,
        jnp.int32(3000),
        disable_time_limit=True,
    )
    assert float(done) == 0.0
    assert float(terminated) == 0.0
    assert float(truncated) == 0.0

    done, terminated, truncated, _ = WildRobotEnv._get_termination(
        dummy,
        fallen_pose,
        jnp.int32(3000),
        disable_time_limit=True,
    )
    assert float(done) == 1.0
    assert float(terminated) == 1.0
    assert float(truncated) == 0.0
