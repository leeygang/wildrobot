from __future__ import annotations

from pathlib import Path

import jax.numpy as jp
import numpy as np
import pytest
import yaml

from assets.robot_config import load_robot_config
from policy_contract.jax.symmetry import mirror_actions
from training.configs.training_config import load_training_config
from training.envs.env_info import PRIVILEGED_OBS_DIM, WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.exports.export_policy_bundle import _build_policy_spec
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_tb1_direct_ppo.yaml")
LEG_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
)


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(CONFIG)


@pytest.fixture(scope="module")
def env(cfg):
    cfg.freeze()
    return WildRobotEnv(config=cfg)


def test_tb1_is_a_direct_ppo_cold_start(cfg) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert cfg.version == "0.21.0-tb1"
    assert "bootstrap" not in raw
    assert cfg.env.loc_ref_frame_zero_from_home is True
    assert cfg.env.loc_ref_reset_base == "home"
    assert cfg.env.loc_ref_residual_base == "home"
    assert cfg.env.loc_ref_rsi_enabled is False
    assert cfg.env.loc_ref_feet_phase_subtract_flat_baseline is False
    assert cfg.ppo.iterations * cfg.ppo.num_envs * cfg.ppo.rollout_steps >= 50_000_000
    assert cfg.ppo.learning_rate == pytest.approx(3.0e-5)
    assert cfg.ppo.entropy_coef == pytest.approx(5.0e-4)
    assert cfg.ppo.value_loss_coef == pytest.approx(0.25)
    assert cfg.ppo.epochs == 4
    assert cfg.ppo.num_minibatches == 16
    assert cfg.ppo.target_kl == pytest.approx(0.01)
    assert cfg.ppo.mirror_loss_coef == pytest.approx(1.0)
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(0.0)
    assert cfg.ppo.rollback.enabled is False
    assert cfg.reward_weights.cmd_yaw_rate_alpha == pytest.approx(7.111111)


def test_tb1_uses_leg_only_policy_with_fixed_home_upper_body(cfg) -> None:
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )

    assert tuple(spec.robot.actuator_names) == LEG_NAMES
    assert spec.model.action_dim == 10
    assert spec.model.obs_dim == 537
    fixed = (spec.provenance or {})["runtime_fixed_home"]
    assert fixed["active_actuator_names"] == list(LEG_NAMES)
    assert set(fixed["fixed_actuator_names"]) == {
        "waist_yaw",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_elbow_pitch",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_elbow_pitch",
    }


def test_tb1_export_builds_the_same_subset_contract(cfg, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pkl"
    spec = _build_policy_spec(
        checkpoint_path=checkpoint,
        config_path=CONFIG,
        robot_config_path=Path(cfg.env.robot_config_path),
    )

    assert tuple(spec.robot.actuator_names) == LEG_NAMES
    assert spec.model.action_dim == 10
    assert (spec.provenance or {})["runtime_fixed_home"][
        "full_actuator_names"
    ][0] == "waist_yaw"


def test_tb1_home_is_reference_frame_zero_and_static_reference(env) -> None:
    home = np.asarray(env._home_q_rad)
    q_ref = np.asarray(env._offline_jax_arrays["q_ref"])
    cmd_keys = np.asarray(env._offline_cmd_keys)

    np.testing.assert_array_equal(np.asarray(env._ref_init_q_rad), home)
    np.testing.assert_allclose(
        q_ref[:, 0, :], np.broadcast_to(home, q_ref[:, 0, :].shape), atol=0.0
    )
    static_index = int(np.argmin(np.linalg.norm(cmd_keys, axis=1)))
    assert np.linalg.norm(cmd_keys[static_index]) < 1e-6
    np.testing.assert_allclose(
        q_ref[static_index],
        np.broadcast_to(home, q_ref[static_index].shape),
        atol=0.0,
    )

    target, residual = env._compose_target_q_from_residual(
        policy_action=jp.zeros(env.action_size, dtype=jp.float32),
        nominal_q_ref=jp.asarray(q_ref[0, 10]),
    )
    np.testing.assert_array_equal(np.asarray(target), home)
    np.testing.assert_array_equal(np.asarray(residual), np.zeros(10, dtype=np.float32))


def test_tb1_critic_keeps_full_mechanical_privileged_width(env) -> None:
    state = env.reset_for_eval(jp.asarray([0, 1], dtype=jp.uint32))

    assert state.info[WR_INFO_KEY].critic_obs.shape == (15 * PRIVILEGED_OBS_DIM,)


def test_tb1_leg_mirror_map_is_an_involution(env) -> None:
    action = jp.arange(env.action_size, dtype=jp.float32)
    mirrored_twice = mirror_actions(
        mirror_actions(action, env._policy_spec), env._policy_spec
    )
    np.testing.assert_array_equal(np.asarray(mirrored_twice), np.asarray(action))


def test_tb1_uses_raw_toddlerbot_feet_phase_reward() -> None:
    kwargs = dict(
        left_foot_z_rel=jp.float32(0.0),
        right_foot_z_rel=jp.float32(0.0),
        phase_sin=jp.float32(1.0),
        phase_cos=jp.float32(0.0),
        is_standing=jp.asarray(False),
        swing_height=jp.float32(0.05),
        alpha=jp.float32(914.304),
        zero_on_standing=False,
    )
    raw = WildRobotEnv._feet_phase_reward(
        **kwargs, subtract_flat_baseline=False
    )
    baseline_subtracted = WildRobotEnv._feet_phase_reward(
        **kwargs, subtract_flat_baseline=True
    )

    assert float(raw) > 0.0
    assert float(baseline_subtracted) == pytest.approx(0.0)
