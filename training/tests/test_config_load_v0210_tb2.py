from __future__ import annotations

import pickle
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
import yaml

from assets.robot_config import load_robot_config
from policy_contract.jax.obs import build_toddlerbot_proprio_frame
from training.algos.ppo.ppo_core import create_networks, init_network_params
from training.configs.training_config import load_training_config
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.v6_eval_adapter import V6EvalAdapter
from training.policy_spec_utils import build_policy_spec_from_training_config
from training.exports.export_onnx import get_checkpoint_dims
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter


CONFIG = Path("training/configs/ppo_walking_v0210_tb2_rsl_parity.yaml")
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
    load_robot_config(cfg.env.robot_config_path)
    cfg.freeze()
    return WildRobotEnv(config=cfg)


@pytest.fixture(scope="module")
def native_adapter(cfg, env):
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )
    model = env._mj_model
    data = mujoco.MjData(model)
    signals = MujocoSignalsAdapter(
        mj_model=model,
        robot_config=robot_cfg,
        policy_spec=spec,
        foot_switch_threshold=cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=cfg,
        mj_model=model,
        policy_spec=spec,
        signals_adapter=signals,
        action_dim=spec.model.action_dim,
    )
    adapter.reset_native_mj_state(
        data,
        apply_noise=False,
        rng=None,
        perturb_pose=False,
        apply_dr=False,
    )
    return adapter, signals, data


def test_tb2_pins_current_toddlerbot_learning_contract(cfg) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert cfg.version == "0.21.0-tb2"
    assert "bootstrap" not in raw
    assert raw["quick_verify"]["ppo"]["num_envs"] == 4
    assert cfg.env.actor_obs_layout_id == "wr_obs_v12_tb_proprio"
    assert cfg.env.loc_ref_frame_zero_from_home is True
    assert cfg.env.loc_ref_reset_base == "ref_init"
    assert cfg.env.loc_ref_residual_base == "home"
    assert cfg.env.loc_ref_rsi_enabled is False
    assert cfg.env.loc_ref_clip_residual_action is False
    assert cfg.ppo.optimizer_profile == "toddlerbot_rsl_rl_2_3_3"
    assert cfg.ppo.epochs == 4
    assert cfg.ppo.num_minibatches == 16
    assert cfg.ppo.target_kl == pytest.approx(0.01)
    assert cfg.ppo.adaptive_kl_factor == pytest.approx(1.5)
    assert cfg.ppo.mirror_loss_coef == pytest.approx(0.0)
    assert cfg.ppo.critic_includes_actor_obs is False
    assert cfg.networks.actor.distribution_type == "normal"
    assert cfg.networks.actor.noise_std_type == "log"
    assert cfg.networks.actor.state_dependent_std is False
    assert np.exp(cfg.networks.actor.log_std_init) == pytest.approx(0.5)


def test_tb2_observes_all_motors_but_controls_legs(cfg) -> None:
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )

    assert tuple(spec.robot.actuator_names) == LEG_NAMES
    assert spec.robot.observation_actuator_names is not None
    assert tuple(spec.robot.observation_actuator_names) == tuple(
        joint["name"] for joint in robot_cfg.actuated_joints
    )
    assert len(spec.robot.observation_home_ctrl_rad or ()) == 17
    assert spec.model.action_dim == 10
    # 15 * (phase2 + cmd3 + q17 + qd17 + action10 + gyro3 + quat4)
    assert spec.model.obs_dim == 840


def test_tb2_reset_builds_newest_first_actor_and_critic_stacks(env) -> None:
    state = env.reset_for_eval(jnp.asarray([0, 1], dtype=jnp.uint32))
    wr = state.info[WR_INFO_KEY]

    assert state.obs.shape == (840,)
    assert wr.proprio_history.shape == (15, 56)
    assert wr.critic_obs.shape == (15 * 97,)
    np.testing.assert_array_equal(
        np.asarray(state.obs), np.asarray(wr.proprio_history).reshape(-1)
    )
    np.testing.assert_array_equal(
        np.asarray(wr.proprio_history[1:]), np.zeros((14, 56), dtype=np.float32)
    )
    assert np.all(np.isfinite(np.asarray(wr.critic_obs)))


def test_tb2_zero_command_keeps_static_pose_but_advances_phase(env) -> None:
    zero_cmd = jnp.zeros(3, dtype=jnp.float32)
    window = env._lookup_offline_window(
        jnp.asarray(12, dtype=jnp.int32), velocity_cmd=zero_cmd
    )

    # 12 * 0.02 s is one quarter of WR's Froude-scaled 0.96 s cycle.
    np.testing.assert_allclose(
        np.asarray([window["phase_sin"], window["phase_cos"]]),
        [1.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(window["contact_mask"]), [1.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(window["q_ref"]), np.asarray(env._home_q_rad), atol=1e-6
    )


def test_tb2_native_eval_uses_full_proprio_and_continuous_phase(
    native_adapter,
) -> None:
    adapter, signals_adapter, data = native_adapter
    assert signals_adapter.read(data).joint_pos_rad.shape == (17,)

    adapter.reset()
    obs0 = adapter.compute_obs(data, np.zeros(3, dtype=np.float32))
    assert obs0.shape == (840,)
    np.testing.assert_allclose(obs0[:2], [0.0, 1.0], atol=1e-6)

    adapter._state.step_idx = 12
    obs12 = adapter.compute_obs(data, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(obs12[:2], [1.0, 0.0], atol=1e-6)


def test_tb2_frame_order_matches_toddlerbot_source() -> None:
    frame = build_toddlerbot_proprio_frame(
        phase_sin_cos=jnp.asarray([1.0, 2.0]),
        velocity_cmd=jnp.asarray([3.0, 4.0, 5.0]),
        motor_pos_delta_rad=jnp.arange(17, dtype=jnp.float32) + 10.0,
        motor_vel_rad_s=jnp.arange(17, dtype=jnp.float32) + 30.0,
        prev_action=jnp.arange(10, dtype=jnp.float32) + 50.0,
        body_angvel_rad_s=jnp.asarray([60.0, 61.0, 62.0]),
        torso_quat_wxyz=jnp.asarray([-1.0, -2.0, -3.0, -4.0]),
    )

    np.testing.assert_array_equal(np.asarray(frame[:5]), np.arange(1.0, 6.0))
    np.testing.assert_array_equal(np.asarray(frame[5:22]), np.arange(10.0, 27.0))
    np.testing.assert_allclose(
        np.asarray(frame[22:39]), np.arange(30.0, 47.0) * 0.05
    )
    np.testing.assert_array_equal(np.asarray(frame[39:49]), np.arange(50.0, 60.0))
    np.testing.assert_array_equal(np.asarray(frame[49:52]), [60.0, 61.0, 62.0])
    np.testing.assert_array_equal(np.asarray(frame[52:]), [1.0, 2.0, 3.0, 4.0])


def test_tb2_normal_action_is_not_preclipped(env) -> None:
    action = jnp.full((env.action_size,), 1.25, dtype=jnp.float32)
    _, residual = env._compose_target_q_from_residual(
        policy_action=action,
        nominal_q_ref=env._home_q_rad,
    )
    np.testing.assert_allclose(
        np.asarray(residual),
        np.full(env.action_size, 0.3125, dtype=np.float32),
        atol=1e-7,
    )


def test_tb2_randomizes_toddlerbot_dynamics_and_encoder_noise(cfg) -> None:
    assert cfg.env.domain_rand_damping_scale_range == [0.8, 1.2]
    assert cfg.env.domain_rand_armature_scale_range == [0.8, 1.2]
    assert cfg.env.domain_rand_frictionloss_scale_range == [0.8, 1.2]
    assert cfg.env.domain_rand_joint_offset_rad == pytest.approx(0.0)
    assert cfg.env.joint_pos_noise_rad == pytest.approx(0.05)
    assert cfg.env.joint_vel_noise_rad_s == pytest.approx(0.10)
    assert cfg.env.reset_arm_joint_offset_range == pytest.approx((-0.1, 0.1))


def test_tb2_normal_actor_checkpoint_has_deployable_dimensions(
    cfg, tmp_path: Path
) -> None:
    network = create_networks(
        obs_dim=840,
        action_dim=10,
        policy_hidden_dims=tuple(cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(cfg.networks.critic.hidden_sizes),
        activation=cfg.networks.actor.activation,
        distribution_type=cfg.networks.actor.distribution_type,
        noise_std_type=cfg.networks.actor.noise_std_type,
        init_noise_std=float(np.exp(cfg.networks.actor.log_std_init)),
        state_dependent_std=cfg.networks.actor.state_dependent_std,
    )
    processor, policy, _ = init_network_params(
        network,
        840,
        10,
        seed=0,
        policy_init_action=jnp.zeros(10, dtype=jnp.float32),
        policy_init_std=0.5,
    )
    mean, std = network.policy_network.apply(
        processor, policy, jnp.zeros((2, 840), dtype=jnp.float32)
    )
    assert mean.shape == (2, 10)
    assert std.shape == (10,)
    np.testing.assert_allclose(np.asarray(std), 0.5, atol=1e-6)

    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(
        pickle.dumps(
            {
                "policy_params": policy,
                "config": {"actor_distribution_type": "normal"},
            }
        )
    )
    assert get_checkpoint_dims(checkpoint) == (840, 10)
