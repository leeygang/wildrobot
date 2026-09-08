from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.verify_walking_stance_geometry import (
    analyze_stance_candidate,
    load_stance_inputs,
)
from training.exports.export_policy_bundle import (
    _build_policy_spec as build_export_policy_spec,
    _get_home_ctrl_from_mjcf,
)
from training.policy_spec_utils import (
    build_policy_spec_from_training_config,
    get_home_ctrl_from_mj_model,
)


CONFIG = Path("training/configs/ppo_walking_v0210_tb3_forward_shared_stance.yaml")
TB2_CONFIG = Path("training/configs/ppo_walking_v0210_tb2_rsl_parity.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(CONFIG)


@pytest.fixture(scope="module")
def env(cfg):
    cfg.freeze()
    return WildRobotEnv(config=cfg)


def test_tb3_changes_task_scope_and_home_but_preserves_tb2_learner(cfg) -> None:
    tb2 = load_training_config(TB2_CONFIG)

    assert cfg.version == "0.21.0-tb3"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 14_991_360
    assert cfg.ppo.eval.interval == 122
    assert cfg.checkpoints.interval == 122

    assert cfg.env.min_velocity == pytest.approx(0.066667)
    assert cfg.env.max_velocity == pytest.approx(0.133333)
    assert cfg.env.min_velocity_y == pytest.approx(0.0)
    assert cfg.env.max_velocity_y == pytest.approx(0.0)
    assert cfg.env.max_yaw_rate == pytest.approx(0.0)
    assert cfg.env.cmd_zero_chance == pytest.approx(0.20)
    assert cfg.env.cmd_turn_chance == pytest.approx(0.0)
    assert cfg.env.cmd_sampler_3d_branched is False
    assert cfg.env.loc_ref_command_axes_3d is False
    assert cfg.env.eval_velocity_cmd_probes == ()

    assert cfg.reward_weights.cmd_velocity_track_dim == 1
    assert cfg.reward_weights.cmd_forward_velocity_alpha == pytest.approx(562.5)
    assert cfg.reward_weights.cmd_yaw_rate_track == pytest.approx(0.0)
    assert cfg.reward_weights.saturation == pytest.approx(0.0)

    assert cfg.env.home_joint_offsets_rad == {
        "left_hip_roll": pytest.approx(0.030),
        "right_hip_roll": pytest.approx(-0.030),
        "left_ankle_roll": pytest.approx(-0.030),
        "right_ankle_roll": pytest.approx(0.030),
    }
    assert cfg.env.loc_ref_walking_joint_offsets_rad == {}

    for field in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "entropy_coef",
        "value_loss_coef",
        "epochs",
        "num_minibatches",
        "max_grad_norm",
        "optimizer_profile",
        "adaptive_kl_min_learning_rate",
        "adaptive_kl_max_learning_rate",
        "adaptive_kl_factor",
        "target_kl",
    ):
        assert getattr(cfg.ppo, field) == getattr(tb2.ppo, field)
    assert cfg.networks.actor == tb2.networks.actor
    assert cfg.networks.critic == tb2.networks.critic


def test_tb3_canonical_home_is_shared_by_env_spec_frame_zero_and_export(
    cfg,
    env,
) -> None:
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )
    actuator_names = list(spec.robot.actuator_names)
    raw_home = get_home_ctrl_from_mj_model(
        mj_model=env._mj_model,
        actuator_names=actuator_names,
    )
    expected = np.asarray(
        [
            value + cfg.env.home_joint_offsets_rad.get(name, 0.0)
            for name, value in zip(actuator_names, raw_home)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(env._home_q_rad, expected, atol=1e-7)
    np.testing.assert_allclose(env._walking_home_q_rad, expected, atol=1e-7)
    np.testing.assert_allclose(env._ref_init_q_rad, expected, atol=1e-7)
    np.testing.assert_allclose(spec.robot.home_ctrl_rad, expected, atol=1e-7)

    exported_home = _get_home_ctrl_from_mjcf(CONFIG, actuator_names)
    np.testing.assert_allclose(exported_home, expected, atol=1e-7)
    export_spec = build_export_policy_spec(
        checkpoint_path=Path("/tmp/tb3-placeholder.pkl"),
        config_path=CONFIG,
        robot_config_path=Path(cfg.env.robot_config_path),
    )
    np.testing.assert_allclose(export_spec.robot.home_ctrl_rad, expected, atol=1e-7)

    state = env.reset_for_eval(jnp.asarray([0, 1], dtype=jnp.uint32))
    np.testing.assert_allclose(
        np.asarray(state.data.qpos[env._actuator_qpos_addrs]),
        expected,
        atol=1e-7,
    )
    zero_window = env._lookup_offline_window(
        jnp.asarray(0, dtype=jnp.int32),
        velocity_cmd=jnp.zeros(3, dtype=jnp.float32),
    )
    np.testing.assert_allclose(zero_window["q_ref"], expected, atol=1e-7)

    full_names = [
        str(env._mj_model.actuator(actuator_id).name)
        for actuator_id in range(env._mj_model.nu)
    ]
    observation_home = dict(
        zip(full_names, spec.robot.observation_home_ctrl_rad or ())
    )
    raw_full_home = get_home_ctrl_from_mj_model(
        mj_model=env._mj_model,
        actuator_names=full_names,
    )
    for name, offset in cfg.env.home_joint_offsets_rad.items():
        raw_index = full_names.index(name)
        assert observation_home[name] == pytest.approx(
            raw_full_home[raw_index] + offset,
            abs=1e-7,
        )


def test_tb3_sampler_emits_only_stand_or_positive_forward(env) -> None:
    commands = np.asarray(
        jax.vmap(env._sample_velocity_cmd)(
            jax.random.split(jax.random.PRNGKey(7), 512)
        )
    )
    np.testing.assert_allclose(commands[:, 1:], 0.0, atol=0.0)
    assert np.any(commands[:, 0] == 0.0)
    walking = commands[commands[:, 0] > 0.0, 0]
    assert walking.size > 0
    assert float(walking.min()) >= 0.066667 - 1e-6
    assert float(walking.max()) <= 0.133333 + 1e-6


def test_tb3_configured_home_passes_static_stance_geometry() -> None:
    (
        model,
        robot_config,
        home_qpos,
        home_foot_rotations,
        close_feet_threshold_m,
    ) = load_stance_inputs(CONFIG)
    result = analyze_stance_candidate(
        model=model,
        robot_config=robot_config,
        home_qpos=home_qpos,
        home_foot_rotations=home_foot_rotations,
        offset_rad=0.0,
        close_feet_threshold_m=close_feet_threshold_m,
        max_support_torque_ratio=0.8,
        max_foot_orientation_delta_deg=1.0,
        max_sole_height_delta_m=0.002,
    )

    assert result["passed"]
    assert result["foot_center_separation_m"] == pytest.approx(0.1564, abs=0.001)
    assert max(
        side["quasi_static_support_ratio"] for side in result["support"].values()
    ) == pytest.approx(0.784, abs=0.003)
