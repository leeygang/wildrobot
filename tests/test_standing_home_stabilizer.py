from __future__ import annotations

import numpy as np


class _ConstantPolicy:
    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=np.float32)
        self.last_obs: np.ndarray | None = None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.last_obs = np.asarray(obs, dtype=np.float32)
        return self.action.copy()


def test_standing_runner_expands_active_policy_to_fixed_home_hardware() -> None:
    from policy_contract.spec_builder import build_policy_spec
    from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
    from runtime.wr_runtime.control.standing_policy_runner import StandingPolicyRunner

    active_specs = [
        {"name": "waist_yaw", "range": [-1.0, 1.0], "max_velocity": 10.0},
        {"name": "left_hip_pitch", "range": [-1.0, 1.0], "max_velocity": 10.0},
        {"name": "right_hip_pitch", "range": [-1.0, 1.0], "max_velocity": 10.0},
    ]
    spec = build_policy_spec(
        robot_name="WildRobotTest",
        actuated_joint_specs=active_specs,
        action_filter_alpha=0.0,
        layout_id="wr_obs_v1",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.0, 0.1, -0.1],
    )
    hardware_names = [
        "waist_yaw",
        "left_wrist_yaw",
        "left_hip_pitch",
        "right_wrist_yaw",
        "right_hip_pitch",
    ]
    fixed_home = {
        "left_wrist_yaw": 0.25,
        "right_wrist_yaw": -0.25,
    }
    hardware_home = np.array([0.0, 0.25, 0.1, -0.25, -0.1], dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=hardware_names,
        control_dt=0.02,
        home_q_rad=hardware_home,
    )
    policy = _ConstantPolicy(np.array([0.5, -0.5, 0.25], dtype=np.float32))
    runner = StandingPolicyRunner(
        spec=spec,
        policy=policy,
        robot_io=robot_io,
        fixed_home_targets_rad=fixed_home,
    )

    info = runner.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert policy.last_obs is not None
    assert policy.last_obs.shape == (spec.model.obs_dim,)
    assert info["target_q_rad"].shape == (3,)
    assert robot_io.written[-1].shape == (5,)
    np.testing.assert_allclose(robot_io.written[-1][1], 0.25, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1][3], -0.25, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1][0], 0.5, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1][2], -0.45, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1][4], 0.175, atol=1e-6)


def test_standing_home_stabilizer_spec_excludes_wrists() -> None:
    from assets.robot_config import get_robot_config, load_robot_config
    from runtime.wr_runtime.control.run_policy import _standing_runtime_plan
    from training.configs.training_config import load_training_config
    from training.policy_spec_utils import build_policy_spec_from_training_config

    load_robot_config("assets/v2/mujoco_robot_config.json")
    training_cfg = load_training_config(
        "training/configs/ppo_standing_home_stabilizer.yaml"
    )
    robot_cfg = get_robot_config()

    spec = build_policy_spec_from_training_config(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )

    wrists = {
        "left_wrist_yaw",
        "left_wrist_pitch",
        "right_wrist_yaw",
        "right_wrist_pitch",
    }
    assert spec.observation.layout_id == "wr_obs_v1"
    assert spec.model.action_dim == 17
    assert spec.model.obs_dim == 63
    assert wrists.isdisjoint(set(spec.robot.actuator_names))
    metadata = spec.provenance["runtime_fixed_home"]
    assert set(metadata["fixed_actuator_names"]) == wrists
    assert metadata["active_actuator_names"] == spec.robot.actuator_names
    assert len(metadata["full_actuator_names"]) == 21
    assert len(metadata["fixed_home_ctrl_rad"]) == 4
    assert set(metadata["fixed_joint_ranges_rad"]) == wrists

    hardware_names, home, mins, maxs, fixed_home = _standing_runtime_plan(
        spec, externally_managed_actuator_names=sorted(wrists)
    )
    assert hardware_names == spec.robot.actuator_names
    assert home.shape == mins.shape == maxs.shape == (17,)
    assert fixed_home == {}


def test_standing_home_stabilizer_uses_active_reward_terms() -> None:
    from training.configs.training_config import load_training_config

    training_cfg = load_training_config(
        "training/configs/ppo_standing_home_stabilizer.yaml"
    )
    weights = training_cfg.reward_weights

    assert weights.alive > 0.0
    assert weights.ref_body_quat_track > 0.0
    assert weights.torso_pos_xy > 0.0
    assert weights.ang_vel_xy > 0.0
    assert weights.torso_pitch_soft > 0.0
    assert weights.torso_roll_soft > 0.0
    assert weights.penalty_pose < 0.0
    assert weights.penalty_feet_ori > 0.0
    assert weights.feet_phase > 0.0
    assert weights.cmd_forward_velocity_track == 0.0


def test_standing_home_stabilizer_v0222_is_tb_bounded_and_randomized() -> None:
    from assets.robot_config import get_robot_config, load_robot_config
    from policy_contract.calib import NumpyCalibOps
    from training.configs.training_config import load_training_config
    from training.policy_spec_utils import build_policy_spec_from_training_config

    load_robot_config("assets/v2/mujoco_robot_config.json")
    training_cfg = load_training_config(
        "training/configs/ppo_standing_home_stabilizer_v0222.yaml"
    )
    spec = build_policy_spec_from_training_config(
        training_cfg=training_cfg,
        robot_cfg=get_robot_config(),
    )

    assert training_cfg.version == "0.22.2"
    assert spec.action.mapping_id == "pos_target_home_025_v1"
    assert training_cfg.env.action_delay_steps == 1
    assert training_cfg.env.domain_randomization_enabled
    assert training_cfg.env.domain_rand_friction_range == [0.4, 1.0]
    assert training_cfg.env.domain_rand_frictionloss_scale_range == [0.8, 1.2]
    assert training_cfg.env.domain_rand_backlash_range == [0.02, 0.10]
    assert training_cfg.ppo.entropy_coef == 5e-4

    zero = np.zeros(spec.model.action_dim, dtype=np.float32)
    positive = np.ones(spec.model.action_dim, dtype=np.float32)
    home = np.asarray(spec.robot.home_ctrl_rad, dtype=np.float32)
    target = NumpyCalibOps.action_to_ctrl(spec=spec, action=positive)
    np.testing.assert_allclose(
        NumpyCalibOps.action_to_ctrl(spec=spec, action=zero), home, atol=1e-6
    )
    assert np.all(target - home <= 0.25 + 1e-6)


def test_v0222_randomization_supports_excluded_actuators() -> None:
    import jax

    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    training_cfg = load_training_config(
        "training/configs/ppo_standing_home_stabilizer_v0222.yaml"
    )
    load_robot_config(training_cfg.env.robot_config_path)
    training_cfg.freeze()
    env = WildRobotEnv(training_cfg)
    params = env._sample_domain_rand_params(jax.random.PRNGKey(0))
    randomized_model = env._get_randomized_mjx_model(params)

    policy_ids = np.asarray(env._ctrl_mapper.policy_to_mj_order, dtype=np.int32)
    excluded_ids = np.setdiff1d(np.arange(env._mj_model.nu), policy_ids)
    assert params["kp_scales"].shape == (env.action_size,)
    np.testing.assert_allclose(
        np.asarray(randomized_model.actuator_gainprm)[excluded_ids],
        np.asarray(env._base_actuator_gainprm)[excluded_ids],
    )


def test_runtime_compat_accepts_declared_fixed_home_subset() -> None:
    from policy_contract.spec import validate_runtime_compat
    from policy_contract.spec_builder import build_policy_spec

    spec = build_policy_spec(
        robot_name="WildRobotTest",
        actuated_joint_specs=[
            {"name": "waist_yaw", "range": [-1.0, 1.0], "max_velocity": 10.0},
            {"name": "left_hip_pitch", "range": [-1.0, 1.0], "max_velocity": 10.0},
        ],
        action_filter_alpha=0.0,
        layout_id="wr_obs_v1",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.0, 0.1],
        provenance={
            "runtime_fixed_home": {
                "full_actuator_names": [
                    "waist_yaw",
                    "left_wrist_yaw",
                    "left_hip_pitch",
                ],
                "active_actuator_names": ["waist_yaw", "left_hip_pitch"],
                "fixed_actuator_names": ["left_wrist_yaw"],
                "fixed_home_ctrl_rad": [0.25],
                "fixed_joint_ranges_rad": {"left_wrist_yaw": [-1.0, 1.0]},
                "source": "test",
            }
        },
    )

    validate_runtime_compat(
        spec=spec,
        mjcf_actuator_names=["waist_yaw", "left_wrist_yaw", "left_hip_pitch"],
        onnx_obs_dim=spec.model.obs_dim,
        onnx_action_dim=spec.model.action_dim,
    )
