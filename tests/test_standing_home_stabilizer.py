from __future__ import annotations

import numpy as np
import pytest


_DEPLOYMENT_BUNDLE = (
    "runtime/bundles/deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90"
)


class _ConstantPolicy:
    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=np.float32)
        self.last_obs: np.ndarray | None = None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.last_obs = np.asarray(obs, dtype=np.float32)
        return self.action.copy()


def test_standing_runner_requires_direct_native_hardware_order() -> None:
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
    hardware_names = list(spec.robot.actuator_names)
    hardware_home = np.array([0.0, 0.1, -0.1], dtype=np.float32)
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
    )

    info = runner.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert policy.last_obs is not None
    assert policy.last_obs.shape == (spec.model.obs_dim,)
    assert info["target_q_rad"].shape == (3,)
    assert robot_io.written[-1].shape == (3,)
    np.testing.assert_allclose(robot_io.written[-1][0], 0.5, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1][1], -0.45, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1][2], 0.175, atol=1e-6)


def test_standing_home_stabilizer_spec_is_natively_wrist_free() -> None:
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
    assert not spec.provenance or "runtime_fixed_home" not in spec.provenance

    hardware_names, home, mins, maxs = _standing_runtime_plan(spec)
    assert hardware_names == spec.robot.actuator_names
    assert home.shape == mins.shape == maxs.shape == (17,)


def test_stable_only_bundle_is_17_action_and_excludes_wrist_io() -> None:
    from runtime.wr_runtime.deployment_bundle import DeploymentBundle
    from runtime.wr_runtime.control.run_policy import (
        _resolve_run_bundle_path,
        _standing_runtime_plan,
    )

    bundle_path = _resolve_run_bundle_path(
        bundle_arg=_DEPLOYMENT_BUNDLE, stable_only=True
    )
    deployment = DeploymentBundle.load(bundle_path)
    bundle = deployment.policy_bundle("standing")
    hardware_names, home, mins, maxs = _standing_runtime_plan(bundle.spec)

    assert bundle_path.name == "deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90"
    assert bundle.spec.observation.layout_id == "wr_obs_v1"
    assert bundle.spec.model.obs_dim == 63
    assert bundle.spec.model.action_dim == 17
    assert hardware_names == bundle.spec.robot.actuator_names
    assert home.shape == mins.shape == maxs.shape == (17,)


def test_stable_only_hardware_runs_without_step_limit() -> None:
    from runtime.wr_runtime.control.run_policy import _policy_loop_max_steps

    assert (
        _policy_loop_max_steps(stable_only=True, dry_run=False, max_steps=500)
        is None
    )
    assert _policy_loop_max_steps(
        stable_only=True, dry_run=True, max_steps=5
    ) == 5
    assert _policy_loop_max_steps(
        stable_only=False, dry_run=False, max_steps=500
    ) == 500


def test_policy_cli_handles_interrupt_without_log(monkeypatch, capsys) -> None:
    from runtime.wr_runtime.control import run_policy

    def _interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_policy, "_run_policy_from_args", _interrupt)

    assert run_policy.main(["--bundle", "ignored"]) == 130
    assert "Interrupted." in capsys.readouterr().err


def test_bundle_is_required_for_all_runtime_modes() -> None:
    from runtime.wr_runtime.control.run_policy import _resolve_run_bundle_path

    for stable_only in (False, True):
        with pytest.raises(SystemExit, match="--bundle is required"):
            _resolve_run_bundle_path(bundle_arg=None, stable_only=stable_only)


def test_standing_runtime_applies_exported_one_step_action_delay() -> None:
    from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
    from runtime.wr_runtime.control.runtime_policy_config import (
        StandingRuntimePolicyConfig,
    )
    from runtime.wr_runtime.control.standing_policy_runner import StandingPolicyRunner
    from runtime.wr_runtime.deployment_bundle import DeploymentBundle

    deployment = DeploymentBundle.load(_DEPLOYMENT_BUNDLE)
    bundle = deployment.policy_bundle("standing")
    runtime_cfg = StandingRuntimePolicyConfig.from_json(
        deployment.policy_dir("standing") / "runtime_policy_config.json"
    )
    robot_io = MockRobotIO(
        actuator_names=bundle.spec.robot.actuator_names,
        control_dt=runtime_cfg.ctrl_dt,
        home_q_rad=np.asarray(bundle.spec.robot.home_ctrl_rad, dtype=np.float32),
    )
    runner = StandingPolicyRunner(
        spec=bundle.spec,
        policy=_ConstantPolicy(np.ones(17, dtype=np.float32)),
        robot_io=robot_io,
        runtime_config=runtime_cfg,
    )

    _, first_applied = runner.compose_and_apply(np.ones(17, dtype=np.float32))
    _, second_applied = runner.compose_and_apply(np.ones(17, dtype=np.float32))

    np.testing.assert_allclose(first_applied, 0.0, atol=1e-7)
    assert np.max(np.abs(second_applied)) > 0.0


def test_startup_standing_honors_policy_diagnostics(capsys) -> None:
    from policy_contract.spec_builder import build_policy_spec
    from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
    from runtime.wr_runtime.control.run_policy import _run_standing_stabilization
    from runtime.wr_runtime.control.standing_policy_runner import StandingPolicyRunner

    actuator_names = ["left_hip_pitch", "right_hip_pitch"]
    spec = build_policy_spec(
        robot_name="WildRobotTest",
        actuated_joint_specs=[
            {"name": name, "range": [-1.0, 1.0], "max_velocity": 10.0}
            for name in actuator_names
        ],
        action_filter_alpha=0.0,
        layout_id="wr_obs_v1",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.0, 0.0],
    )
    runner = StandingPolicyRunner(
        spec=spec,
        policy=_ConstantPolicy(np.array([0.25, -0.5], dtype=np.float32)),
        robot_io=MockRobotIO(
            actuator_names=actuator_names,
            control_dt=0.02,
            home_q_rad=np.zeros(2, dtype=np.float32),
        ),
    )

    _run_standing_stabilization(
        runner=runner,
        steps=1,
        log_steps=1,
        ctrl_dt=0.02,
        realtime=False,
        actuator_names=actuator_names,
        diagnostic_log_policy=True,
        stability_check=False,
        stability_max_tilt_deg=10.0,
        confirm_before_walk=False,
        confirm_imu_timeout_s=1.0,
    )

    output = capsys.readouterr().out
    assert "leg_deg=LHP=+14.3 RHP=-28.6" in output
    assert "obs_leg_deg=LHP=+0.0 RHP=+0.0" in output
    assert "diag[|raw|max=0.500" in output
    assert "raw_lr=[HP=+0.750" in output


def test_startup_standing_aborts_on_first_unsafe_tilt() -> None:
    from policy_contract.numpy.signals import Signals
    from policy_contract.spec_builder import build_policy_spec
    from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
    from runtime.wr_runtime.control.run_policy import _run_standing_stabilization
    from runtime.wr_runtime.control.standing_policy_runner import StandingPolicyRunner

    actuator_names = ["left_hip_pitch", "right_hip_pitch"]
    spec = build_policy_spec(
        robot_name="WildRobotTest",
        actuated_joint_specs=[
            {"name": name, "range": [-1.0, 1.0], "max_velocity": 10.0}
            for name in actuator_names
        ],
        action_filter_alpha=0.0,
        layout_id="wr_obs_v1",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.0, 0.0],
    )

    class _TiltedRobotIO(MockRobotIO):
        def read(self) -> Signals:
            signals = super().read()
            pitch = np.deg2rad(20.0)
            return Signals(
                quat_wxyz=np.array(
                    [np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0],
                    dtype=np.float32,
                ),
                gyro_rad_s=signals.gyro_rad_s,
                joint_pos_rad=signals.joint_pos_rad,
                joint_vel_rad_s=signals.joint_vel_rad_s,
                foot_switches=signals.foot_switches,
                timestamp_s=signals.timestamp_s,
            )

    robot_io = _TiltedRobotIO(
        actuator_names=actuator_names,
        control_dt=0.02,
        home_q_rad=np.zeros(2, dtype=np.float32),
    )
    runner = StandingPolicyRunner(
        spec=spec,
        policy=_ConstantPolicy(np.zeros(2, dtype=np.float32)),
        robot_io=robot_io,
    )

    with pytest.raises(SystemExit, match="safety abort.*step 1.*tilt 20.0deg"):
        _run_standing_stabilization(
            runner=runner,
            steps=100,
            log_steps=5,
            ctrl_dt=0.02,
            realtime=False,
            actuator_names=actuator_names,
            diagnostic_log_policy=False,
            stability_check=True,
            stability_max_tilt_deg=15.0,
            confirm_before_walk=False,
            confirm_imu_timeout_s=1.0,
        )

    assert len(robot_io.written) == 1


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


def test_v0222_randomization_leaves_mujoco_only_actuators_unchanged() -> None:
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


def test_runtime_compat_accepts_policy_subset_of_mujoco_actuators() -> None:
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
    )

    validate_runtime_compat(
        spec=spec,
        mjcf_actuator_names=["waist_yaw", "left_wrist_yaw", "left_hip_pitch"],
        onnx_obs_dim=spec.model.obs_dim,
        onnx_action_dim=spec.model.action_dim,
    )
