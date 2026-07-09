"""Mock control-loop smoke test: read -> obs -> predict -> compose -> write.

Runs the full runtime loop with a hardware-free MockRobotIO and a stub policy
for a finite number of steps, with no servos/IMU attached.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
from runtime.wr_runtime.control.policy_runner import RuntimePolicyRunner
from runtime.wr_runtime.control.run_policy import (
    _format_leg_targets_deg,
    _output_log_context,
    _print_timing_summary,
    run_policy_loop,
)


class _SmallPolicy:
    """Stub ONNX policy: tiny constant residual so the loop produces motion."""

    def __init__(self, action_dim: int) -> None:
        self._n = action_dim

    def predict(self, obs):
        assert obs.shape == (1129,), f"unexpected obs shape {obs.shape}"
        return np.full(self._n, 0.1, dtype=np.float32)


class _FailsAfterFirstPolicy:
    def __init__(self, action_dim: int) -> None:
        self._n = action_dim
        self._calls = 0

    def predict(self, obs):
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("policy failed")
        return np.zeros(self._n, dtype=np.float32)


class _CountingPolicy:
    def __init__(self, action_dim: int) -> None:
        self._n = action_dim
        self.calls = 0

    def predict(self, obs):
        self.calls += 1
        return np.full(self._n, 0.1, dtype=np.float32)


class _PolicyMustNotRun:
    def predict(self, obs):
        raise AssertionError("zero command should hold home without policy predict")


def test_mock_loop_runs_without_hardware(v8_spec, runtime_policy_config):
    action_dim = v8_spec.model.action_dim
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_SmallPolicy(action_dim),
        robot_io=robot_io,
    )

    logs = run_policy_loop(
        runner=runner,
        max_steps=8,
        velocity_cmd=np.array([0.13, 0.0, 0.0], dtype=np.float32),
        log_steps=2,
        ctrl_dt=runtime_policy_config.ctrl_dt,
        realtime=False,
        actuator_names=list(v8_spec.robot.actuator_names),
    )

    # One write per step, each in actuator order, finite, within joint limits.
    assert len(robot_io.written) == 8
    for ctrl in robot_io.written:
        assert ctrl.shape == (action_dim,)
        assert np.all(np.isfinite(ctrl))
    assert logs, "expected at least one logged step"
    assert logs[0]["obs"].shape == (1129,)
    assert logs[0]["timing_s"]["read"] >= 0.0
    assert logs[0]["timing_s"]["policy"] >= 0.0
    assert logs[0]["timing_s"]["write"] >= 0.0
    assert logs[0]["timing_s"]["work"] >= 0.0

    # step_idx advanced and clamped within the reference horizon.
    assert runner.step_idx == min(8, runtime_policy_config.reference.n_steps - 1)

    robot_io.close()
    assert robot_io.closed is True


def test_loop_prints_partial_timing_summary_on_error(
    v8_spec, runtime_policy_config, capsys
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_FailsAfterFirstPolicy(v8_spec.model.action_dim),
        robot_io=robot_io,
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        run_policy_loop(
            runner=runner,
            max_steps=8,
            velocity_cmd=np.array([0.13, 0.0, 0.0], dtype=np.float32),
            log_steps=1,
            ctrl_dt=runtime_policy_config.ctrl_dt,
            realtime=False,
            actuator_names=list(v8_spec.robot.actuator_names),
        )

    captured = capsys.readouterr()
    assert "Timing summary: status=partial" in captured.out
    assert "steps=1" in captured.out


def test_timing_summary_prints_servo_cache_metrics(capsys):
    _print_timing_summary(
        timing_samples=[
            {
                "work": 0.01,
                "io_servo_cache_age_max_s": 0.08,
                "io_servo_cache_age_leg_max_s": 0.04,
                "io_servo_cache_age_arm_max_s": 0.08,
                "io_servo_read_count": 5,
                "io_servo_read_fail_count": 1,
                "io_servo_cache_stale_joint_count": 0,
                "io_servo_cache_uninitialized_count": 0,
                "io_servo_latest_write_queue_latency_s": 0.001,
                "io_servo_latest_write_latency_s": 0.002,
                "io_servo_latest_read_latency_s": 0.003,
            }
        ],
        ctrl_dt=0.02,
        realtime=True,
        servo_metric_samples=[
            {
                "servo_read_group": "left_leg",
                "servo_read_ids": [1, 2, 3, 4, 9],
                "servo_read_count": 5,
                "servo_read_fail_count": 1,
                "servo_forced_read_after_write": 2,
                "servo_forced_read_after_write_missed": 0,
                "servo_write_targets_submitted": 4,
                "servo_write_targets_replaced": 1,
                "servo_write_commands": 3,
                "servo_write_commands_skipped": 2,
                "servo_write_failures": 0,
            }
        ],
    )

    out = capsys.readouterr().out
    assert "IO bottleneck p95/max ms" in out
    assert "Servo cache avg/p95/max ms" in out
    assert "Servo read/cache summary" in out
    assert "Servo worker sampled delta" in out
    assert "last_group=left_leg" in out
    assert "queue_ms_avg/p95/max" in out


def test_mock_loop_first_ctrl_is_home(v8_spec, runtime_policy_config):
    """With 1-step delay the first applied action is zero -> first ctrl == home."""
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_SmallPolicy(v8_spec.model.action_dim),
        robot_io=robot_io,
    )
    info = runner.step(np.array([0.13, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(info["target_q_rad"], runner.home_q_rad, atol=1e-6)
    assert set(["read", "obs", "policy", "compose", "write", "step"]).issubset(
        info["timing_s"]
    )


def test_zero_command_holds_home_without_policy_predict(
    v8_spec, runtime_policy_config
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_PolicyMustNotRun(),
        robot_io=robot_io,
        zero_cmd_hold_home_deadzone=1e-6,
    )

    info = runner.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert info["control_mode"] == "zero_cmd_hold_home"
    np.testing.assert_allclose(info["raw_action"], 0.0, atol=1e-6)
    np.testing.assert_allclose(info["applied_action"], 0.0, atol=1e-6)
    np.testing.assert_allclose(info["target_q_rad"], runner.home_q_rad, atol=1e-6)
    np.testing.assert_allclose(robot_io.written[-1], runner.home_q_rad, atol=1e-6)


def test_nonzero_command_uses_policy_with_zero_hold_enabled(
    v8_spec, runtime_policy_config
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    policy = _CountingPolicy(v8_spec.model.action_dim)
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=policy,
        robot_io=robot_io,
        zero_cmd_hold_home_deadzone=1e-6,
    )

    info = runner.step(np.array([0.13, 0.0, 0.0], dtype=np.float32))

    assert info["control_mode"] == "policy"
    assert policy.calls == 1
    np.testing.assert_allclose(info["raw_action"], 0.1, atol=1e-6)


def test_startup_home_hold_runs_before_policy_and_resets_state(
    v8_spec, runtime_policy_config, capsys
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    policy = _CountingPolicy(v8_spec.model.action_dim)
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=policy,
        robot_io=robot_io,
    )

    logs = run_policy_loop(
        runner=runner,
        max_steps=2,
        velocity_cmd=np.array([0.13, 0.0, 0.0], dtype=np.float32),
        log_steps=1,
        ctrl_dt=runtime_policy_config.ctrl_dt,
        realtime=False,
        actuator_names=list(v8_spec.robot.actuator_names),
        startup_home_hold_steps=3,
    )

    assert policy.calls == 2
    assert len(robot_io.written) == 5
    for ctrl in robot_io.written[:3]:
        np.testing.assert_allclose(ctrl, home, atol=1e-6)
    assert logs[0]["step_idx"] == 1
    assert runner.step_idx == 2

    out = capsys.readouterr().out
    assert "Startup home hold: steps=3" in out
    assert "mode=startup_home_hold" in out
    assert "resetting policy state before command" in out


def test_startup_command_ramp_reaches_policy_observation(
    v8_spec, runtime_policy_config
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    policy = _CountingPolicy(v8_spec.model.action_dim)
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=policy,
        robot_io=robot_io,
    )

    requested_cmd = np.array([0.13, 0.0, 0.0], dtype=np.float32)
    logs = run_policy_loop(
        runner=runner,
        max_steps=4,
        velocity_cmd=requested_cmd,
        log_steps=1,
        ctrl_dt=runtime_policy_config.ctrl_dt,
        realtime=False,
        actuator_names=list(v8_spec.robot.actuator_names),
        startup_command_ramp_steps=4,
    )

    assert policy.calls == 4
    np.testing.assert_allclose(
        [log["command_ramp_scale"] for log in logs],
        [0.25, 0.5, 0.75, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [log["obs_debug"]["velocity_cmd"] for log in logs],
        [
            requested_cmd * 0.25,
            requested_cmd * 0.5,
            requested_cmd * 0.75,
            requested_cmd,
        ],
        atol=1e-6,
    )


def test_startup_action_ramp_scales_delayed_applied_action(
    v8_spec, runtime_policy_config
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    policy = _CountingPolicy(v8_spec.model.action_dim)
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=policy,
        robot_io=robot_io,
    )

    logs = run_policy_loop(
        runner=runner,
        max_steps=4,
        velocity_cmd=np.array([0.13, 0.0, 0.0], dtype=np.float32),
        log_steps=1,
        ctrl_dt=runtime_policy_config.ctrl_dt,
        realtime=False,
        actuator_names=list(v8_spec.robot.actuator_names),
        startup_action_ramp_steps=4,
    )

    assert policy.calls == 4
    np.testing.assert_allclose(
        [log["action_ramp_scale"] for log in logs],
        [0.25, 0.5, 0.75, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(logs[0]["applied_action"], 0.0, atol=1e-6)
    np.testing.assert_allclose(logs[1]["applied_action"], 0.025, atol=1e-6)
    np.testing.assert_allclose(logs[2]["applied_action"], 0.05, atol=1e-6)
    np.testing.assert_allclose(logs[3]["applied_action"], 0.075, atol=1e-6)


def test_diagnostic_log_includes_base_orientation(
    v8_spec, runtime_policy_config, capsys
):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_SmallPolicy(v8_spec.model.action_dim),
        robot_io=robot_io,
    )

    run_policy_loop(
        runner=runner,
        max_steps=1,
        velocity_cmd=np.array([0.13, 0.0, 0.0], dtype=np.float32),
        log_steps=1,
        ctrl_dt=runtime_policy_config.ctrl_dt,
        realtime=False,
        actuator_names=list(v8_spec.robot.actuator_names),
        diagnostic_log_policy=True,
    )

    out = capsys.readouterr().out
    assert "rpy_deg=[" in out
    assert "tilt_deg=0.0" in out


class _BadShapePolicy:
    """Stub policy that returns a length-1 action (the review repro)."""

    def predict(self, obs):
        return np.array([0.5], dtype=np.float32)


def test_loop_rejects_wrong_sized_policy_output(v8_spec, runtime_policy_config):
    home = np.asarray(v8_spec.robot.home_ctrl_rad, dtype=np.float32)
    robot_io = MockRobotIO(
        actuator_names=list(v8_spec.robot.actuator_names),
        control_dt=runtime_policy_config.ctrl_dt,
        home_q_rad=home,
    )
    runner = RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_BadShapePolicy(),
        robot_io=robot_io,
    )
    with pytest.raises(ValueError, match="expected"):
        runner.step(np.array([0.13, 0.0, 0.0], dtype=np.float32))


def test_leg_target_summary_uses_actuator_names():
    actuator_names = [
        "right_knee_pitch",
        "left_hip_pitch",
        "left_knee_pitch",
    ]
    target_q_rad = np.deg2rad(np.array([30.0, 12.5, 28.0], dtype=np.float32))

    assert _format_leg_targets_deg(target_q_rad, actuator_names) == (
        "LHP=+12.5 LK=+28.0 RK=+30.0"
    )


def test_output_log_context_tees_stdout_and_stderr(tmp_path, capsys):
    log_path = tmp_path / "run.log"

    with _output_log_context(str(log_path), mirror_console=True):
        print("console and file")
        print("stderr too", file=sys.stderr)

    captured = capsys.readouterr()
    assert "console and file" in captured.out
    assert "stderr too" in captured.err
    text = log_path.read_text()
    assert "console and file" in text
    assert "stderr too" in text


def test_output_log_context_can_suppress_console(tmp_path, capsys):
    log_path = tmp_path / "run-only.log"

    with _output_log_context(str(log_path), mirror_console=False):
        print("file only")
        print("hidden stderr", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    text = log_path.read_text()
    assert "file only" in text
    assert "hidden stderr" in text
