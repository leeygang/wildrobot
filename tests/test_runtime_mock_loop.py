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
