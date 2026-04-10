from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp

from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.configs.training_config import load_training_config
from training.eval.visualize_nominal_ref import (
    _CommandPathTracker,
    _compute_stance_ground_shift,
    _configure_nominal_only,
    _dominant_termination_from_metrics,
    _enable_temp_logging,
    _geom_bottom_z,
    _print_log_only,
    _extract_lateral_semantics,
    _extract_reset_lateral_semantics,
    _format_grounded_nominal_init_line,
    _extract_support_posture,
    _format_support_posture_line,
    _format_step_line,
    _format_reset_lateral_line,
    _format_reset_lateral_line_from_state,
    _replace_state_with_nominal_qref,
    _replace_state_with_nominal_qref_grounded,
    parse_args,
)
from training.envs.env_info import WR_INFO_KEY


def test_parse_args_nominal_ref_viewer() -> None:
    args = parse_args(
        [
            "--config",
            "training/configs/ppo_walking_v0193a.yaml",
            "--forward-cmd",
            "0.10",
            "--horizon",
            "64",
            "--headless",
            "--print-every",
            "4",
            "--seed",
            "7",
            "--no-stop-on-done",
        ]
    )
    assert args.config.endswith("ppo_walking_v0193a.yaml")
    assert abs(args.forward_cmd - 0.10) < 1e-9
    assert args.horizon == 64
    assert args.headless is True
    assert args.print_every == 4
    assert args.seed == 7
    assert args.stop_on_done is False
    assert args.force_support_only is False
    assert args.init_from_nominal_qref is False
    assert args.log is False
    assert args.disable_action_filter is False
    assert args.startup_target_rate_deg_s is None


def test_parse_args_disable_action_filter_flag() -> None:
    args = parse_args(["--disable-action-filter"])
    assert args.disable_action_filter is True


def test_parse_args_startup_target_rate_deg_s_flag() -> None:
    args = parse_args(["--startup-target-rate-deg-s", "10"])
    assert abs(args.startup_target_rate_deg_s - 10.0) < 1e-9


def test_compute_stance_ground_shift_uses_selected_stance_foot() -> None:
    assert abs(
        _compute_stance_ground_shift(stance_foot=0, left_foot_z=0.12, right_foot_z=-0.03) + 0.12
    ) < 1e-9
    assert abs(
        _compute_stance_ground_shift(stance_foot=1, left_foot_z=0.12, right_foot_z=-0.03) - 0.03
    ) < 1e-9


def test_geom_bottom_z_uses_box_extent() -> None:
    class _MjModel:
        geom_size = jnp.asarray([[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]], dtype=jnp.float32)
        geom_type = jnp.asarray([0, int(__import__("mujoco").mjtGeom.mjGEOM_BOX)], dtype=jnp.int32)

    class _MjData:
        geom_xpos = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)
        geom_xmat = jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )

    assert abs(_geom_bottom_z(_MjModel(), _MjData(), 1) - 2.6) < 1e-6


def test_dominant_termination_from_metrics() -> None:
    vec = jnp.zeros((len(METRIC_INDEX),), dtype=jnp.float32)
    assert _dominant_termination_from_metrics(vec) == "none"

    vec = vec.at[METRIC_INDEX["term/pitch"]].set(1.0)
    vec = vec.at[METRIC_INDEX["term/height_low"]].set(0.5)
    assert _dominant_termination_from_metrics(vec) == "term/pitch"


def test_configure_nominal_only_forces_no_push_and_no_action_delay() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0193a.yaml")
    # Deliberately set conflicting values to verify normalization.
    cfg.env.push_enabled = True
    cfg.env.action_delay_steps = 1
    cfg.env.action_filter_alpha = 0.3
    cfg.env.loc_ref_enabled = False
    cfg.env.base_ctrl_enabled = True
    cfg.env.fsm_enabled = True
    _configure_nominal_only(cfg, forward_cmd=0.10, horizon=32)
    assert cfg.ppo.num_envs == 1
    assert cfg.ppo.rollout_steps == 32
    assert abs(cfg.env.min_velocity - 0.10) < 1e-9
    assert abs(cfg.env.max_velocity - 0.10) < 1e-9
    assert cfg.env.loc_ref_enabled is True
    assert cfg.env.controller_stack == "ppo"
    assert cfg.env.base_ctrl_enabled is False
    assert cfg.env.fsm_enabled is False
    assert cfg.env.push_enabled is False
    assert cfg.env.action_delay_steps == 0
    assert abs(cfg.env.action_filter_alpha - 0.3) < 1e-9


def test_configure_nominal_only_can_disable_action_filter() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0193a.yaml")
    cfg.env.action_filter_alpha = 0.3
    _configure_nominal_only(
        cfg,
        forward_cmd=0.10,
        horizon=32,
        disable_action_filter=True,
    )
    assert abs(cfg.env.action_filter_alpha - 0.0) < 1e-9


def test_configure_nominal_only_can_override_startup_target_rate() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0193a.yaml")
    _configure_nominal_only(
        cfg,
        forward_cmd=0.10,
        horizon=32,
        startup_target_rate_deg_s=10.0,
    )
    assert abs(cfg.env.loc_ref_v2_startup_target_rate_design_rad_s - float(jnp.deg2rad(10.0))) < 1e-8


def test_command_path_tracker_reports_startup_early_late_support() -> None:
    tracker = _CommandPathTracker(ctrl_dt=0.02, support_entry_window_s=0.12)

    def _posture(scale: float) -> dict[str, float]:
        return {
            "hip_roll_ref": 0.10 * scale,
            "hip_roll_ctrl": 0.10 * scale,
            "hip_roll_act": 0.08 * scale,
            "hip_pitch_ref": 0.20 * scale,
            "hip_pitch_ctrl": 0.20 * scale,
            "hip_pitch_act": 0.18 * scale,
            "knee_pitch_ref": 0.30 * scale,
            "knee_pitch_ctrl": 0.30 * scale,
            "knee_pitch_act": 0.05 * scale,
            "ankle_pitch_ref": -0.10 * scale,
            "ankle_pitch_ctrl": -0.10 * scale,
            "ankle_pitch_act": -0.06 * scale,
        }

    tracker.update(step=1, mode_id=0, mode_time_s=0.02, posture=_posture(1.0))
    p2 = _posture(1.1)
    p2["knee_pitch_ctrl"] = p2["knee_pitch_ref"] + 0.15  # induce target->ctrl divergence
    tracker.update(step=2, mode_id=0, mode_time_s=0.04, posture=p2)
    tracker.update(step=3, mode_id=1, mode_time_s=0.05, posture=_posture(0.9))
    tracker.update(step=4, mode_id=1, mode_time_s=0.16, posture=_posture(0.8))

    summary = tracker.get_summary()
    assert "Phase: STARTUP_SUPPORT_RAMP" in summary
    assert "Phase: EARLY_SUPPORT_STABILIZE" in summary
    assert "Phase: LATE_SUPPORT_STABILIZE" in summary
    assert "target step Δ mean/max" in summary
    assert "first_divergence target->ctrl=2" in summary


def test_format_step_line_includes_lateral_and_hip_roll_diagnostics() -> None:
    class _WR:
        velocity_cmd = 0.10

    class _State:
        info = {WR_INFO_KEY: _WR()}

    vec = jnp.zeros((len(METRIC_INDEX),), dtype=jnp.float32)
    vec = vec.at[METRIC_INDEX["tracking/loc_ref_stance_foot"]].set(1.0)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_left_foot_y_actual"]].set(0.09)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_right_foot_y_actual"]].set(-0.08)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_support_width_y_actual"]].set(0.17)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_support_width_y_nominal"]].set(0.08)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_support_width_y_commanded"]].set(0.12)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_base_support_y"]].set(-0.08)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_lateral_release_y"]].set(0.08)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_swing_y_target"]].set(0.12)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_swing_y_actual"]].set(-0.05)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_swing_y_error"]].set(-0.17)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_pelvis_roll_target"]].set(0.02)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_hip_roll_left_target"]].set(-0.01)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_hip_roll_right_target"]].set(0.01)

    line = _format_step_line(3, _State(), vec)
    assert "stance=1" in line
    assert "width_act=0.170" in line
    assert "width_ref=0.080" in line
    assert "width_cmd=0.120" in line
    assert "base_y=-0.080" in line
    assert "release_y=+0.080" in line
    assert "swing_y_tgt=+0.120" in line
    assert "swing_y_act=-0.050" in line
    assert "swing_y_err=-0.170" in line
    assert "pelvis_roll_tgt=+0.020" in line
    assert "lhr_tgt=-0.010" in line
    assert "rhr_tgt=+0.010" in line


def test_extract_lateral_semantics_and_reset_line() -> None:
    vec = jnp.zeros((len(METRIC_INDEX),), dtype=jnp.float32)
    vec = vec.at[METRIC_INDEX["tracking/loc_ref_stance_foot"]].set(0.0)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_left_foot_y_actual"]].set(0.085)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_right_foot_y_actual"]].set(-0.086)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_support_width_y_actual"]].set(0.171)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_support_width_y_nominal"]].set(0.080)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_support_width_y_commanded"]].set(0.000)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_base_support_y"]].set(-0.080)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_lateral_release_y"]].set(0.080)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_swing_y_target"]].set(0.0)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_swing_y_actual"]].set(-0.171)
    vec = vec.at[METRIC_INDEX["debug/loc_ref_swing_y_error"]].set(-0.171)

    sem = _extract_lateral_semantics(vec)
    assert abs(sem["width_actual"] - 0.171) < 1e-6
    assert abs(sem["width_reference"] - 0.080) < 1e-6
    assert abs(sem["width_commanded"] - 0.000) < 1e-6
    assert abs(sem["base_support_y"] + 0.080) < 1e-6
    assert abs(sem["lateral_release_y"] - 0.080) < 1e-6
    assert abs(sem["swing_y_target"] - (sem["base_support_y"] + sem["lateral_release_y"])) < 1e-6
    assert abs(sem["width_reference"] - abs(sem["base_support_y"])) < 1e-6
    assert abs(sem["width_commanded"] - abs(sem["swing_y_target"])) < 1e-6

    line = _format_reset_lateral_line(vec)
    assert "reset lateral geometry" in line
    assert "width_act=0.171" in line
    assert "width_ref=0.080" in line
    assert "width_cmd=0.000" in line


def test_extract_reset_lateral_semantics_uses_grounded_state_values() -> None:
    class _WR:
        loc_ref_stance_foot = jnp.asarray(1, dtype=jnp.int32)
        loc_ref_next_foothold = jnp.asarray([0.02, 0.08], dtype=jnp.float32)
        loc_ref_swing_pos = jnp.asarray([0.03, 0.10, 0.00], dtype=jnp.float32)

    class _Cal:
        @staticmethod
        def get_foot_positions(_data, normalize=False, frame=None):
            assert normalize is False
            left = jnp.asarray([0.0, 0.20, 0.01], dtype=jnp.float32)
            right = jnp.asarray([0.0, -0.05, 0.00], dtype=jnp.float32)
            return left, right

    class _State:
        info = {WR_INFO_KEY: _WR()}
        data = object()
        metrics = {
            METRICS_VEC_KEY: jnp.zeros((len(METRIC_INDEX),), dtype=jnp.float32).at[
                METRIC_INDEX["tracking/loc_ref_stance_foot"]
            ].set(0.0)
        }

    class _Env:
        _cal = _Cal()

    sem = _extract_reset_lateral_semantics(_Env(), _State())
    assert sem["stance"] == 1.0
    assert abs(sem["left_y"] - 0.20) < 1e-6
    assert abs(sem["right_y"] + 0.05) < 1e-6
    assert abs(sem["width_actual"] - 0.25) < 1e-6
    assert abs(sem["width_reference"] - 0.08) < 1e-6
    assert abs(sem["width_commanded"] - 0.10) < 1e-6
    assert abs(sem["swing_y_actual"] - 0.25) < 1e-6
    assert abs(sem["swing_y_error"] - 0.15) < 1e-6
    line = _format_reset_lateral_line_from_state(_Env(), _State())
    assert "stance=1" in line
    assert "width_act=0.250" in line


def test_format_support_posture_line_reports_joint_errors() -> None:
    class _RootPose:
        height = 0.45

        def euler_angles(self):
            return (0.12, 0.34, 0.0)

    class _Cal:
        @staticmethod
        def get_root_pose(_data):
            return _RootPose()

    class _WR:
        nominal_q_ref = jnp.asarray([0.10, 0.20, 0.30, 0.40], dtype=jnp.float32)

    class _State:
        info = {WR_INFO_KEY: _WR()}
        data = type("D", (), {
            "qpos": jnp.asarray([1.0, 0.11, 0.22, 0.33, 0.44], dtype=jnp.float32),
            "ctrl": jnp.asarray([0.10, 0.20, 0.30, 0.40], dtype=jnp.float32),
        })()

    class _Env:
        _cal = _Cal()
        _actuator_qpos_addrs = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
        _idx_left_hip_roll = 0
        _idx_left_hip_pitch = 1
        _idx_left_knee_pitch = 2
        _idx_left_ankle_pitch = 3
        _idx_right_hip_roll = 0
        _idx_right_hip_pitch = 1
        _idx_right_knee_pitch = 2
        _idx_right_ankle_pitch = 3

    vec = jnp.zeros((len(METRIC_INDEX),), dtype=jnp.float32)
    vec = vec.at[METRIC_INDEX["tracking/loc_ref_stance_foot"]].set(0.0)
    pose = _extract_support_posture(_Env(), _State(), vec)
    assert pose["stance_leg"] == "L"
    assert abs(pose["root_roll"] - 0.12) < 1e-6
    assert abs(pose["root_height"] - 0.45) < 1e-6
    assert abs(pose["hip_pitch_ref_err"] - 0.02) < 1e-6  # Updated field name
    line = _format_support_posture_line(_Env(), _State(), vec)
    assert "support-posture" in line
    assert "leg=L" in line
    assert "root_roll=+0.120" in line
    assert "hp_ref=+0.200" in line  # Check ref instead of err


def test_enable_temp_logging_returns_path_and_tees_output() -> None:
    fake_path = Path("/tmp/wildrobot_nominal_ref_123.log")
    fake_file = io.StringIO()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    with (
        patch("training.eval.visualize_nominal_ref.tempfile.gettempdir", return_value="/tmp"),
        patch("training.eval.visualize_nominal_ref.time.time", return_value=123),
        patch("builtins.open", return_value=fake_file),
        patch("training.eval.visualize_nominal_ref.atexit.register"),
        patch.object(sys, "argv", ["viewer.py", "--log"]),
    ):
        path = _enable_temp_logging()
        try:
            print("hello-log")
            sys.stderr.write("stderr-log\n")
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
    assert path == fake_path
    content = fake_file.getvalue()
    assert "log file: /tmp/wildrobot_nominal_ref_123.log" in content
    assert "command: viewer.py --log" in content
    assert "hello-log" in content
    assert "stderr-log" in content


def test_print_log_only_skips_primary_stdout_and_writes_secondary() -> None:
    primary = io.StringIO()
    secondary = io.StringIO()
    orig_stdout = sys.stdout
    try:
        from training.eval.visualize_nominal_ref import _TeeTextIO

        sys.stdout = _TeeTextIO(primary, secondary)
        _print_log_only("step=0001 hidden-from-screen")
    finally:
        sys.stdout = orig_stdout
    assert primary.getvalue() == ""
    assert "step=0001 hidden-from-screen" in secondary.getvalue()


def test_format_grounded_nominal_init_line_includes_geometry_fields() -> None:
    line = _format_grounded_nominal_init_line(
        {
            "stance_foot": 1.0,
            "root_height": 0.401,
            "root_dz_applied": -0.1234,
            "stance_support_z_before": 0.1234,
            "stance_foot_z": 0.0,
            "left_support_z": 0.005,
            "right_support_z": 0.0,
        }
    )
    assert "nominal_qref init grounded" in line
    assert "stance=1" in line
    assert "root_h=0.401" in line
    assert "root_dz=-0.1234" in line
    assert "stance_support_z=+0.0000" in line


def test_replace_state_with_nominal_qref_overwrites_actuated_qpos() -> None:
    class _WR:
        nominal_q_ref = jnp.asarray([0.11, 0.22], dtype=jnp.float32)

    class _Data:
        qpos = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
        ctrl = jnp.asarray([0.0, 0.0], dtype=jnp.float32)

        def replace(self, **kwargs):
            obj = _Data()
            obj.qpos = kwargs.get("qpos", self.qpos)
            obj.ctrl = kwargs.get("ctrl", self.ctrl)
            return obj

    class _State:
        info = {WR_INFO_KEY: _WR()}
        data = _Data()

        def replace(self, **kwargs):
            obj = _State()
            obj.info = kwargs.get("info", self.info)
            obj.data = kwargs.get("data", self.data)
            return obj

    class _Env:
        _actuator_qpos_addrs = jnp.asarray([1, 2], dtype=jnp.int32)
        _mjx_model = object()

    with patch("training.eval.visualize_nominal_ref.mujoco.mjx.forward", side_effect=lambda _m, d: d):
        out = _replace_state_with_nominal_qref(_Env(), _State())
    assert jnp.allclose(out.data.qpos, jnp.asarray([1.0, 0.11, 0.22], dtype=jnp.float32))
    assert jnp.allclose(out.data.ctrl, jnp.asarray([0.11, 0.22], dtype=jnp.float32))


def test_replace_state_with_nominal_qref_grounded_shifts_root_to_stance_contact() -> None:
    class _RootSpec:
        qpos_addr = 0

    class _FootSpec:
        def __init__(self, toe_geom_id, heel_geom_id):
            self.toe_geom_id = toe_geom_id
            self.heel_geom_id = heel_geom_id

    class _Cal:
        root_spec = _RootSpec()
        _foot_specs = [_FootSpec(0, 1), _FootSpec(2, 3)]

        @staticmethod
        def get_foot_positions(data, normalize=False):
            assert normalize is False
            left = jnp.asarray([0.0, 0.0, data.qpos[2] - 0.10], dtype=jnp.float32)
            right = jnp.asarray([0.0, 0.0, data.qpos[2] - 0.20], dtype=jnp.float32)
            return left, right

        @staticmethod
        def get_root_height(data):
            return data.qpos[2]

    class _WR:
        nominal_q_ref = jnp.asarray([0.11, 0.22], dtype=jnp.float32)
        loc_ref_stance_foot = jnp.asarray(1, dtype=jnp.int32)

    class _Data:
        qpos = jnp.asarray([0.0, 0.0, 0.50, 0.0, 0.0], dtype=jnp.float32)
        ctrl = jnp.asarray([0.0, 0.0], dtype=jnp.float32)

        def replace(self, **kwargs):
            obj = _Data()
            obj.qpos = kwargs.get("qpos", self.qpos)
            obj.ctrl = kwargs.get("ctrl", self.ctrl)
            return obj

    class _State:
        info = {WR_INFO_KEY: _WR()}
        data = _Data()

        def replace(self, **kwargs):
            obj = _State()
            obj.info = kwargs.get("info", self.info)
            obj.data = kwargs.get("data", self.data)
            return obj

    class _Env:
        _actuator_qpos_addrs = jnp.asarray([3, 4], dtype=jnp.int32)
        _mjx_model = object()
        _mj_model = object()
        _cal = _Cal()

    with (
        patch("training.eval.visualize_nominal_ref.mjx.forward", side_effect=lambda _m, d: d),
        patch(
            "training.eval.visualize_nominal_ref._compute_support_bottom_zs",
            side_effect=[(0.40, 0.30), (0.10, 0.0)],
        ),
    ):
        grounded_state, geometry = _replace_state_with_nominal_qref_grounded(_Env(), _State())
    assert jnp.allclose(grounded_state.data.qpos[3:], jnp.asarray([0.11, 0.22], dtype=jnp.float32))
    assert abs(float(grounded_state.data.qpos[2]) - 0.20) < 1e-6
    assert abs(float(geometry["stance_foot_z"])) < 1e-6
    assert abs(float(geometry["root_dz_applied"]) + 0.30) < 1e-6
    assert abs(float(geometry["stance_support_z_before"]) - 0.30) < 1e-6


def test_check_tracking_threshold() -> None:
    """Test threshold rule for joint tracking."""
    from training.eval.visualize_nominal_ref import _check_tracking_threshold
    
    # Test abs_floor dominates for small targets
    assert not _check_tracking_threshold(0.00, 0.10, abs_floor=0.05, frac_tol=0.10)  # err=0.10 > max(0.05, 0.01)=0.05
    assert _check_tracking_threshold(0.06, 0.10, abs_floor=0.05, frac_tol=0.10)  # err=0.04 < 0.05
    
    # Test frac_tol dominates for large targets
    assert _check_tracking_threshold(0.90, 1.00, abs_floor=0.05, frac_tol=0.10)  # err=0.10 == max(0.05, 0.10)=0.10
    assert not _check_tracking_threshold(0.85, 1.00, abs_floor=0.05, frac_tol=0.10)  # err=0.15 > 0.10
    
    # Test exact threshold boundary
    assert _check_tracking_threshold(0.95, 1.00, abs_floor=0.05, frac_tol=0.10)  # err=0.05 < 0.10
    assert _check_tracking_threshold(1.00, 1.00, abs_floor=0.05, frac_tol=0.10)  # err=0.00


def test_is_stable_window() -> None:
    """Test stable window detection."""
    from training.eval.visualize_nominal_ref import _is_stable_window
    
    # Not enough history
    assert not _is_stable_window([0], [0.1], [0.5], min_dwell=5)
    
    # Mode changed recently
    assert not _is_stable_window([0, 0, 0, 1, 1], [0.1] * 5, [0.5] * 5, min_dwell=5)
    
    # Pitch too high
    assert not _is_stable_window([0] * 5, [0.35, 0.35, 0.35, 0.35, 0.35], [0.5] * 5, min_dwell=5, pitch_thresh=0.30)
    
    # Pitch rate too high
    assert not _is_stable_window([0] * 5, [0.1] * 5, [2.0, 2.0, 2.0, 2.0, 2.0], min_dwell=5, pitch_rate_thresh=1.5)
    
    # All conditions met
    assert _is_stable_window([0] * 5, [0.1] * 5, [0.5] * 5, min_dwell=5, pitch_thresh=0.30, pitch_rate_thresh=1.5)
