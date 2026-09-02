from __future__ import annotations

import numpy as np
import pytest

from training.eval.diagnose_roll_load_sharing import summarize_roll_load_sharing


JOINT_NAMES = (
    "left_hip_roll",
    "left_ankle_roll",
    "right_hip_roll",
    "right_ankle_roll",
)


def test_roll_load_summary_splits_support_and_excludes_startup() -> None:
    shape = (5, 1, 4)
    torque_nm = np.zeros(shape, dtype=np.float32)
    torque_nm[2, 0] = [9.6, 2.0, 1.0, 1.0]
    torque_nm[3, 0] = [1.0, 1.0, 4.0, 3.0]
    torque_nm[4, 0] = [8.0, 7.0, 6.0, 5.0]
    torque_ratio = np.abs(torque_nm) / 10.0
    policy_action = torque_nm / 10.0
    applied_action = policy_action / 2.0
    target_error_rad = torque_nm / 100.0
    com_to_left_foot_lateral_m = np.full((5, 1), -0.08, dtype=np.float32)
    com_to_right_foot_lateral_m = np.full((5, 1), 0.07, dtype=np.float32)
    left_loaded = np.asarray([[1], [1], [1], [0], [1]], dtype=np.float32)
    right_loaded = np.asarray([[1], [1], [0], [1], [1]], dtype=np.float32)
    dones = np.zeros((5, 1), dtype=np.float32)
    truncations = np.zeros_like(dones)

    result = summarize_roll_load_sharing(
        joint_names=JOINT_NAMES,
        torque_nm=torque_nm,
        torque_ratio=torque_ratio,
        policy_action=policy_action,
        applied_action=applied_action,
        target_error_rad=target_error_rad,
        com_to_left_foot_lateral_m=com_to_left_foot_lateral_m,
        com_to_right_foot_lateral_m=com_to_right_foot_lateral_m,
        left_loaded=left_loaded,
        right_loaded=right_loaded,
        dones=dones,
        truncations=truncations,
        ctrl_dt=1.0,
        robot_weight_n=40.0,
        stable_start_s=2.0,
        pre_fall_window_s=1.0,
    )

    stable = result["windows"]["stable_survivors"]
    assert stable["sample_count"] == 3
    assert stable["support_phases"]["left_only"]["sample_count"] == 1
    assert stable["support_phases"]["right_only"]["sample_count"] == 1
    assert stable["support_phases"]["double_support"]["sample_count"] == 1
    left_hip = stable["support_phases"]["left_only"]["joints"]["left_hip_roll"]
    assert left_hip["torque_abs_mean_nm"] == pytest.approx(9.6)
    assert left_hip["torque_saturation_frac"] == pytest.approx(1.0)
    assert left_hip["applied_action_abs_mean"] == pytest.approx(0.48)
    assert left_hip["target_error_abs_mean_rad"] == pytest.approx(0.096)
    leverage = stable["support_phases"]["left_only"]["com_to_loaded_foot"]
    assert leverage["lateral_lever_signed_mean_m"] == pytest.approx(-0.08)
    assert leverage["quasi_static_gravity_moment_abs_mean_nm"] == pytest.approx(3.2)


def test_roll_load_summary_uses_only_pre_terminal_pre_fall_samples() -> None:
    shape = (5, 1, 4)
    torque_nm = np.zeros(shape, dtype=np.float32)
    torque_nm[2, 0, 0] = 2.0
    torque_nm[3, 0, 0] = 9.6
    torque_nm[4, 0, 0] = 100.0
    torque_ratio = np.abs(torque_nm) / 10.0
    left_loaded = np.ones((5, 1), dtype=np.float32)
    right_loaded = np.zeros((5, 1), dtype=np.float32)
    dones = np.zeros((5, 1), dtype=np.float32)
    dones[4, 0] = 1.0
    truncations = np.zeros_like(dones)
    com_to_left_foot_lateral_m = np.full((5, 1), -0.08, dtype=np.float32)
    com_to_right_foot_lateral_m = np.full((5, 1), 0.07, dtype=np.float32)

    result = summarize_roll_load_sharing(
        joint_names=JOINT_NAMES,
        torque_nm=torque_nm,
        torque_ratio=torque_ratio,
        policy_action=np.zeros(shape, dtype=np.float32),
        applied_action=np.zeros(shape, dtype=np.float32),
        target_error_rad=np.zeros(shape, dtype=np.float32),
        com_to_left_foot_lateral_m=com_to_left_foot_lateral_m,
        com_to_right_foot_lateral_m=com_to_right_foot_lateral_m,
        left_loaded=left_loaded,
        right_loaded=right_loaded,
        dones=dones,
        truncations=truncations,
        ctrl_dt=1.0,
        robot_weight_n=40.0,
        stable_start_s=0.0,
        pre_fall_window_s=2.0,
    )

    assert result["failed_env_count"] == 1
    assert result["windows"]["stable_survivors"]["sample_count"] == 0
    pre_fall = result["windows"]["pre_fall"]
    assert pre_fall["sample_count"] == 2
    left_hip = pre_fall["support_phases"]["left_only"]["joints"]["left_hip_roll"]
    assert left_hip["torque_abs_mean_nm"] == pytest.approx(5.8)
    assert left_hip["torque_saturation_frac"] == pytest.approx(0.5)
