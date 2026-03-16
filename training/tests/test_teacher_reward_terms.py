# training/tests/test_teacher_reward_terms.py
"""Teacher reward helper tests for v0.16.0 motion-tracking branch."""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jp

from training.envs.teacher_rewards import (
    foot_position_contact_reward,
    joint_pose_tracking_reward,
    phase_consistency_reward,
    root_pose_tracking_reward,
    root_velocity_tracking_reward,
    upright_reward,
)


@pytest.mark.sim
def test_root_pose_tracking_reward_is_higher_when_closer():
    ref_pos = jp.asarray([0.0, 0.0, 0.45])
    ref_quat = jp.asarray([1.0, 0.0, 0.0, 0.0])
    r_good, pos_err_good, _ = root_pose_tracking_reward(
        root_pos=jp.asarray([0.01, 0.0, 0.45]),
        root_quat_wxyz=ref_quat,
        ref_root_pos=ref_pos,
        ref_root_quat_wxyz=ref_quat,
        pos_sigma=jp.asarray(0.10),
        quat_sigma=jp.asarray(0.20),
    )
    r_bad, pos_err_bad, _ = root_pose_tracking_reward(
        root_pos=jp.asarray([0.20, 0.0, 0.45]),
        root_quat_wxyz=jp.asarray([0.0, 1.0, 0.0, 0.0]),
        ref_root_pos=ref_pos,
        ref_root_quat_wxyz=ref_quat,
        pos_sigma=jp.asarray(0.10),
        quat_sigma=jp.asarray(0.20),
    )
    assert float(pos_err_good) < float(pos_err_bad)
    assert float(r_good) > float(r_bad)


@pytest.mark.sim
def test_root_velocity_tracking_reward_is_higher_when_matched():
    r_match, _, _ = root_velocity_tracking_reward(
        root_lin_vel=jp.asarray([0.10, 0.0, 0.0]),
        root_ang_vel=jp.asarray([0.0, 0.0, 0.0]),
        ref_root_lin_vel=jp.asarray([0.10, 0.0, 0.0]),
        ref_root_ang_vel=jp.asarray([0.0, 0.0, 0.0]),
        lin_sigma=jp.asarray(0.25),
        ang_sigma=jp.asarray(0.30),
    )
    r_miss, _, _ = root_velocity_tracking_reward(
        root_lin_vel=jp.asarray([0.0, 0.3, 0.0]),
        root_ang_vel=jp.asarray([0.8, 0.0, 0.0]),
        ref_root_lin_vel=jp.asarray([0.10, 0.0, 0.0]),
        ref_root_ang_vel=jp.asarray([0.0, 0.0, 0.0]),
        lin_sigma=jp.asarray(0.25),
        ang_sigma=jp.asarray(0.30),
    )
    assert float(r_match) > float(r_miss)


@pytest.mark.sim
def test_joint_pose_tracking_reward_decreases_with_error():
    ref = jp.asarray([0.1, -0.1, 0.2], dtype=jp.float32)
    r_close, err_close = joint_pose_tracking_reward(
        joint_pos=jp.asarray([0.11, -0.09, 0.21], dtype=jp.float32),
        ref_joint_pos=ref,
        sigma=jp.asarray(0.20),
    )
    r_far, err_far = joint_pose_tracking_reward(
        joint_pos=jp.asarray([0.4, -0.6, 0.8], dtype=jp.float32),
        ref_joint_pos=ref,
        sigma=jp.asarray(0.20),
    )
    assert float(err_close) < float(err_far)
    assert float(r_close) > float(r_far)


@pytest.mark.sim
def test_foot_contact_timing_reward_prefers_agreement():
    r_agree, _, contact_r_agree, contact_agree = foot_position_contact_reward(
        left_foot_pos=jp.asarray([0.0, 0.1, 0.0]),
        right_foot_pos=jp.asarray([0.0, -0.1, 0.0]),
        ref_left_foot_pos=jp.asarray([0.0, 0.1, 0.0]),
        ref_right_foot_pos=jp.asarray([0.0, -0.1, 0.0]),
        left_contact=jp.asarray(1.0),
        right_contact=jp.asarray(0.0),
        ref_left_contact=jp.asarray(1.0),
        ref_right_contact=jp.asarray(0.0),
        pos_sigma=jp.asarray(0.06),
    )
    r_miss, _, contact_r_miss, _ = foot_position_contact_reward(
        left_foot_pos=jp.asarray([0.0, 0.1, 0.0]),
        right_foot_pos=jp.asarray([0.0, -0.1, 0.0]),
        ref_left_foot_pos=jp.asarray([0.0, 0.1, 0.0]),
        ref_right_foot_pos=jp.asarray([0.0, -0.1, 0.0]),
        left_contact=jp.asarray(0.0),
        right_contact=jp.asarray(1.0),
        ref_left_contact=jp.asarray(1.0),
        ref_right_contact=jp.asarray(0.0),
        pos_sigma=jp.asarray(0.06),
    )
    assert float(contact_r_agree) > float(contact_r_miss)
    assert float(contact_agree) > 0.9
    assert float(r_agree) > float(r_miss)


@pytest.mark.sim
def test_phase_consistency_and_upright_reward():
    r_phase_good, e_good = phase_consistency_reward(
        current_phase=jp.asarray(0.25),
        reference_phase=jp.asarray(0.26),
        sigma=jp.asarray(0.10),
    )
    r_phase_bad, e_bad = phase_consistency_reward(
        current_phase=jp.asarray(0.25),
        reference_phase=jp.asarray(0.75),
        sigma=jp.asarray(0.10),
    )
    assert float(e_good) < float(e_bad)
    assert float(r_phase_good) > float(r_phase_bad)

    r_upright, tilt_upright = upright_reward(
        pitch=jp.asarray(0.03),
        roll=jp.asarray(0.04),
        sigma=jp.asarray(0.25),
    )
    r_tilted, tilt_tilted = upright_reward(
        pitch=jp.asarray(0.45),
        roll=jp.asarray(0.20),
        sigma=jp.asarray(0.25),
    )
    assert float(tilt_upright) < float(tilt_tilted)
    assert float(r_upright) > float(r_tilted)
    assert np.isfinite(float(r_upright))

