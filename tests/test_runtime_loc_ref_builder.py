from __future__ import annotations

import numpy as np

from runtime.wr_runtime.control.loc_ref_runtime import RuntimeLocRefBuilder
from training.envs.wildrobot_env import (
    REF_LEFT_STANCE,
    REF_RIGHT_STANCE,
    _swing_state_in_stance_frame,
)


def test_runtime_loc_ref_builder_emits_wr_obs_v4_feature_shapes() -> None:
    builder = RuntimeLocRefBuilder(default_dt_s=0.02)
    f0 = builder.step(forward_speed_mps=0.12, dt_s=0.02)
    f1 = builder.step(forward_speed_mps=0.12, dt_s=0.02)

    assert f0.phase_sin_cos.shape == (2,)
    assert f0.stance_foot.shape == (1,)
    assert f0.next_foothold.shape == (2,)
    assert f0.swing_pos.shape == (3,)
    assert f0.swing_vel.shape == (3,)
    assert f0.pelvis_targets.shape == (3,)
    assert f0.history.shape == (4,)
    assert np.allclose(f1.history[:2], f0.phase_sin_cos, atol=1e-6)


def test_runtime_loc_ref_builder_uses_joint_state_for_velocity_estimate() -> None:
    names = ("left_hip_pitch", "left_knee_pitch", "right_hip_pitch", "right_knee_pitch")
    builder = RuntimeLocRefBuilder(default_dt_s=0.02, actuator_names=names)
    q = np.asarray([0.1, 0.2, 0.1, 0.2], dtype=np.float32)
    qd_zero = np.zeros((4,), dtype=np.float32)
    qd_move = np.asarray([0.8, 0.0, 0.8, 0.0], dtype=np.float32)

    f0 = builder.step(forward_speed_mps=0.0, dt_s=0.02, joint_pos_rad=q, joint_vel_rad_s=qd_zero)
    f1 = builder.step(forward_speed_mps=0.0, dt_s=0.02, joint_pos_rad=q, joint_vel_rad_s=qd_move)

    assert np.linalg.norm(f1.next_foothold - f0.next_foothold) > 1e-6


def test_swing_state_projection_uses_stance_frame() -> None:
    left_pos = np.asarray([0.0, 0.1, 0.0], dtype=np.float32)
    right_pos = np.asarray([0.2, -0.1, 0.05], dtype=np.float32)
    left_vel = np.asarray([0.01, 0.02, 0.0], dtype=np.float32)
    right_vel = np.asarray([0.03, -0.01, 0.01], dtype=np.float32)

    swing_pos_l, swing_vel_l = _swing_state_in_stance_frame(
        left_foot_pos_h=left_pos,
        right_foot_pos_h=right_pos,
        left_foot_vel_h=left_vel,
        right_foot_vel_h=right_vel,
        stance_foot_id=REF_LEFT_STANCE,
    )
    assert np.allclose(swing_pos_l, right_pos - left_pos, atol=1e-6)
    assert np.allclose(swing_vel_l, right_vel - left_vel, atol=1e-6)

    swing_pos_r, swing_vel_r = _swing_state_in_stance_frame(
        left_foot_pos_h=left_pos,
        right_foot_pos_h=right_pos,
        left_foot_vel_h=left_vel,
        right_foot_vel_h=right_vel,
        stance_foot_id=REF_RIGHT_STANCE,
    )
    assert np.allclose(swing_pos_r, left_pos - right_pos, atol=1e-6)
    assert np.allclose(swing_vel_r, left_vel - right_vel, atol=1e-6)
