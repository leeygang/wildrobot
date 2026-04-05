from __future__ import annotations

import jax.numpy as jnp

from control.references.walking_ref_v2 import (
    WalkingRefV2Config,
    WalkingRefV2Mode,
    compute_support_health_v2_jax,
    step_reference_v2_jax,
)


def test_step_reference_v2_jax_mode_time_not_reset_when_condition_persists() -> None:
    cfg = WalkingRefV2Config(
        support_open_threshold=0.5,
        support_release_threshold=0.2,
        support_release_phase_start=0.0,
        touchdown_phase_min=0.95,
        capture_hold_s=0.04,
        settle_hold_s=0.04,
    )
    phase_time = jnp.asarray(0.0, dtype=jnp.float32)
    stance_foot = jnp.asarray(0, dtype=jnp.int32)
    switch_count = jnp.asarray(0, dtype=jnp.int32)
    mode_id = jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32)
    mode_time = jnp.asarray(0.0, dtype=jnp.float32)
    for i in range(3):
        (
            _ref,
            phase_time,
            stance_foot,
            switch_count,
            mode_id,
            mode_time,
            _health,
            _instability,
            _permission,
        ) = step_reference_v2_jax(
            config=cfg,
            phase_time_s=phase_time,
            stance_foot_id=stance_foot,
            stance_switch_count=switch_count,
            mode_id=mode_id,
            mode_time_s=mode_time,
            forward_speed_mps=jnp.asarray(0.08, dtype=jnp.float32),
            com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
            root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
            left_foot_loaded=jnp.asarray(True),
            right_foot_loaded=jnp.asarray(False),
            dt_s=0.02,
        )
        assert int(mode_id) == int(WalkingRefV2Mode.SWING_RELEASE)
        if i == 0:
            assert abs(float(mode_time) - 0.0) < 1e-6
        if i == 1:
            assert abs(float(mode_time) - 0.02) < 1e-6
        if i == 2:
            assert abs(float(mode_time) - 0.04) < 1e-6


def test_compute_support_health_v2_jax_bounded() -> None:
    cfg = WalkingRefV2Config()
    health, instability = compute_support_health_v2_jax(
        config=cfg,
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.18, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(1.2, dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.35, 0.0], dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(True),
    )
    assert 0.0 <= float(health) <= 1.0
    assert float(instability) >= 0.0


def test_support_mode_low_permission_keeps_base_width_and_zero_release() -> None:
    cfg = WalkingRefV2Config()
    (
        ref,
        _phase_time,
        _stance_foot,
        _switch_count,
        mode_id,
        _mode_time,
        _health,
        _instability,
        permission,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.20, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.08, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.30, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.30, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(2.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )
    assert int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE)
    assert float(permission) <= 1e-5
    assert abs(float(ref["swing_pos"][0])) <= 1e-6
    assert abs(float(ref["swing_vel"][0])) <= 1e-6
    base_support_y = float(ref["next_foothold"][1])
    lateral_release_y = float(ref["swing_pos"][1] - ref["next_foothold"][1])
    assert abs(float(ref["swing_pos"][1]) - base_support_y) <= 1e-6
    assert abs(lateral_release_y) <= 1e-6
    assert abs(abs(float(ref["swing_pos"][1])) - abs(base_support_y)) <= 1e-6
    assert abs(float(ref["swing_vel"][1])) <= 1e-6


def test_support_mode_high_permission_still_keeps_zero_lateral_release_before_phase_gate() -> None:
    cfg = WalkingRefV2Config(support_release_phase_start=0.90)
    (
        ref,
        _phase_time,
        _stance_foot,
        _switch_count,
        mode_id,
        _mode_time,
        _health,
        _instability,
        permission,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.20, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )
    assert int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE)
    assert float(permission) >= 0.99
    base_support_y = float(ref["next_foothold"][1])
    lateral_release_y = float(ref["swing_pos"][1] - ref["next_foothold"][1])
    assert abs(float(ref["swing_pos"][1]) - base_support_y) <= 1e-6
    assert abs(lateral_release_y) <= 1e-6
    assert abs(abs(float(ref["swing_pos"][1])) - abs(base_support_y)) <= 1e-6
    assert abs(float(ref["swing_vel"][1])) <= 1e-6


def test_swing_release_mode_applies_bounded_lateral_release_delta() -> None:
    cfg = WalkingRefV2Config(support_release_phase_start=0.20, max_lateral_release_m=0.015)
    (
        ref,
        _phase_time,
        _stance_foot,
        _switch_count,
        mode_id,
        _mode_time,
        _health,
        _instability,
        permission,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.30, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.SWING_RELEASE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )
    assert int(mode_id) == int(WalkingRefV2Mode.SWING_RELEASE)
    assert float(permission) >= 0.99
    phase = float(ref["phase_progress"])
    release_prog = max((phase - cfg.support_release_phase_start) / (1.0 - cfg.support_release_phase_start), 0.0)
    base = cfg.support_swing_progress_min_scale + (1.0 - cfg.support_swing_progress_min_scale) * release_prog
    expected_release = -cfg.max_lateral_release_m * float(permission) * base
    base_support_y = float(ref["next_foothold"][1])
    lateral_release_y = float(ref["swing_pos"][1] - ref["next_foothold"][1])
    assert abs(lateral_release_y - expected_release) <= 1e-5
    assert abs(lateral_release_y) <= cfg.max_lateral_release_m + 1e-6
    assert abs(float(ref["swing_pos"][1]) - (base_support_y + lateral_release_y)) <= 1e-6


def test_debug_force_support_only_prevents_release_and_phase_progress() -> None:
    cfg = WalkingRefV2Config(debug_force_support_only=True)
    (
        ref,
        phase_time,
        _stance_foot,
        _switch_count,
        mode_id,
        mode_time,
        _health,
        _instability,
        permission,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.49, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.SWING_RELEASE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.10, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )
    assert int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE)
    assert abs(float(phase_time)) <= 1e-6
    assert abs(float(mode_time)) <= 1e-6
    assert float(permission) >= 0.99
    assert abs(float(ref["phase_progress"])) <= 1e-6
    assert abs(float(ref["swing_pos"][0])) <= 1e-6
    assert abs(float(ref["swing_vel"][0])) <= 1e-6


def test_startup_mode_exists_and_transitions_to_support() -> None:
    cfg = WalkingRefV2Config(startup_ramp_s=0.06)
    phase_time = jnp.asarray(0.0, dtype=jnp.float32)
    stance_foot = jnp.asarray(0, dtype=jnp.int32)
    switch_count = jnp.asarray(0, dtype=jnp.int32)
    mode_id = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    mode_time = jnp.asarray(0.0, dtype=jnp.float32)

    observed_modes: list[int] = []
    for _ in range(4):
        (
            _ref,
            phase_time,
            stance_foot,
            switch_count,
            mode_id,
            mode_time,
            _health,
            _instability,
            _permission,
        ) = step_reference_v2_jax(
            config=cfg,
            phase_time_s=phase_time,
            stance_foot_id=stance_foot,
            stance_switch_count=switch_count,
            mode_id=mode_id,
            mode_time_s=mode_time,
            forward_speed_mps=jnp.asarray(0.08, dtype=jnp.float32),
            com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
            root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
            left_foot_loaded=jnp.asarray(True),
            right_foot_loaded=jnp.asarray(False),
            dt_s=0.02,
        )
        observed_modes.append(int(mode_id))

    assert int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP) in observed_modes
    assert int(WalkingRefV2Mode.SUPPORT_STABILIZE) == observed_modes[-1]


def test_startup_phase_keeps_release_suppressed_and_interpolation_bounded() -> None:
    cfg = WalkingRefV2Config(
        startup_ramp_s=0.20,
        startup_pelvis_height_offset_m=0.04,
        support_release_phase_start=0.0,
    )
    (
        ref,
        phase_time,
        _stance_foot,
        _switch_count,
        mode_id,
        mode_time,
        _health,
        _instability,
        _permission,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )

    assert int(mode_id) == int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP)
    assert abs(float(phase_time)) <= 1e-6
    assert abs(float(ref["phase_progress"])) <= 1e-6
    assert abs(float(ref["swing_pos"][0])) <= 1e-6
    assert abs(float(ref["swing_vel"][0])) <= 1e-6
    assert abs(float(ref["swing_pos"][1] - ref["next_foothold"][1])) <= 1e-6
    assert abs(float(ref["swing_vel"][1])) <= 1e-6

    startup_alpha = float(mode_time) / cfg.startup_ramp_s
    assert 0.0 <= startup_alpha <= 1.0

    expected_height = cfg.nominal_com_height_m + (1.0 - startup_alpha) * cfg.startup_pelvis_height_offset_m
    assert float(ref["pelvis_height"]) > cfg.nominal_com_height_m
    assert float(ref["pelvis_height"]) <= cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m + 1e-6
    assert abs(float(ref["pelvis_height"]) - expected_height) <= 1e-6


def test_support_stabilize_foothold_clamp_uses_config() -> None:
    cfg = WalkingRefV2Config(
        support_foothold_min_scale=0.0,
        support_stabilize_max_foothold_scale=0.20,
        min_step_length_m=0.01,
        max_step_length_m=0.10,
        step_time_s=0.5,
        dcm_correction_gain=0.0,
    )
    ref, *_ = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.20, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.30, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )
    nominal_step = min(max(float(0.30 * cfg.step_time_s), cfg.min_step_length_m), cfg.max_step_length_m)
    expected_x = cfg.min_step_length_m + cfg.support_stabilize_max_foothold_scale * (
        nominal_step - cfg.min_step_length_m
    )
    assert abs(float(ref["next_foothold"][0]) - expected_x) <= 1e-6


def test_post_settle_swing_scale_uses_config() -> None:
    cfg = WalkingRefV2Config(post_settle_swing_scale=0.35)
    ref, *_rest, permission = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.20, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )
    expected_release = cfg.post_settle_swing_scale * float(permission)
    expected_x = float(ref["next_foothold"][0]) * expected_release
    assert abs(float(ref["swing_pos"][0]) - expected_x) <= 1e-6
