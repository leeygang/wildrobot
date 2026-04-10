from __future__ import annotations

import jax.numpy as jnp

from control.references.walking_ref_v2 import (
    StartupRouteReason,
    StartupRouteStage,
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


def test_debug_force_support_only_keeps_mode_time_progressing_in_support() -> None:
    cfg = WalkingRefV2Config(debug_force_support_only=True)
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
        phase_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.10, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(True),
        dt_s=0.02,
    )
    assert int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE)
    assert abs(float(phase_time)) <= 1e-6
    assert abs(float(mode_time) - 0.12) <= 1e-6


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
            root_height_m=jnp.asarray(
                cfg.nominal_com_height_m + cfg.support_pelvis_height_offset_m,
                dtype=jnp.float32,
            ),
            left_foot_loaded=jnp.asarray(True),
            right_foot_loaded=jnp.asarray(False),
            dt_s=0.02,
        )
        observed_modes.append(int(mode_id))

    assert all(
        m in (int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), int(WalkingRefV2Mode.SUPPORT_STABILIZE))
        for m in observed_modes
    )
    assert int(WalkingRefV2Mode.SUPPORT_STABILIZE) == observed_modes[-1]


def test_startup_phase_keeps_release_suppressed_and_interpolation_bounded() -> None:
    cfg = WalkingRefV2Config(
        startup_ramp_s=0.20,
        startup_pelvis_height_offset_m=0.04,
        support_pelvis_height_offset_m=0.020,
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
        root_height_m=jnp.asarray(
            cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m,
            dtype=jnp.float32,
        ),
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

    startup_alpha = float(ref["startup_route_progress"])
    if startup_alpha <= cfg.startup_route_w1_alpha:
        startup_scale = cfg.startup_route_w1_scale * startup_alpha / cfg.startup_route_w1_alpha
    elif startup_alpha <= cfg.startup_route_w2_alpha:
        span = cfg.startup_route_w2_alpha - cfg.startup_route_w1_alpha
        t = (startup_alpha - cfg.startup_route_w1_alpha) / span
        startup_scale = cfg.startup_route_w1_scale + t * (
            cfg.startup_route_w2_scale - cfg.startup_route_w1_scale
        )
    elif startup_alpha <= cfg.startup_route_w3_alpha:
        span = cfg.startup_route_w3_alpha - cfg.startup_route_w2_alpha
        t = (startup_alpha - cfg.startup_route_w2_alpha) / span
        startup_scale = cfg.startup_route_w2_scale + t * (
            cfg.startup_route_w3_scale - cfg.startup_route_w2_scale
        )
    else:
        span = 1.0 - cfg.startup_route_w3_alpha
        t = (startup_alpha - cfg.startup_route_w3_alpha) / span
        startup_scale = cfg.startup_route_w3_scale + t * (1.0 - cfg.startup_route_w3_scale)
    assert abs(float(mode_time) - 0.02) <= 1e-6

    pelvis_height_startup = cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m
    pelvis_height_support = cfg.nominal_com_height_m + cfg.support_pelvis_height_offset_m
    expected_height = pelvis_height_startup + startup_scale * (
        pelvis_height_support - pelvis_height_startup
    )
    assert float(ref["pelvis_height"]) > cfg.nominal_com_height_m
    assert float(ref["pelvis_height"]) <= cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m + 1e-6
    assert abs(float(ref["pelvis_height"]) - expected_height) <= 1e-6


def test_startup_pelvis_realization_drives_progress() -> None:
    cfg = WalkingRefV2Config(startup_realization_lead_alpha=0.0)
    startup_height = cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m
    support_height = cfg.nominal_com_height_m + cfg.support_pelvis_height_offset_m

    ref_low, *_ = step_reference_v2_jax(
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
        root_height_m=jnp.asarray(startup_height, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )

    ref_high, *_ = step_reference_v2_jax(
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
        root_height_m=jnp.asarray(support_height, dtype=jnp.float32),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
    )

    low_dist = abs(float(ref_low["pelvis_height"]) - support_height)
    high_dist = abs(float(ref_high["pelvis_height"]) - support_height)
    assert high_dist < low_dist


def test_startup_readiness_governor_softly_slows_progress_without_freezing_timeout() -> None:
    cfg = WalkingRefV2Config(
        startup_progress_min_scale=0.20,
        startup_realization_lead_alpha=0.12,
    )
    (
        ref,
        _phase_time,
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
        root_height_m=jnp.asarray(
            cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m,
            dtype=jnp.float32,
        ),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
        stance_knee_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
        stance_ankle_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(mode_id) == int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP)
    assert abs(float(mode_time) - 0.02) <= 1e-6
    low_readiness_alpha = float(ref["startup_route_progress"])
    if low_readiness_alpha <= cfg.startup_route_w1_alpha:
        low_readiness_scale = cfg.startup_route_w1_scale * low_readiness_alpha / cfg.startup_route_w1_alpha
    elif low_readiness_alpha <= cfg.startup_route_w2_alpha:
        span = cfg.startup_route_w2_alpha - cfg.startup_route_w1_alpha
        t = (low_readiness_alpha - cfg.startup_route_w1_alpha) / span
        low_readiness_scale = cfg.startup_route_w1_scale + t * (
            cfg.startup_route_w2_scale - cfg.startup_route_w1_scale
        )
    elif low_readiness_alpha <= cfg.startup_route_w3_alpha:
        span = cfg.startup_route_w3_alpha - cfg.startup_route_w2_alpha
        t = (low_readiness_alpha - cfg.startup_route_w2_alpha) / span
        low_readiness_scale = cfg.startup_route_w2_scale + t * (
            cfg.startup_route_w3_scale - cfg.startup_route_w2_scale
        )
    else:
        span = 1.0 - cfg.startup_route_w3_alpha
        t = (low_readiness_alpha - cfg.startup_route_w3_alpha) / span
        low_readiness_scale = cfg.startup_route_w3_scale + t * (
            1.0 - cfg.startup_route_w3_scale
        )
    pelvis_height_startup = cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m
    pelvis_height_support = cfg.nominal_com_height_m + cfg.support_pelvis_height_offset_m
    expected_height = pelvis_height_startup + low_readiness_scale * (
        pelvis_height_support - pelvis_height_startup
    )
    assert abs(float(ref["pelvis_height"]) - expected_height) <= 1e-6


def test_startup_handoff_requires_readiness_not_time_only() -> None:
    cfg = WalkingRefV2Config(
        startup_handoff_min_alpha=0.85,
        startup_handoff_min_readiness=0.60,
        startup_handoff_min_pelvis_realization=0.60,
        startup_handoff_timeout_s=1.0,
    )
    (
        _ref,
        _phase_time,
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
        mode_time_s=jnp.asarray(0.19, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        root_height_m=jnp.asarray(
            cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m,
            dtype=jnp.float32,
        ),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(False),
        dt_s=0.02,
        stance_knee_tracking_error_rad=jnp.asarray(0.9, dtype=jnp.float32),
        stance_ankle_tracking_error_rad=jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert int(mode_id) == int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP)
    assert abs(float(mode_time) - 0.21) <= 1e-6


def test_startup_timeout_fallback_prevents_permanent_startup_lock() -> None:
    cfg = WalkingRefV2Config(
        startup_ramp_s=0.20,
        startup_handoff_min_readiness=0.95,
        startup_handoff_timeout_s=0.06,
        startup_progress_min_scale=0.20,
    )
    phase_time = jnp.asarray(0.0, dtype=jnp.float32)
    stance_foot = jnp.asarray(0, dtype=jnp.int32)
    switch_count = jnp.asarray(0, dtype=jnp.int32)
    mode_id = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    mode_time = jnp.asarray(0.0, dtype=jnp.float32)

    transitioned = False
    for _ in range(64):
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
            forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
            com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
            root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
            left_foot_loaded=jnp.asarray(True),
            right_foot_loaded=jnp.asarray(True),
            dt_s=0.02,
            stance_knee_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
            stance_ankle_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
        )
        if int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE):
            transitioned = True
            break

    assert transitioned is True


def test_startup_double_support_does_not_stall_with_default_progress_floor() -> None:
    cfg = WalkingRefV2Config()
    phase_time = jnp.asarray(0.0, dtype=jnp.float32)
    stance_foot = jnp.asarray(0, dtype=jnp.int32)
    switch_count = jnp.asarray(0, dtype=jnp.int32)
    mode_id = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    mode_time = jnp.asarray(0.0, dtype=jnp.float32)

    transitioned = False
    for _ in range(64):
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
            forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
            com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
            root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
            left_foot_loaded=jnp.asarray(True),
            right_foot_loaded=jnp.asarray(True),  # realistic startup double-support
            dt_s=0.02,
            stance_knee_tracking_error_rad=jnp.asarray(0.0, dtype=jnp.float32),
            stance_ankle_tracking_error_rad=jnp.asarray(0.0, dtype=jnp.float32),
        )
        if int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE):
            transitioned = True
            break

    assert transitioned is True


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


def test_startup_route_stage_metadata_progresses_to_b_before_handoff() -> None:
    cfg = WalkingRefV2Config(startup_ramp_s=0.20, startup_handoff_timeout_s=1.0)
    phase_time = jnp.asarray(0.0, dtype=jnp.float32)
    stance_foot = jnp.asarray(0, dtype=jnp.int32)
    switch_count = jnp.asarray(0, dtype=jnp.int32)
    mode_id = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    mode_time = jnp.asarray(0.0, dtype=jnp.float32)
    support_height = jnp.asarray(
        cfg.nominal_com_height_m + cfg.support_pelvis_height_offset_m, dtype=jnp.float32
    )

    stage_ids: list[int] = []
    for _ in range(32):
        (
            ref,
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
            forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
            com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
            root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
            root_height_m=support_height,
            left_foot_loaded=jnp.asarray(True),
            right_foot_loaded=jnp.asarray(False),
            dt_s=0.02,
            stance_knee_tracking_error_rad=jnp.asarray(0.0, dtype=jnp.float32),
            stance_ankle_tracking_error_rad=jnp.asarray(0.0, dtype=jnp.float32),
        )
        stage_id = int(ref["startup_route_stage_id"])
        stage_ids.append(stage_id)
        assert 0.0 <= float(ref["startup_route_progress"]) <= 1.0
        assert 0.0 <= float(ref["startup_route_ceiling"]) <= 1.0
        if int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE):
            break

    assert stage_ids
    assert all(int(StartupRouteStage.A) <= sid <= int(StartupRouteStage.B) for sid in stage_ids)
    assert int(StartupRouteStage.B) in stage_ids
    assert all(stage_ids[i] <= stage_ids[i + 1] for i in range(len(stage_ids) - 1))


def test_startup_route_timeout_transition_reports_timeout_reason() -> None:
    cfg = WalkingRefV2Config(
        startup_handoff_timeout_s=0.02,
        startup_handoff_min_readiness=0.95,
        startup_handoff_min_alpha=0.95,
    )
    (
        ref,
        _phase_time,
        _stance_foot,
        _switch_count,
        mode_id,
        _mode_time,
        _health,
        _instability,
        _permission,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.03, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        root_height_m=jnp.asarray(
            cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m,
            dtype=jnp.float32,
        ),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(True),
        dt_s=0.02,
        stance_knee_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
        stance_ankle_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(mode_id) == int(WalkingRefV2Mode.SUPPORT_STABILIZE)
    assert int(ref["startup_route_transition_reason"]) == int(StartupRouteReason.TIMEOUT_FALLBACK)


def test_startup_route_state_is_not_forced_to_b_after_timeout_handoff() -> None:
    cfg = WalkingRefV2Config(
        startup_handoff_timeout_s=0.02,
        startup_handoff_min_readiness=0.95,
        startup_handoff_min_alpha=0.95,
    )
    (
        ref_1,
        phase_time_1,
        stance_1,
        switches_1,
        mode_1,
        mode_time_1,
        _health_1,
        _instability_1,
        _permission_1,
    ) = step_reference_v2_jax(
        config=cfg,
        phase_time_s=jnp.asarray(0.0, dtype=jnp.float32),
        stance_foot_id=jnp.asarray(0, dtype=jnp.int32),
        stance_switch_count=jnp.asarray(0, dtype=jnp.int32),
        mode_id=jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32),
        mode_time_s=jnp.asarray(0.03, dtype=jnp.float32),
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        root_height_m=jnp.asarray(
            cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m,
            dtype=jnp.float32,
        ),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(True),
        dt_s=0.02,
        stance_knee_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
        stance_ankle_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(mode_1) == int(WalkingRefV2Mode.SUPPORT_STABILIZE)
    assert int(ref_1["startup_route_stage_id"]) == int(StartupRouteStage.A)
    assert int(ref_1["startup_route_transition_reason"]) == int(StartupRouteReason.TIMEOUT_FALLBACK)

    ref_2, *_ = step_reference_v2_jax(
        config=cfg,
        phase_time_s=phase_time_1,
        stance_foot_id=stance_1,
        stance_switch_count=switches_1,
        mode_id=mode_1,
        mode_time_s=mode_time_1,
        forward_speed_mps=jnp.asarray(0.10, dtype=jnp.float32),
        com_position_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        com_velocity_stance_frame=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        root_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        root_pitch_rate_rad_s=jnp.asarray(0.0, dtype=jnp.float32),
        root_height_m=jnp.asarray(
            cfg.nominal_com_height_m + cfg.startup_pelvis_height_offset_m,
            dtype=jnp.float32,
        ),
        left_foot_loaded=jnp.asarray(True),
        right_foot_loaded=jnp.asarray(True),
        dt_s=0.02,
        stance_knee_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
        stance_ankle_tracking_error_rad=jnp.asarray(1.0, dtype=jnp.float32),
        startup_route_progress_prev=ref_1["startup_route_progress"],
        startup_route_ceiling_prev=ref_1["startup_route_ceiling"],
        startup_route_stage_id_prev=ref_1["startup_route_stage_id"],
        startup_route_transition_reason_prev=ref_1["startup_route_transition_reason"],
    )
    assert int(ref_2["startup_route_stage_id"]) == int(StartupRouteStage.A)
    assert int(ref_2["startup_route_transition_reason"]) == int(StartupRouteReason.TIMEOUT_FALLBACK)
