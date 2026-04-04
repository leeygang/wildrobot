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
