from __future__ import annotations

from control.references import WalkingRefV1Config, WalkingRefV1Input, WalkingRefV1State, step_reference


def test_phase_progression_and_stance_switching() -> None:
    cfg = WalkingRefV1Config(step_time_s=0.2)
    state = WalkingRefV1State()
    stances = []
    for _ in range(30):
        out = step_reference(
            config=cfg,
            state=state,
            inputs=WalkingRefV1Input(forward_speed_mps=0.12),
            dt_s=0.02,
        )
        state = out.next_state
        stances.append(state.stance_foot_id)
    assert state.stance_switch_count >= 2
    assert 0 in stances and 1 in stances


def test_foothold_and_pelvis_bounds() -> None:
    cfg = WalkingRefV1Config(
        min_step_length_m=0.02,
        max_step_length_m=0.14,
        max_lateral_step_m=0.14,
        max_pelvis_pitch_rad=0.08,
    )
    out = step_reference(
        config=cfg,
        state=WalkingRefV1State(),
        inputs=WalkingRefV1Input(forward_speed_mps=2.0),
        dt_s=0.02,
    )
    ref = out.reference
    assert cfg.min_step_length_m <= ref.desired_next_foothold_stance_frame[0] <= cfg.max_step_length_m
    assert abs(ref.desired_next_foothold_stance_frame[1]) <= cfg.max_lateral_step_m
    assert abs(ref.desired_pelvis_pitch_rad) <= cfg.max_pelvis_pitch_rad


def test_swing_trajectory_clearance_and_velocity_present() -> None:
    cfg = WalkingRefV1Config(swing_height_m=0.05)
    state = WalkingRefV1State(phase_time_s=cfg.step_time_s * 0.5)
    out = step_reference(
        config=cfg,
        state=state,
        inputs=WalkingRefV1Input(forward_speed_mps=0.1),
        dt_s=0.01,
    )
    ref = out.reference
    assert ref.desired_swing_foot_position[2] >= 0.0
    assert abs(ref.desired_swing_foot_velocity[0]) > 0.0
