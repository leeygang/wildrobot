from __future__ import annotations

from control.kinematics import LegIkConfig, forward_leg_sagittal, solve_leg_sagittal_ik


def test_leg_ik_representative_target_solves() -> None:
    cfg = LegIkConfig(upper_leg_length_m=0.18, lower_leg_length_m=0.18)
    result = solve_leg_sagittal_ik(target_x_m=0.04, target_z_m=-0.30, config=cfg)
    x, z = forward_leg_sagittal(
        hip_pitch_rad=result.hip_pitch_rad,
        knee_pitch_rad=result.knee_pitch_rad,
        config=cfg,
    )
    assert result.reachable is True
    assert abs(x - 0.04) < 0.03
    assert abs(z - (-0.30)) < 0.03


def test_leg_ik_unreachable_target_clips_and_reports() -> None:
    cfg = LegIkConfig(upper_leg_length_m=0.18, lower_leg_length_m=0.18)
    result = solve_leg_sagittal_ik(target_x_m=0.6, target_z_m=-0.6, config=cfg)
    assert result.reachable is False
    assert abs(result.clipped_target_x_m) < 0.3
    assert abs(result.clipped_target_z_m) < 0.3
