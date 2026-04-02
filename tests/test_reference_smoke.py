from __future__ import annotations

from tools.reference_smoke.run_reference_smoke import run_smoke


def test_reference_smoke_summary_fields_and_bounds() -> None:
    summary = run_smoke(
        robot_config_path="assets/v2/mujoco_robot_config.json",
        steps=120,
        dt_s=0.02,
        forward_speed_mps=0.12,
    )
    assert summary.steps == 120
    assert summary.stance_switches > 0
    assert summary.mid_swing_clearance_min_m >= 0.0
    assert summary.touchdown_xy_error_max_m < 0.2
    assert summary.touchdown_z_abs_max_m < 0.08
    assert summary.min_joint_limit_margin_rad > 0.01
    assert 0.0 <= summary.ik_unreachable_frac <= 1.0
