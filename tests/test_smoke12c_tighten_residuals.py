"""Pin smoke12c: tighten smoke12b's widened leg-pitch residual authority.

smoke12b finally produced forward stepping, but the late run became
asymmetric and crossed the G5 residual knee cap. smoke12c keeps the
smoke12b basin-break recipe intact and changes one thing only:
reduce the widened hip/knee pitch scales from the smoke12b 75%-coverage
values toward a 60%-coverage correction regime.

These tests pin that contract:
  1. smoke12c loads and keeps smoke12b's basin / cmd / reward recipe.
  2. Only the four hip/knee pitch per-joint scales change vs smoke12b.
  3. The new scales are strictly tighter than smoke12b while staying
     above the legacy 0.25 floor on the widened joints.
  4. Env build + zero-action-under-home contract is preserved.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jp
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE12B_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12b.yaml"
_SMOKE12C_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12c.yaml"

_TIGHTENED_JOINTS = (
    "left_hip_pitch", "left_knee_pitch",
    "right_hip_pitch", "right_knee_pitch",
)
_UNCHANGED_JOINTS = (
    "left_hip_roll", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_roll", "right_ankle_pitch", "right_ankle_roll",
)


def _load_cfg(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from training.configs.training_config import load_training_config
    return load_training_config(str(path))


def _maybe_build_env(cfg_path: Path):
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path.name} not found")
    try:
        from assets.robot_config import load_robot_config
        from training.configs.training_config import load_training_config
        from training.envs.wildrobot_env import WildRobotEnv
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"env deps unavailable: {exc}")
    load_robot_config(str(_REPO_ROOT / "assets/v2/mujoco_robot_config.json"))
    cfg = load_training_config(str(cfg_path))
    cfg.freeze()
    return WildRobotEnv(cfg)


def test_smoke12c_loads_and_keeps_smoke12b_basin_break_recipe() -> None:
    s12b = _load_cfg(_SMOKE12B_CFG)
    s12c = _load_cfg(_SMOKE12C_CFG)

    assert s12c.env.actor_obs_layout_id == s12b.env.actor_obs_layout_id
    assert s12c.env.loc_ref_residual_base == s12b.env.loc_ref_residual_base
    assert s12c.env.loc_ref_reset_base == s12b.env.loc_ref_reset_base
    assert s12c.env.loc_ref_penalty_pose_anchor == s12b.env.loc_ref_penalty_pose_anchor

    assert s12c.env.min_velocity == s12b.env.min_velocity
    assert s12c.env.max_velocity == s12b.env.max_velocity
    assert s12c.env.cmd_zero_chance == s12b.env.cmd_zero_chance
    assert s12c.env.eval_velocity_cmd == s12b.env.eval_velocity_cmd
    assert s12c.env.loc_ref_offline_command_vx == s12b.env.loc_ref_offline_command_vx

    for field in (
        "alive", "cmd_forward_velocity_track", "cmd_forward_velocity_alpha",
        "ref_body_quat_track", "ang_vel_xy", "action_rate", "penalty_pose",
        "penalty_close_feet_xy", "feet_phase", "feet_phase_alpha",
        "feet_phase_swing_height", "penalty_feet_ori",
        "ref_q_track", "ref_contact_match", "ref_feet_z_track",
    ):
        assert getattr(s12c.reward_weights, field) == getattr(s12b.reward_weights, field)


def test_smoke12c_only_tightens_four_leg_pitch_scales() -> None:
    s12b = _load_cfg(_SMOKE12B_CFG)
    s12c = _load_cfg(_SMOKE12C_CFG)
    b = dict(s12b.env.loc_ref_residual_scale_per_joint)
    c = dict(s12c.env.loc_ref_residual_scale_per_joint)

    for joint in _TIGHTENED_JOINTS:
        assert float(c[joint]) < float(b[joint]), (
            f"smoke12c residual_scale[{joint}] must be tighter than smoke12b; "
            f"smoke12b={b[joint]}, smoke12c={c[joint]}."
        )
        assert float(c[joint]) > 0.25, (
            f"smoke12c residual_scale[{joint}] fell back to the legacy 0.25 floor; "
            "that would erase the gait-emergence authority retained from smoke12b."
        )

    for joint in _UNCHANGED_JOINTS:
        assert float(c[joint]) == float(b[joint]), (
            f"smoke12c residual_scale[{joint}] drifted from smoke12b; only the "
            "hip/knee pitch joints should change."
        )


def test_smoke12c_exact_tightened_values_are_pinned() -> None:
    s12c = _load_cfg(_SMOKE12C_CFG)
    c = dict(s12c.env.loc_ref_residual_scale_per_joint)
    assert c["left_hip_pitch"] == pytest.approx(0.2827)
    assert c["left_knee_pitch"] == pytest.approx(0.5336)
    assert c["right_hip_pitch"] == pytest.approx(0.3094)
    assert c["right_knee_pitch"] == pytest.approx(0.5326)


def test_smoke12c_env_builds_and_zero_action_targets_home() -> None:
    env = _maybe_build_env(_SMOKE12C_CFG)
    assert env._residual_base_mode == "home"
    assert env._reset_base_mode == "home"

    home = np.asarray(env._home_q_rad).astype(np.float32)
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    for step_idx in (0, 5, 24, 47):
        win = env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        nominal_q_ref = win["q_ref"].astype(jp.float32)
        target_q, _ = env._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=nominal_q_ref,
        )
        np.testing.assert_allclose(
            np.asarray(target_q), home, atol=1e-5,
            err_msg=(
                f"smoke12c zero residual must still target home exactly at step_idx={step_idx}."
            ),
        )
