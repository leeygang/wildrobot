"""Pin the locomotion-ready ``home`` keyframe migration (2026-05-17).

After the migration, ``home`` no longer means the near-straight standing
pose; it is the phase-neutral, mirror-symmetric pose derived from the WR
walking prior by ``assets/derive_walk_ready_home.py``.  The tests in this
module are the contract for that meaning:

  1. The MJCF ``home`` keyframe's leg slots match the pose that the
     derivation script produces today.
  2. The env's ``_home_q_rad`` reflects the new locomotion-ready pose
     (knees > 0.30 rad) — i.e. ``home`` is wired through to runtime.
  3. With ``loc_ref_residual_base="home"`` and zero policy action, the
     composed target_q is exactly ``_home_q_rad`` (TB ``default_action``
     contract).
  4. ``ref_init_q_rad`` is still the offline frame-0 q_ref and remains
     conceptually distinct from ``home`` (we keep the semantic split
     even though both poses are now near each other numerically).
  5. The envelope-coverage report runs without error (informational
     gate, not numeric assertion).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENE_XML = _REPO_ROOT / "assets" / "v2" / "scene_flat_terrain.xml"
_SMOKE9C_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke9c.yaml"
_SMOKE8_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke8.yaml"


LEG_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
    "right_ankle_pitch", "right_ankle_roll",
)


def _key_qpos(model: mujoco.MjModel, key_name: str) -> np.ndarray:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    assert kid >= 0, f"keyframe '{key_name}' missing"
    return np.array(model.key_qpos[kid], dtype=np.float64)


def _actuator_qpos(
    model: mujoco.MjModel, key_qpos: np.ndarray, actuator_name: str
) -> float:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    assert aid >= 0, actuator_name
    jid = int(model.actuator_trnid[aid, 0])
    qaddr = int(model.jnt_qposadr[jid])
    return float(key_qpos[qaddr])


# -----------------------------------------------------------------------------
# 1. MJCF home keyframe matches what the derivation script produces today
# -----------------------------------------------------------------------------


def test_home_keyframe_matches_derivation_script() -> None:
    """Re-running the derivation script and re-settling must reproduce the
    leg-joint slots stored in the ``home`` keyframe within settling noise.

    Pins the migration: if anyone re-derives the prior (e.g. changes
    operating-point vx, IK geometry, or the symmetry tolerance) the new
    home keyframe must be regenerated.  The test re-runs only the pose
    derivation (cheap), not the 2000-step settling (slow), and compares
    against the on-disk MJCF home slots with a settle-aware tolerance.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from assets.derive_walk_ready_home import (
        DERIVATION_VX,
        DSP_SYMMETRY_TOL_RAD,
        _derive_joint_pose,
        _load_walking_q_ref,
        _select_phase_neutral_dsp_frames,
    )

    q_ref, contact_mask, joint_order = _load_walking_q_ref(DERIVATION_VX)
    selected = _select_phase_neutral_dsp_frames(
        q_ref, contact_mask, joint_order, sym_tol=DSP_SYMMETRY_TOL_RAD,
    )
    derived = _derive_joint_pose(q_ref, selected, joint_order)
    name_to_col = {name: i for i, name in enumerate(joint_order)}

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    home_qpos = _key_qpos(model, "home")

    # Settling shifts knees by ~25 mrad and ankles by ~6 mrad on the way
    # to equilibrium (recorded in derive_walk_ready_home.py output); the
    # static (pre-settle) pose is what we compare against, with a 5e-2
    # rad envelope to absorb settling drift without admitting an
    # unrelated re-derivation.
    for joint in LEG_JOINTS:
        keyframe_v = _actuator_qpos(model, home_qpos, joint)
        derived_v = float(derived[name_to_col[joint]])
        assert abs(keyframe_v - derived_v) < 5e-2, (
            f"keyframe home {joint}={keyframe_v:+.4f} drifts from "
            f"derived {derived_v:+.4f} by more than 50 mrad — "
            "re-run assets/derive_walk_ready_home.py --write."
        )


# -----------------------------------------------------------------------------
# 2. home keyframe is locomotion-ready
# -----------------------------------------------------------------------------


def test_home_keyframe_is_locomotion_ready() -> None:
    """``home``'s knee_pitch slots must be in the locomotion-ready range
    (≥0.30 rad).  This is the headline of the migration: before, ``home``
    had knees ≈ 0.036 rad and was a clean standing pose; after, ``home``
    is the phase-neutral walking-stance pose."""
    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    home_qpos = _key_qpos(model, "home")
    for joint in ("left_knee_pitch", "right_knee_pitch"):
        v = _actuator_qpos(model, home_qpos, joint)
        assert v > 0.30, (
            f"home {joint}={v:+.4f} no longer locomotion-ready "
            "(expected > 0.30 rad after the 2026-05-17 migration)."
        )

    # And the pose is mirror-symmetric on the paired chain (up to
    # post-settling drift).
    for left, right in (
        ("left_hip_pitch", "right_hip_pitch"),     # opposite-sign mirror
        ("left_hip_roll", "right_hip_roll"),
        ("left_ankle_roll", "right_ankle_roll"),
    ):
        lv = _actuator_qpos(model, home_qpos, left)
        rv = _actuator_qpos(model, home_qpos, right)
        assert abs(lv + rv) < 5e-3, (
            f"opposite-sign mirror broken: {left}={lv:+.4f}, "
            f"{right}={rv:+.4f}"
        )
    for left, right in (
        ("left_knee_pitch", "right_knee_pitch"),
        ("left_ankle_pitch", "right_ankle_pitch"),
    ):
        lv = _actuator_qpos(model, home_qpos, left)
        rv = _actuator_qpos(model, home_qpos, right)
        assert abs(lv - rv) < 5e-3, (
            f"same-sign mirror broken: {left}={lv:+.4f}, "
            f"{right}={rv:+.4f}"
        )


# -----------------------------------------------------------------------------
# 3 + 4 + 5. env wiring: _home_q_rad, residual_base="home" zero-action,
#            ref_init_q distinct from home, envelope report runs.
# -----------------------------------------------------------------------------


def _maybe_build_env(cfg_path: Path):
    """Helper: load a config + build the env, or skip if deps missing."""
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path.name} not found")
    try:
        from assets.robot_config import load_robot_config
        from training.configs.training_config import load_training_config
        from training.envs.wildrobot_env import WildRobotEnv
    except Exception as exc:  # pragma: no cover - import gate
        pytest.skip(f"env deps unavailable: {exc}")
    load_robot_config(str(_REPO_ROOT / "assets/v2/mujoco_robot_config.json"))
    cfg = load_training_config(str(cfg_path))
    cfg.freeze()
    return WildRobotEnv(cfg)


def test_env_home_q_rad_reflects_new_keyframe() -> None:
    """Env-level ``_home_q_rad`` (built via
    ``ControlAbstractionLayer.get_ctrl_for_default_pose()``) must reflect
    the new locomotion-ready home keyframe — knee slots > 0.30 rad and
    mirror-symmetric.  Catches any path that bypasses the keyframe and
    falls back to ``qpos0`` (which is all zeros)."""
    env = _maybe_build_env(_SMOKE9C_CFG)
    home_q = np.asarray(env._home_q_rad).astype(np.float64)
    actuator_names = env._policy_spec.robot.actuator_names

    def _idx(name: str) -> int:
        return actuator_names.index(name)

    for joint in ("left_knee_pitch", "right_knee_pitch"):
        v = float(home_q[_idx(joint)])
        assert v > 0.30, f"env._home_q_rad {joint}={v:+.4f} <= 0.30"

    # Mirror-symmetry up to settling drift.
    lh, rh = float(home_q[_idx("left_hip_pitch")]), float(home_q[_idx("right_hip_pitch")])
    lk, rk = float(home_q[_idx("left_knee_pitch")]), float(home_q[_idx("right_knee_pitch")])
    la, ra = float(home_q[_idx("left_ankle_pitch")]), float(home_q[_idx("right_ankle_pitch")])
    assert abs(lh + rh) < 5e-3
    assert abs(lk - rk) < 5e-3
    assert abs(la - ra) < 5e-3


def test_zero_action_under_home_base_targets_home_q_rad() -> None:
    """Smoke8 config has ``loc_ref_residual_base: home``.  With a zero
    policy action the composed target_q must equal ``_home_q_rad`` at
    every step of the offline cycle (TB ``default_action`` contract).
    """
    import jax.numpy as jp
    env = _maybe_build_env(_SMOKE8_CFG)
    assert env._residual_base_mode == "home"
    home = np.asarray(env._home_q_rad).astype(np.float32)
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    for step_idx in (0, 5, 10, 20, 30, 47):
        win = env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        nominal_q_ref = win["q_ref"].astype(jp.float32)
        target_q, _ = env._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=nominal_q_ref,
        )
        np.testing.assert_allclose(
            np.asarray(target_q), home, atol=1e-5,
            err_msg=(
                f"step {step_idx}: target_q != _home_q_rad under "
                "loc_ref_residual_base=home with zero action."
            ),
        )


def test_ref_init_q_distinct_from_home_yet_close() -> None:
    """``ref_init_q_rad`` is the offline frame-0 q_ref — it must remain
    distinct from ``_home_q_rad`` as a CONCEPT (different sources), but
    since both poses are phase-neutral DSP equilibria they should be
    numerically close (within ~30 mrad on the leg joints, mainly from
    settling drift on the keyframe side).
    """
    env = _maybe_build_env(_SMOKE9C_CFG)
    ref_init = np.asarray(env._ref_init_q_rad).astype(np.float64)
    home = np.asarray(env._home_q_rad).astype(np.float64)

    # The two arrays must not point at the same buffer (distinct sources).
    assert ref_init is not home

    # And the recorded values must not be byte-equal: ref_init is raw
    # q_ref(0) (no physics), home is the settled keyframe — they differ
    # by the post-settle drift on the knees/ankles (~25 mrad / ~6 mrad
    # from the derivation output).
    assert not np.array_equal(ref_init, home), (
        "_ref_init_q_rad and _home_q_rad are byte-equal — the two "
        "concepts have collapsed.  ref_init is the offline frame-0 "
        "q_ref; home is the settled keyframe."
    )

    # But they must agree on phase-neutrality: per-joint difference is
    # bounded by the settling drift envelope.
    max_diff = float(np.max(np.abs(ref_init - home)))
    assert max_diff < 5e-2, (
        f"ref_init and home diverged by {max_diff:.4f} rad — they should "
        "still be co-located after the home migration (both are settled "
        "or near-settled phase-neutral poses).  Investigate the offline "
        "library bin or the home derivation."
    )


def test_envelope_report_runs_without_error() -> None:
    """The derivation script's envelope-coverage report must run end-to-end
    without raising.  Informational gate: we do NOT assert on the deficits
    because the residual scale is small by design and the report is the
    basis for deciding whether to grow residual authority later."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from assets.derive_walk_ready_home import (
        DERIVATION_VX,
        DSP_SYMMETRY_TOL_RAD,
        RESIDUAL_SCALE_PER_LEG_JOINT_RAD,
        _derive_joint_pose,
        _envelope_report,
        _load_walking_q_ref,
        _select_phase_neutral_dsp_frames,
    )

    q_ref, contact_mask, joint_order = _load_walking_q_ref(DERIVATION_VX)
    selected = _select_phase_neutral_dsp_frames(
        q_ref, contact_mask, joint_order, sym_tol=DSP_SYMMETRY_TOL_RAD,
    )
    derived = _derive_joint_pose(q_ref, selected, joint_order)
    lines = _envelope_report(
        q_ref, derived, joint_order, RESIDUAL_SCALE_PER_LEG_JOINT_RAD,
    )
    assert lines and all("home=" in line for line in lines)
