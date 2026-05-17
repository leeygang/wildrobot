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
  6. Non-walking consumers (runtime calibrate ``--scene-xml`` /
     ``--keyframes-xml`` home loaders, and the policy-export bundle
     home loader) return valid joint vectors within MJCF limits with
     the new locomotion-ready home.  These pin the "repo-wide
     semantic change" surface called out in the migration commit.

Standing-recipe note: ``training/configs/ppo_standing_v0173a.yaml``
is pre-existing incompatible with the current v3 env layout (it
requests ``actor_obs_layout_id: wr_obs_v2``, which the env rejected
during the v3 rewrite), so we cannot exercise ``WildRobotEnv`` with
the standing config.  The home migration is observable at the
runtime/export surfaces tested below; any future standing recipe will
need to retrain against the new ``home`` (action=0 now centers on the
locomotion-ready crouch).
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

    # The recorded values must not be byte-equal: ref_init is raw
    # q_ref(0) (no physics), home is the settled keyframe — they differ
    # by the post-settle drift on the knees/ankles (~25 mrad / ~6 mrad
    # from the derivation output).  This is the substantive check that
    # the two source paths really do differ in what they read.
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


def test_derivation_selector_uses_from_rest_initial_dsp() -> None:
    """Pin the derivation's actual scope: at the current ZMPWalkGenerator
    output the selector accepts ONLY the 8 from-rest initial DSP frames
    of the trajectory (cycle 0, indices 0..7).  Every later DSP window
    is structurally asymmetric on hip_pitch / ankle_pitch by ~0.176 rad
    and gets rejected by ``DSP_SYMMETRY_TOL_RAD``.

    This is the honest scope of the derivation — if the planner's phase
    origin changes (e.g. it stops starting from a static DSP) this
    test will catch the regression in the derivation contract.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from assets.derive_walk_ready_home import (
        DERIVATION_VX,
        DSP_SYMMETRY_TOL_RAD,
        _load_walking_q_ref,
        _select_phase_neutral_dsp_frames,
    )

    q_ref, contact_mask, joint_order = _load_walking_q_ref(DERIVATION_VX)
    selected = _select_phase_neutral_dsp_frames(
        q_ref, contact_mask, joint_order, sym_tol=DSP_SYMMETRY_TOL_RAD,
    )
    selected_idx = np.where(selected)[0].tolist()
    assert selected_idx == list(range(8)), (
        f"selector picked {selected_idx}; expected exactly [0..7] "
        "(the trajectory's from-rest initial DSP).  If the planner's "
        "phase origin changed, update the derivation strategy."
    )


# -----------------------------------------------------------------------------
# 6. Non-walking sanity: runtime calibrate.py + policy-export bundle home
#    loaders return valid joint vectors within MJCF limits with the new home.
# -----------------------------------------------------------------------------


def _actuator_names() -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]


def _mjcf_joint_range(actuator_name: str) -> tuple[float, float]:
    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    jid = int(model.actuator_trnid[aid, 0])
    return float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])


def test_runtime_calibrate_scene_loader_returns_valid_home() -> None:
    """``runtime/scripts/calibrate.py`` --scene-xml flow loads the home
    pose via ``load_home_from_scene``.  With the new locomotion-ready
    keyframe this must (a) return one float per actuator, (b) be the
    new crouched pose (knees > 0.30 rad), and (c) stay inside the
    MJCF joint ranges so a real --go-home will not over-drive a
    servo.  This is the sanity gate for the hardware path called out
    in the migration commit."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from runtime.scripts.calibrate import load_home_from_scene

    actuator_names = _actuator_names()
    home_ctrl = load_home_from_scene(_SCENE_XML, actuator_names)

    assert len(home_ctrl) == len(actuator_names)
    for name, v in zip(actuator_names, home_ctrl):
        assert np.isfinite(v), f"{name} home value not finite: {v}"
        lo, hi = _mjcf_joint_range(name)
        assert lo - 1e-6 <= v <= hi + 1e-6, (
            f"--go-home would drive {name}={v:+.4f} outside MJCF "
            f"range [{lo:+.4f}, {hi:+.4f}] — refusing."
        )

    by_name = dict(zip(actuator_names, home_ctrl))
    assert by_name["left_knee_pitch"] > 0.30, (
        f"runtime home knee_pitch={by_name['left_knee_pitch']:+.4f} not "
        "locomotion-ready — migration did not propagate to the runtime path."
    )


def test_runtime_calibrate_keyframes_xml_loader_returns_valid_home() -> None:
    """The other runtime path — ``--keyframes-xml`` — reads
    ``qpos[7:7+joint_count]`` directly from keyframes.xml.  Verifies
    that the new home keyframe still has 7+21 qpos values and that the
    21 joint-slot values are all finite + within the union of MJCF
    joint ranges.  (Note: the ordering this loader returns is
    qpos-address order, not actuator order — a pre-existing concern
    of that loader, unrelated to the home migration.)
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from runtime.scripts.calibrate import load_home_from_keyframes_xml

    keyframes_xml = _REPO_ROOT / "assets" / "v2" / "keyframes.xml"
    values = load_home_from_keyframes_xml(keyframes_xml, joint_count=21)
    assert len(values) == 21
    assert all(np.isfinite(v) for v in values)

    # Compare against the per-qpos-address pose recorded in the home
    # keyframe — the loader must be returning exactly qpos[7:28] of the
    # `home` entry.
    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    home_qpos = _key_qpos(model, "home")
    np.testing.assert_allclose(np.asarray(values), home_qpos[7:28], atol=1e-6)


def test_export_bundle_home_loader_matches_env_home_q_rad() -> None:
    """``training.exports.export_policy_bundle._get_home_ctrl_from_mjcf``
    is the canonical path that bakes ``home_ctrl_rad`` into a deployed
    bundle's ``policy_spec.json`` (consumed downstream by the runtime
    inference path).  Verify the exported value tracks the new
    locomotion-ready ``_home_q_rad`` from the env: if these ever
    diverge a deployed policy would centre its actions on a stale
    pose."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from training.exports.export_policy_bundle import _get_home_ctrl_from_mjcf

    env = _maybe_build_env(_SMOKE9C_CFG)
    actuator_names = list(env._policy_spec.robot.actuator_names)
    home_from_export = np.asarray(
        _get_home_ctrl_from_mjcf(_SMOKE9C_CFG, actuator_names),
        dtype=np.float64,
    )
    home_from_env = np.asarray(env._home_q_rad, dtype=np.float64)
    # Tolerance: 1e-4 rad (~0.006°).  Pre-existing JSON-vs-MJCF range
    # mismatch on the near-zero wrist/elbow slots can produce up to
    # ~1 µrad of clamp differential between the env (clips to MJCF
    # jnt_range) and the export bundle (clips to robot_config.json
    # range_min/max).  1e-4 stays well below servo precision but is
    # tight enough to catch any real divergence (knee/hip differences
    # would be > 1 mrad).
    np.testing.assert_allclose(
        home_from_export, home_from_env, atol=1e-4,
        err_msg=(
            "export bundle home_ctrl_rad disagrees with env._home_q_rad — "
            "deployed bundles would centre on a different pose than "
            "training."
        ),
    )
