"""Contract tests for smoke13 (home-anchored residual walking).

Smoke13 is the simplification step on top of smoke12b:

  - one canonical action anchor: ``home`` (``loc_ref_residual_base:
    home``).  No ``action_mode`` selector and no separate
    direct-vs-residual code path.
  - TB-parity privileged critic (``critic_imitation_refs: true``).
    Actor observation stays clean (``wr_obs_v7_phase_proprio`` — phase
    + proprio + proprio_history; no offline-reference window).
  - TB-scalar action scale (0.25) with per-joint widening only on the
    leg-pitch chain (knee_pitch 0.5, hip_pitch 0.35) — narrower than
    smoke12b's widened {knee 0.667, hip 0.353/0.387} so the G5
    anti-exploit gate stays interior on knees + hip-pitch.
  - TB-active reward block (same weights as smoke12b); every hybrid
    / imitation term explicitly zeroed.
  - V6EvalAdapter mirrors training reset (joint noise + DR offsets
    + torso pose perturbation) and uses the same home-anchored base
    for visual/native-MuJoCo eval.

Spec: /tmp/smoke13_home_base_prompt.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.configs.training_config import load_training_config


_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")


@pytest.fixture(scope="module")
def smoke13_cfg():
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    return load_training_config(str(_SMOKE13))


# ---------------------------------------------------------------------------
# 1. Config loads with the simpler home-anchored contract.
# ---------------------------------------------------------------------------
def test_smoke13_loads(smoke13_cfg) -> None:
    assert smoke13_cfg is not None


def test_smoke13_uses_home_as_only_action_anchor(smoke13_cfg) -> None:
    """One canonical anchor — ``loc_ref_residual_base: home``.  No
    ``action_mode`` selector should reappear in the config."""
    env_cfg = smoke13_cfg.env
    assert env_cfg.loc_ref_residual_base == "home"
    assert env_cfg.loc_ref_reset_base == "home"
    assert env_cfg.loc_ref_penalty_pose_anchor == "home"
    assert not hasattr(env_cfg, "action_mode"), (
        "smoke13 must not reintroduce env.action_mode; "
        "loc_ref_residual_base is the sole anchor selector"
    )


# ---------------------------------------------------------------------------
# 2. Zero action targets exactly _home_q_rad (no residual offset, no
#    q_ref drift).
# ---------------------------------------------------------------------------
def _make_env(cfg):
    # Importing the env is heavy (mjx + mujoco + JAX); keep it inside
    # the test functions so import-only test discovery stays cheap.
    from training.envs.wildrobot_env import WildRobotEnv
    return WildRobotEnv(config=cfg)


def test_smoke13_zero_action_targets_home(smoke13_cfg) -> None:
    """Zero policy_action under home base must compose to exactly
    ``_home_q_rad`` (clipped to joint range, which is a no-op because
    home is inside the range).  Pins the zero-residual invariant
    ``target_q = home + scale * 0 = home``."""
    import jax.numpy as jp

    env = _make_env(smoke13_cfg)
    zero_action = jp.zeros((env.action_size,), dtype=jp.float32)
    win0 = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))
    target_q, residual_delta = env._compose_target_q_from_residual(
        policy_action=zero_action, nominal_q_ref=win0["q_ref"]
    )
    assert np.allclose(np.asarray(residual_delta), 0.0, atol=1e-7)
    assert np.allclose(
        np.asarray(target_q), np.asarray(env._home_q_rad), atol=1e-6
    ), "smoke13 home base: zero action must target exactly home_q_rad"


def test_smoke13_action_path_is_independent_of_qref(smoke13_cfg) -> None:
    """Under ``loc_ref_residual_base == "home"`` the env's
    ``_compose_target_q_from_residual`` must not consult
    ``nominal_q_ref(t)`` — pinning that the action path is genuinely
    anchored on home, not on the offline ZMP trajectory.

    Feed adversarial (massively different) q_ref values and confirm
    target_q is byte-equal."""
    import jax.numpy as jp

    env = _make_env(smoke13_cfg)
    action = jp.full((env.action_size,), 0.3, dtype=jp.float32)
    win0 = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))
    target_real, _ = env._compose_target_q_from_residual(
        policy_action=action, nominal_q_ref=win0["q_ref"]
    )
    fake_q_ref = jp.ones_like(win0["q_ref"]) * 999.0
    target_fake, _ = env._compose_target_q_from_residual(
        policy_action=action, nominal_q_ref=fake_q_ref
    )
    np.testing.assert_allclose(
        np.asarray(target_real), np.asarray(target_fake), atol=1e-6,
        err_msg=(
            "smoke13 action path leaked dependence on nominal_q_ref — "
            "loc_ref_residual_base=home should hard-anchor on home_q_rad"
        ),
    )


# ---------------------------------------------------------------------------
# 3. Actor obs is the clean v7 layout (no actor-side offline-ref leak).
# ---------------------------------------------------------------------------
def test_smoke13_actor_obs_layout_is_phase_proprio_only(smoke13_cfg) -> None:
    """``wr_obs_v7_phase_proprio`` — v3 base + phase(2) +
    proprio_history.  No offline-reference window channels (loc_ref
    q_ref, pelvis, per-foot pos/vel, contact_mask) in the actor
    stream.  The layout dispatch in policy_contract/numpy/obs.py
    enforces this; pinning at the config layer too keeps the contract
    anchored at the config layer."""
    assert (
        smoke13_cfg.env.actor_obs_layout_id == "wr_obs_v7_phase_proprio"
    )


# ---------------------------------------------------------------------------
# 4. Privileged critic stays TB-parity (legacy imitation refs allowed
#    on the critic; actor stays clean).
# ---------------------------------------------------------------------------
def test_smoke13_critic_keeps_tb_parity_imitation_refs(smoke13_cfg) -> None:
    """``critic_imitation_refs: true`` keeps motor_pos_error vs
    nominal_q_ref(t) and ref_stance from win["contact_mask"] on the
    critic.  Mirrors TB walk.gin:25-28 critic payload.  Smoke13 does
    not strip the critic — actor cleanliness is the contract, not
    critic cleanliness."""
    assert smoke13_cfg.env.critic_imitation_refs is True


# ---------------------------------------------------------------------------
# 5. V6EvalAdapter mirrors home-base semantics so visual/native eval
#    composes the same target_q as the training env.
# ---------------------------------------------------------------------------
def test_smoke13_v6_adapter_uses_home_base(smoke13_cfg) -> None:
    """V6EvalAdapter reads loc_ref_residual_base, picks home, and
    ctrl_init = home_q (matches the env's _make_initial_state under
    the same config)."""
    import mujoco
    from training.eval.v6_eval_adapter import V6EvalAdapter
    from training.policy_spec_utils import build_policy_spec_from_training_config
    from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter
    from training.configs.training_config import load_robot_config

    mj_model = mujoco.MjModel.from_xml_path(str(Path(smoke13_cfg.env.model_path)))
    rcfg = load_robot_config(smoke13_cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=smoke13_cfg, robot_cfg=rcfg, action_filter_alpha=0.0
    )
    sigA = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=rcfg,
        policy_spec=spec,
        foot_switch_threshold=smoke13_cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=smoke13_cfg,
        mj_model=mj_model,
        policy_spec=spec,
        signals_adapter=sigA,
        action_dim=int(spec.model.action_dim),
    )
    assert adapter._residual_base_mode == "home"
    np.testing.assert_allclose(
        adapter._ctrl_init_policy_order(), adapter._home_q_rad, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 6. Reward block matches TB-active; hybrid / imitation terms zero.
# ---------------------------------------------------------------------------
_TB_ACTIVE_NONZERO = {
    # walk.gin row -> expected smoke13 weight
    "alive": 1.0,                       # walk.gin:127
    "cmd_forward_velocity_track": 2.0,  # walk.gin:111 (lin_vel_xy)
    "ref_body_quat_track": 2.5,         # walk.gin:117 (torso_quat)
    "ang_vel_xy": 1.0,                  # walk.gin:115 (penalty_ang_vel_xy)
    "action_rate": -2.0,                # walk.gin:119 (penalty_action_rate)
    "penalty_pose": -0.5,               # walk.gin:120
    "penalty_close_feet_xy": 10.0,      # walk.gin:126
    "feet_phase": 7.5,                  # walk.gin:128
    "penalty_feet_ori": 5.0,            # walk.gin:129
}

_HYBRID_MUST_BE_ZERO = (
    "ref_q_track",          # joint-imitation; TB-active drops
    "ref_contact_match",    # contact-imitation
    "ref_feet_z_track",     # foot-z imitation
    "torso_pos_xy",         # torso-XY imitation
    "lin_vel_z",            # vertical-bobbing imitation
    "feet_air_time",        # sparse; replaced by feet_phase
    "feet_clearance",       # sparse; replaced by feet_phase
    "feet_distance",        # superseded by penalty_close_feet_xy
    "torso_pitch_soft",     # superseded by penalty_ang_vel_xy
    "torso_roll_soft",      # superseded by penalty_ang_vel_xy
    "torque",               # not in TB-active
    "joint_velocity",       # not in TB-active
    "slip",                 # not in TB-active
    "pitch_rate",           # M1 diagnostic hook only
)


def test_smoke13_reward_block_matches_tb_active(smoke13_cfg) -> None:
    w = smoke13_cfg.reward_weights
    assert w.cmd_velocity_track_dim == 2
    for name, expected in _TB_ACTIVE_NONZERO.items():
        got = getattr(w, name)
        assert got == pytest.approx(expected), (
            f"smoke13 reward_weights.{name} = {got}, expected {expected} "
            "(TB-active walk.gin:110-131)"
        )


def test_smoke13_hybrid_rewards_explicitly_zero(smoke13_cfg) -> None:
    w = smoke13_cfg.reward_weights
    for name in _HYBRID_MUST_BE_ZERO:
        got = getattr(w, name)
        assert got == pytest.approx(0.0), (
            f"smoke13 reward_weights.{name} = {got}, must be 0.0 "
            "(hybrid / imitation term; explicitly disabled in TB-active block)"
        )


# ---------------------------------------------------------------------------
# 7. Action scale spec: scalar 0.25 + per-joint widening on leg-pitch only.
# ---------------------------------------------------------------------------
def test_smoke13_action_scale_spec(smoke13_cfg) -> None:
    """Smoke13 keeps smoke12b's residual authority exactly.  Earlier
    smoke13 drafts narrowed knees to 0.5 / hip-pitch to 0.35 and
    produced a failed run (height-low collapse + back-drive in
    offline-run-20260521_221534-29exm4g5); the restored values are
    those generated by ``assets/derive_residual_scales.py --coverage
    0.75`` for the WR walking prior at vx=0.1333.  Pinning the exact
    map so a future config rewrite can't drift back to the narrowed
    cap or to pure TB scalar 0.25 (which is too tight for WR's longer
    legs)."""
    env = smoke13_cfg.env
    assert env.loc_ref_residual_mode == "absolute"
    assert env.loc_ref_residual_scale == pytest.approx(0.2)
    expected = {
        "left_hip_pitch": 0.3534,
        "left_hip_roll": 0.25,
        "left_knee_pitch": 0.667,
        "left_ankle_pitch": 0.25,
        "left_ankle_roll": 0.25,
        "right_hip_pitch": 0.3867,
        "right_hip_roll": 0.25,
        "right_knee_pitch": 0.6657,
        "right_ankle_pitch": 0.25,
        "right_ankle_roll": 0.25,
    }
    got = dict(env.loc_ref_residual_scale_per_joint)
    assert got == pytest.approx(expected), (
        f"smoke13 per-joint scale must match smoke12b exactly; "
        f"expected {expected}; got {got}"
    )


def test_smoke13_enables_command_conditioned_reference_lookup(
    smoke13_cfg,
) -> None:
    """Smoke13 turns on TB-style command-conditioned reference lookup
    (``loc_ref_command_conditioned: true``) so the offline ref
    selected each episode tracks the sampled velocity_cmd instead of
    a fixed offline_vx."""
    assert smoke13_cfg.env.loc_ref_command_conditioned is True


def test_smoke13_uses_2d_velocity_tracking(smoke13_cfg) -> None:
    """Smoke13 enables 2D velocity tracking (``cmd_velocity_track_dim:
    2``) so the velocity reward compares actual (vx, vy) to the
    cmd-conditioned reference's pelvis velocity (TB parity)."""
    assert smoke13_cfg.reward_weights.cmd_velocity_track_dim == 2


# ---------------------------------------------------------------------------
# 8. Reset adapter parity: V6EvalAdapter applies torso pose perturbation
#    AND DR joint offsets when configured.  (Three layered perturbations:
#    joint noise (a), DR offsets (b), torso pose perturbation (c) —
#    smoke13 yaml header documents all three.)
# ---------------------------------------------------------------------------
def _build_adapter(smoke13_cfg):
    import mujoco
    from training.eval.v6_eval_adapter import V6EvalAdapter
    from training.policy_spec_utils import build_policy_spec_from_training_config
    from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter
    from training.configs.training_config import load_robot_config

    mj_model = mujoco.MjModel.from_xml_path(str(Path(smoke13_cfg.env.model_path)))
    rcfg = load_robot_config(smoke13_cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=smoke13_cfg, robot_cfg=rcfg, action_filter_alpha=0.0
    )
    sigA = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=rcfg,
        policy_spec=spec,
        foot_switch_threshold=smoke13_cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=smoke13_cfg,
        mj_model=mj_model,
        policy_spec=spec,
        signals_adapter=sigA,
        action_dim=int(spec.model.action_dim),
    )
    mj_data = mujoco.MjData(mj_model)
    return adapter, mj_data


def test_smoke13_adapter_reset_applies_torso_perturbation(smoke13_cfg) -> None:
    """smoke13 sets reset_torso_{roll,pitch}_range to [-0.1, 0.1]; the
    adapter must report perturbation enabled and reset_native_mj_state
    must actually move root quat / leg-pitch qpos across resets."""
    adapter, mj_data = _build_adapter(smoke13_cfg)
    assert adapter._reset_perturbation_enabled(), (
        "smoke13 sets reset_torso_{roll,pitch}_range to [-0.1, 0.1]; "
        "the adapter must report the perturbation enabled."
    )
    adapter.reset_native_mj_state(mj_data, apply_noise=False, rng=None)
    qpos_baseline = mj_data.qpos.copy()
    root_quat_baseline = qpos_baseline[3:7].copy()
    rng = np.random.default_rng(42)
    leg_addrs = adapter._leg_pitch_qpos_addrs
    moved = False
    for _ in range(5):
        adapter.reset_native_mj_state(mj_data, apply_noise=True, rng=rng)
        if not np.allclose(mj_data.qpos[3:7], root_quat_baseline, atol=1e-5):
            moved = True
            break
        if leg_addrs.size > 0 and not np.allclose(
            mj_data.qpos[leg_addrs], qpos_baseline[leg_addrs], atol=1e-5
        ):
            moved = True
            break
    assert moved, (
        "Adapter reset did not modify root quat or leg-pitch qpos "
        "across 5 resets — torso pose perturbation is not being "
        "applied even though range is non-degenerate."
    )


def test_smoke13_adapter_reset_applies_dr_joint_offsets(smoke13_cfg) -> None:
    """smoke13 keeps DR enabled; the adapter must (a) report DR
    enabled and (b) widen the per-joint qpos distribution materially
    when ``apply_dr=True`` vs ``apply_dr=False`` (joint noise alone)."""
    adapter, mj_data = _build_adapter(smoke13_cfg)
    assert adapter._dr_enabled is True
    assert adapter._dr_joint_offset_rad > 0.0

    n_resets = 64
    rng_noise_only = np.random.default_rng(123)
    noise_only_q = np.empty((n_resets, adapter._action_dim), dtype=np.float32)
    for i in range(n_resets):
        adapter.reset_native_mj_state(
            mj_data,
            apply_noise=True,
            rng=rng_noise_only,
            perturb_pose=False,
            apply_dr=False,
        )
        noise_only_q[i] = mj_data.qpos[adapter._actuator_qpos_addrs].astype(
            np.float32
        )

    rng_with_dr = np.random.default_rng(123)
    with_dr_q = np.empty((n_resets, adapter._action_dim), dtype=np.float32)
    for i in range(n_resets):
        adapter.reset_native_mj_state(
            mj_data,
            apply_noise=True,
            rng=rng_with_dr,
            perturb_pose=False,
            apply_dr=True,
        )
        with_dr_q[i] = mj_data.qpos[adapter._actuator_qpos_addrs].astype(
            np.float32
        )

    # Joint noise alone std ≈ 0.05/sqrt(3) ≈ 0.0289; +DR offset
    # 0.03 → combined std sqrt(0.05² + 0.03²)/sqrt(3) ≈ 0.0337.
    # Theoretical ratio ≈ 1.166; threshold 1.08 leaves headroom for
    # sample noise over 64 resets.
    noise_only_std = float(np.median(np.std(noise_only_q, axis=0)))
    with_dr_std = float(np.median(np.std(with_dr_q, axis=0)))
    assert with_dr_std > noise_only_std * 1.08, (
        f"smoke13 adapter reset DR joint offsets not visible: with_dr "
        f"std={with_dr_std:.4f} vs noise_only std={noise_only_std:.4f}."
    )
    # Independent check: max excursion with DR exceeds the
    # joint-noise-only theoretical cap of 0.05 rad.
    home_q = adapter._home_q_rad
    noise_only_max = float(np.max(np.abs(noise_only_q - home_q[None, :])))
    with_dr_max = float(np.max(np.abs(with_dr_q - home_q[None, :])))
    assert with_dr_max > noise_only_max, (
        f"smoke13 adapter reset DR offsets did not widen max excursion: "
        f"with_dr max={with_dr_max:.4f} vs noise_only max={noise_only_max:.4f}."
    )


# ---------------------------------------------------------------------------
# 9. Command-conditioned reference lookup: at runtime the env picks the
#    bin nearest the sampled velocity_cmd, AND legacy configs still
#    default to single-bin behavior.
# ---------------------------------------------------------------------------
def test_smoke13_cmd_conditioned_lookup_selects_nearest_bin(smoke13_cfg) -> None:
    """Under loc_ref_command_conditioned=true the env's
    _lookup_offline_window must pick a different bin when called with
    velocity_cmd at the low end vs the high end of the configured
    range.  Frame 0 q_ref happens to be identical across bins (all
    start at the same crouch pose), so we probe mid-trajectory and
    compare pelvis_vel — that differs by construction because the
    finite-diff'd reference is integrated at the configured vx."""
    import jax.numpy as jp

    env = _make_env(smoke13_cfg)
    assert env._offline_command_conditioned, (
        "smoke13 must enable cmd-conditioned lookup; runtime flag is off"
    )
    # Probe mid-trajectory so the finite-diff'd pelvis_vel has spun up.
    mid = jp.asarray(100, dtype=jp.int32)
    w_lo = env._lookup_offline_window(mid, velocity_cmd=jp.float32(0.08))
    w_hi = env._lookup_offline_window(mid, velocity_cmd=jp.float32(0.1333))
    ref_vx_lo = float(np.asarray(w_lo["pelvis_vel"])[0])
    ref_vx_hi = float(np.asarray(w_hi["pelvis_vel"])[0])
    # ref_vx for vx=0.08 should be near 0.08; for vx=0.1333 near 0.1333.
    # Use loose 25% bands — the finite-diff is noisy mid-cycle.
    assert 0.05 < ref_vx_lo < 0.11, (
        f"ref_vx at velocity_cmd=0.08 should be near 0.08; got {ref_vx_lo}"
    )
    assert 0.10 < ref_vx_hi < 0.17, (
        f"ref_vx at velocity_cmd=0.1333 should be near 0.1333; got {ref_vx_hi}"
    )
    assert ref_vx_lo < ref_vx_hi, (
        "cmd-conditioned lookup must produce a smaller ref_vx for the "
        "smaller cmd; the bin selection is not nearest-neighbor."
    )


def test_smoke13_command_grid_uses_reward_sensitivity_interval(smoke13_cfg) -> None:
    """Smoke13 builds the cmd-conditioned vx grid with a reward-sensitivity
    interval, then unions loc_ref_offline_command_vx so the eval cmd
    snaps to its own bin exactly.

    TB's default lookup interval is 0.05 m/s, but smoke13's
    cmd_velocity_track_alpha=562.5 makes a 0.02 m/s interval the
    smallest simple grid that keeps worst-case nearest-bin mismatch
    around <=0.01 m/s over the narrow [0.08, 0.1333] range."""
    assert smoke13_cfg.env.loc_ref_command_grid_interval == pytest.approx(0.02)
    env = _make_env(smoke13_cfg)
    grid = np.asarray(env._offline_vx_grid)
    expected = np.array([0.08, 0.10, 0.12, 0.1333333333], dtype=np.float32)
    np.testing.assert_allclose(grid, expected, atol=1e-5)
    # eval cmd must hit its own bin exactly so post-training
    # deterministic eval doesn't snap to the nearest arange step.
    eval_cmd = float(smoke13_cfg.env.eval_velocity_cmd)
    nearest = float(grid[int(np.argmin(np.abs(grid - eval_cmd)))])
    assert nearest == pytest.approx(eval_cmd, abs=1e-5), (
        f"eval_velocity_cmd={eval_cmd} should hit its own bin "
        f"exactly; got nearest={nearest}"
    )


def test_legacy_configs_default_to_single_bin_reference_lookup() -> None:
    """Older configs (smoke9, smoke12b) must NOT silently switch to
    command-conditioned lookup — the new flag defaults to false and
    they should keep using the historical single-bin path."""
    smoke12b_path = Path("training/configs/ppo_walking_v0201_smoke12b.yaml")
    if not smoke12b_path.exists():
        pytest.skip(f"{smoke12b_path.name} not found")
    cfg = load_training_config(str(smoke12b_path))
    assert cfg.env.loc_ref_command_conditioned is False, (
        "loc_ref_command_conditioned default must remain false so "
        "legacy configs keep their single-bin reference behavior."
    )
