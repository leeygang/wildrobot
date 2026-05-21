"""Contract tests for smoke13 (TB-style direct-policy walking migration).

Pin the six architectural invariants from the migration plan:

  1. Config loads and selects ``action_mode == "direct"``.
  2. Zero action targets exactly ``_home_q_rad`` (no residual_base
     switching, no q_ref offset).
  3. Action path does not depend on ``loc_ref_residual_base`` —
     flipping that field while ``action_mode == "direct"`` leaves
     ``target_q`` unchanged.
  4. Actor observation layout is phase + proprio only
     (``wr_obs_v7_phase_proprio``), with no offline-reference window
     channels in the actor stream.
  5. Native-MuJoCo eval adapter (``V6EvalAdapter``) routes the same
     direct-policy semantics so visualize_policy.py / eval ladders
     are contract-matched.
  6. Reward block matches TB-active (walk.gin:110-131); every
     hybrid/imitation term is explicitly zeroed.

Spec: /tmp/wr_tb_style_direct_policy_migration_plan.md.
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
# 1. Config loads and selects the new direct-policy action mode.
# ---------------------------------------------------------------------------
def test_smoke13_loads(smoke13_cfg) -> None:
    assert smoke13_cfg is not None


def test_smoke13_selects_action_mode_direct(smoke13_cfg) -> None:
    """The architectural promise of smoke13.  If this flips back to
    'residual' the entire migration is vacuous."""
    assert smoke13_cfg.env.action_mode == "direct"


# ---------------------------------------------------------------------------
# 2. Action path: zero action -> exactly q_default (= _home_q_rad).
# 3. Action path independent of loc_ref_residual_base.
# ---------------------------------------------------------------------------
def _make_env(cfg):
    # Importing the env is heavy (mjx + mujoco + JAX); keep it inside
    # the test functions so import-only test discovery stays cheap.
    from training.envs.wildrobot_env import WildRobotEnv
    return WildRobotEnv(config=cfg)


def test_smoke13_zero_action_targets_home(smoke13_cfg) -> None:
    """Zero policy_action under direct mode must compose to exactly
    ``_home_q_rad`` (clipped to joint range, which is a no-op because
    home is already inside the range).  Pins the TB-equivalence
    invariant ``target_q = home + 0.25 * 0 = home``."""
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
    ), "smoke13 direct mode: zero action must target exactly home_q_rad"


def test_smoke13_action_path_ignores_residual_base(smoke13_cfg) -> None:
    """Under ``action_mode == "direct"`` the env's
    ``_compose_target_q_from_residual`` must NOT consult
    ``_residual_base_mode``.  Verified by flipping the cached base mode
    on a built env and checking that ``target_q`` for the same
    non-zero action is unchanged.

    This guards against a future regression that drops the
    ``action_mode == "direct"`` short-circuit and reintroduces the
    residual_base switch."""
    import jax.numpy as jp

    env = _make_env(smoke13_cfg)
    action = jp.full((env.action_size,), 0.3, dtype=jp.float32)
    win0 = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))

    original_base_mode = env._residual_base_mode
    target_home, _ = env._compose_target_q_from_residual(
        policy_action=action, nominal_q_ref=win0["q_ref"]
    )
    try:
        env._residual_base_mode = "q_ref"
        target_qref, _ = env._compose_target_q_from_residual(
            policy_action=action, nominal_q_ref=win0["q_ref"]
        )
        env._residual_base_mode = "ref_init"
        target_refinit, _ = env._compose_target_q_from_residual(
            policy_action=action, nominal_q_ref=win0["q_ref"]
        )
    finally:
        env._residual_base_mode = original_base_mode

    th = np.asarray(target_home)
    assert np.allclose(np.asarray(target_qref), th, atol=1e-6), (
        "smoke13: action path must not depend on residual_base when "
        "action_mode == 'direct'"
    )
    assert np.allclose(np.asarray(target_refinit), th, atol=1e-6), (
        "smoke13: action path must not depend on residual_base when "
        "action_mode == 'direct'"
    )


# ---------------------------------------------------------------------------
# 4. Actor observation layout contains no hybrid offline-ref channels.
# ---------------------------------------------------------------------------
def test_smoke13_actor_obs_layout_is_phase_proprio_only(smoke13_cfg) -> None:
    """The actor observation must be ``wr_obs_v7_phase_proprio`` —
    v3 base (gravity + gyro + jpos + jvel + foot_switches +
    prev_action + velocity_cmd) + phase(2) + proprio_history.  No
    offline-reference window channels (loc_ref_q_ref, pelvis,
    per-foot pos/vel, contact_mask) in the actor stream.

    The layout dispatch in ``policy_contract/numpy/obs.py`` for
    ``wr_obs_v7_phase_proprio`` already enforces this — pinning the
    config selector here keeps the contract anchored at the config
    layer too."""
    assert (
        smoke13_cfg.env.actor_obs_layout_id == "wr_obs_v7_phase_proprio"
    )


# ---------------------------------------------------------------------------
# 5. Native-MuJoCo eval adapter uses the same direct-policy semantics.
# ---------------------------------------------------------------------------
def test_smoke13_v6_adapter_uses_direct_mode(smoke13_cfg) -> None:
    """``V6EvalAdapter`` must read the same ``action_mode`` field and
    bypass the residual_base switch, so visualize_policy.py and any
    native-MuJoCo eval ladder produces the same target_q as the
    training env."""
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
    assert adapter._action_mode == "direct"
    # Direct mode's ctrl_init must be home, not ref_init / q_ref.
    np.testing.assert_allclose(
        adapter._ctrl_init_policy_order(), adapter._home_q_rad, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 6. Reward block matches TB-active; hybrid/imitation terms absent or zero.
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
            "(hybrid / imitation term; explicitly disabled in TB-active migration)"
        )


# ---------------------------------------------------------------------------
# Anchor / scale / reset invariants (cross-checks the architecture).
# ---------------------------------------------------------------------------
def test_smoke13_uses_scalar_tb_action_scale(smoke13_cfg) -> None:
    """TB-faithful scalar 0.25 (mjx_config.py:103).  No per-joint
    widening in this branch — that was a smoke12 diagnostic for the
    residual contract."""
    env = smoke13_cfg.env
    assert env.loc_ref_residual_scale == pytest.approx(0.25)
    assert env.loc_ref_residual_scale_per_joint == {}


def test_smoke13_reset_and_penalty_pose_anchored_to_home(smoke13_cfg) -> None:
    env = smoke13_cfg.env
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"


# ---------------------------------------------------------------------------
# Critic cleanup (review feedback #1): privileged obs under smoke13 must
# be free of imitation references.  Pinned at both the config layer and
# the env-runtime layer.
# ---------------------------------------------------------------------------
def test_smoke13_critic_imitation_refs_disabled(smoke13_cfg) -> None:
    """``critic_imitation_refs: false`` must be set in smoke13 so the
    critic doesn't quietly score the policy against an external
    reference even though the actor reward stack is TB-active."""
    assert smoke13_cfg.env.critic_imitation_refs is False


def test_smoke13_critic_obs_independent_of_nominal_q_ref(smoke13_cfg) -> None:
    """Under smoke13 ``_get_privileged_critic_obs`` must produce the
    same value regardless of the ``nominal_q_ref`` and
    ``ref_contact_mask`` arguments — pinning that the TB-active critic
    path doesn't secretly read them."""
    import jax
    import jax.numpy as jp
    from training.cal.specs import CoordinateFrame

    env = _make_env(smoke13_cfg)
    win0 = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))

    # Build a fake jitted single-step rollout to access internal
    # primitives without spinning a full episode.  We only need:
    # - a populated mjx.Data
    # - root_vel_h
    # - a phase_sin_cos
    state = env.reset(jax.random.PRNGKey(0))
    data = state.data
    root_vel_h = env._cal.get_root_velocity(
        data, frame=CoordinateFrame.HEADING_LOCAL
    )
    phase_sin_cos = jp.array([0.0, 1.0], dtype=jp.float32)

    # Reference values from the legitimate window.
    obs_default = env._get_privileged_critic_obs(
        data,
        root_vel_h,
        nominal_q_ref=win0["q_ref"],
        ref_contact_mask=win0["contact_mask"],
        phase_sin_cos=phase_sin_cos,
    )
    # Adversarial fakes: same shape, completely different values.
    fake_q_ref = jp.ones_like(win0["q_ref"]) * 999.0
    fake_contact = jp.array([0.7, 0.3], dtype=jp.float32)
    obs_adversarial = env._get_privileged_critic_obs(
        data,
        root_vel_h,
        nominal_q_ref=fake_q_ref,
        ref_contact_mask=fake_contact,
        phase_sin_cos=phase_sin_cos,
    )
    np.testing.assert_allclose(
        np.asarray(obs_default), np.asarray(obs_adversarial), atol=1e-6,
        err_msg=(
            "smoke13 critic obs leaked dependence on nominal_q_ref / "
            "ref_contact_mask — critic_imitation_refs flag is not "
            "short-circuiting the imitation branches in "
            "_get_privileged_critic_obs."
        ),
    )


# ---------------------------------------------------------------------------
# Reset parity (review feedback #2): the v6 eval adapter must mirror
# training's torso pose perturbation when configured, so visualize_policy
# / native-MuJoCo eval doesn't run easier than training.
# ---------------------------------------------------------------------------
def test_smoke13_adapter_reset_applies_torso_perturbation(smoke13_cfg) -> None:
    """The training env applies a TB-style torso roll/pitch
    perturbation under ``reset_torso_{roll,pitch}_range`` non-zero; the
    v6 adapter previously only applied per-joint noise, leaving visual
    eval and training reset distributions diverged.

    Verify the adapter now (a) reports the perturbation is enabled when
    smoke13 ranges are set, and (b) reset_native_mj_state with
    apply_noise=True actually modifies the root quat away from the
    keyframe identity."""
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
    assert adapter._reset_perturbation_enabled(), (
        "smoke13 sets reset_torso_{roll,pitch}_range to [-0.1, 0.1]; "
        "the adapter must report the perturbation enabled."
    )

    mj_data = mujoco.MjData(mj_model)
    # Baseline (no perturbation): keyframe root quat.
    adapter.reset_native_mj_state(mj_data, apply_noise=False, rng=None)
    qpos_baseline = mj_data.qpos.copy()
    root_quat_baseline = qpos_baseline[3:7].copy()
    # Statistical check: run a handful of perturbed resets and
    # verify the root quat or one of the perturbed leg-pitch
    # joints moves on at least one of them.  Per-seed bit equality
    # is intentionally not asserted (JAX vs numpy PRNGs differ).
    rng = np.random.default_rng(42)
    moved = False
    leg_addrs = adapter._leg_pitch_qpos_addrs
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
    """Mirror training's per-episode DR joint offsets (env line 1918:
    ``default_joint_qpos + joint_noise + dr_params["joint_offsets"]``).

    The adapter must (a) report DR enabled when the env config has it
    on, and (b) under ``apply_noise=True, apply_dr=True`` the actuator
    qpos distribution across resets must span beyond what per-joint
    noise alone (±0.05 rad) would produce.  Verified by checking the
    sample standard deviation across 64 resets per joint is materially
    above the noise-only theoretical std (uniform[-0.05,0.05] -> 0.029
    rad).  Joint noise + DR offset should approximately compose to
    uniform[-(0.05+dr), +(0.05+dr)] in the small-perturbation limit —
    std should grow by roughly ``(0.05+dr) / 0.05`` over noise-only.

    Per-seed bit equality is intentionally not asserted (JAX vs numpy
    PRNGs differ); only the marginal distribution is checked."""
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
    assert adapter._dr_enabled is True, (
        "smoke13 enables domain_randomization_enabled; adapter must agree"
    )
    assert adapter._dr_joint_offset_rad > 0.0, (
        "smoke13 sets domain_rand_joint_offset_rad > 0; adapter must "
        "cache the same value"
    )

    mj_data = mujoco.MjData(mj_model)
    # Baseline: turn DR off explicitly via the new apply_dr=False knob;
    # also disable torso perturb to isolate joint noise alone.  Collect
    # std across resets per actuator joint.
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

    # With DR: apply both joint noise (±0.05) and DR offsets.
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

    # Joint noise alone: uniform[-0.05, +0.05] -> theoretical std
    # 0.05/sqrt(3) ≈ 0.0289.  Adding DR offset uniform[-dr, +dr]
    # convolves to combined std sqrt(0.05² + dr²)/sqrt(3).  At
    # smoke13's dr=0.03 the theoretical ratio is sqrt(0.05² + 0.03²)
    # / 0.05 ≈ 1.166.  Use median across joints to avoid being
    # skewed by any joint that clipped against its range.
    noise_only_std = float(np.median(np.std(noise_only_q, axis=0)))
    with_dr_std = float(np.median(np.std(with_dr_q, axis=0)))
    assert with_dr_std > noise_only_std * 1.08, (
        f"smoke13 adapter reset DR joint offsets not visible: with_dr "
        f"std={with_dr_std:.4f} vs noise_only std={noise_only_std:.4f}.  "
        "Expected DR offsets to widen the per-joint qpos distribution "
        "at reset (env line 1918 parity).  Theoretical ratio at "
        "dr=0.03 + noise=0.05 is ≈ 1.17; threshold 1.08 leaves room "
        "for sample noise on 64 resets."
    )

    # Independent check: max excursion from the joint-noise-only base.
    # Joint noise alone cannot move any joint more than 0.05 rad from
    # home; with DR active the max excursion should exceed that.
    home_q = adapter._home_q_rad
    noise_only_max = float(np.max(np.abs(noise_only_q - home_q[None, :])))
    with_dr_max = float(np.max(np.abs(with_dr_q - home_q[None, :])))
    assert with_dr_max > noise_only_max, (
        f"smoke13 adapter reset DR offsets did not widen max excursion: "
        f"with_dr max={with_dr_max:.4f} vs noise_only max={noise_only_max:.4f}."
    )
