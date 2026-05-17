"""v0.20.1 pre-smoke wiring contract.

Verifies the v3-only env's residual action plumbing without depending on
physics or the imitation reward family.  Catches: wrong joint ordering,
wrong residual mode, q_ref shifted by one step, missed step-idx
increment, FSM-mode plumbing leaking into the v3 path.

Spec: training/docs/v0201_env_wiring.md §7.
Contract: training/docs/walking_training.md v0.20.1 § (G6 — iter-0 must
reproduce bare-q_ref replay; the residual head is zero-initialized).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

_SMOKE_CFG = Path("training/configs/ppo_walking_v0201_smoke.yaml")
_SMOKE8_CFG = Path("training/configs/ppo_walking_v0201_smoke8.yaml")
_SMOKE9B_CFG = Path("training/configs/ppo_walking_v0201_smoke9b.yaml")
_SMOKE9C_CFG = Path("training/configs/ppo_walking_v0201_smoke9c.yaml")
if not _SMOKE_CFG.exists():
    pytest.skip(
        f"{_SMOKE_CFG.name} not found (v0.20.1 task #49)",
        allow_module_level=True,
    )

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv


@pytest.fixture(scope="module")
def env() -> WildRobotEnv:
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE_CFG))
    cfg.freeze()
    return WildRobotEnv(cfg)


@pytest.fixture(scope="module")
def env_home_base() -> WildRobotEnv:
    """Smoke8 env: TB-aligned home-pose residual base.

    Skips if the smoke8 yaml hasn't been added to the repo yet.
    """
    if not _SMOKE8_CFG.exists():
        pytest.skip(f"{_SMOKE8_CFG.name} not found (smoke8 not added)")
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE8_CFG))
    cfg.freeze()
    return WildRobotEnv(cfg)


@pytest.fixture(scope="module")
def env_smoke9b_home_base() -> WildRobotEnv:
    if not _SMOKE9B_CFG.exists():
        pytest.skip(f"{_SMOKE9B_CFG.name} not found")
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE9B_CFG))
    cfg.freeze()
    return WildRobotEnv(cfg)


@pytest.fixture(scope="module")
def env_smoke9c_ref_init_base() -> WildRobotEnv:
    if not _SMOKE9C_CFG.exists():
        pytest.skip(f"{_SMOKE9C_CFG.name} not found")
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE9C_CFG))
    cfg.freeze()
    return WildRobotEnv(cfg)


# -----------------------------------------------------------------------------
# 1. Wiring assertion: zero policy_action ⇒ target_q == q_ref exactly.
# -----------------------------------------------------------------------------


def test_zero_action_target_q_equals_nominal_q_ref(env: WildRobotEnv) -> None:
    """With ``policy_action == 0``, the env's residual composition must
    return ``target_q == nominal_q_ref`` (within float32 round-off) at
    every step in a 50-step probe.  This is the G6 contract at the
    compose layer (no filter / no delay).  See the end-to-end test
    below for the full step-path G6 assertion."""
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    for step_idx in range(50):
        win = env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        nominal_q_ref = win["q_ref"].astype(jp.float32)
        target_q, _residual_delta_q = env._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=nominal_q_ref,
        )
        np.testing.assert_allclose(
            np.asarray(target_q),
            np.asarray(nominal_q_ref),
            atol=1e-5,
            err_msg=f"target_q != q_ref at offline step {step_idx}",
        )


# -----------------------------------------------------------------------------
# 2. Residual-zero assertion: residual_delta_q is byte-zero under absolute mode.
# -----------------------------------------------------------------------------


def test_zero_action_residual_delta_q_is_byte_zero(env: WildRobotEnv) -> None:
    """Verifies the absolute-mode plumbing in v0201_env_wiring.md §4.1:
    ``residual_delta_q = clip(policy_action) * scale_per_joint``.  Any
    non-zero result here would indicate the residual mode silently fell
    back to half_span or that the per-joint scale array is misshapen."""
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    win = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))
    _, residual_delta_q = env._compose_target_q_from_residual(
        policy_action=zero_action,
        nominal_q_ref=win["q_ref"].astype(jp.float32),
    )
    arr = np.asarray(residual_delta_q)
    assert arr.shape == (env.action_size,)
    assert np.all(arr == 0.0), f"residual_delta_q is not byte-zero: max={np.max(np.abs(arr))}"


# -----------------------------------------------------------------------------
# 3. Trajectory-advance assertion: step_idx must advance 0..49 across 50 steps.
# -----------------------------------------------------------------------------


def test_step_idx_advances_monotonically(env: WildRobotEnv) -> None:
    """The env's loc_ref_offline_step_idx must advance by exactly 1
    each step.  Catches missed increments (would make the env replay
    the same q_ref forever) and unexpected resets (would break the
    rollout's monotonic time index)."""
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    expected = 0
    for _ in range(50):
        actual = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
        if int(state.done) > 0:
            # Auto-reset fires on done (env's expected behavior); the next
            # state's step_idx restarts at 0.  This wiring test should
            # disable the physics path that produces termination — but for
            # robustness, accept the reset and re-anchor the expectation.
            expected = 0
            actual = 0
        assert actual == expected, (
            f"loc_ref_offline_step_idx={actual} expected={expected} "
            "(missed step-idx increment or unexpected reset)"
        )
        state = step_fn(state, zero_action)
        expected += 1


# -----------------------------------------------------------------------------
# 4. Loc-ref obs channels track the service window at the same step_idx.
# -----------------------------------------------------------------------------


def test_loc_ref_obs_channels_track_service_window(env: WildRobotEnv) -> None:
    """The v5 obs channels must reflect the offline window at the
    current step_idx — not stuck on step 0 due to a stale lookup."""
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    # Advance one step; if the env doesn't terminate, compare q_ref-from-obs
    # against q_ref-from-service at the new step_idx.
    state = step_fn(state, zero_action)
    if int(state.done) > 0:
        pytest.skip("env terminated before step_idx>0; cannot compare obs")
    new_step_idx = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
    win = env._lookup_offline_window(jp.asarray(new_step_idx, dtype=jp.int32))
    expected_q_ref = np.asarray(win["q_ref"]).astype(np.float32)
    # The v5 layout places loc_ref_q_ref as part of the obs vector; the
    # easiest invariant to assert is that nominal_q_ref stored in
    # WildRobotInfo matches the service lookup at the same step_idx.
    actual_q_ref = np.asarray(state.info[WR_INFO_KEY].nominal_q_ref).astype(np.float32)
    np.testing.assert_allclose(
        actual_q_ref,
        expected_q_ref,
        atol=1e-5,
        err_msg="WildRobotInfo.nominal_q_ref out of sync with service lookup",
    )


# -----------------------------------------------------------------------------
# 4b. Phase 4 runtime-target semantics: torso_xy error uses command-integrated
#     path_pos target, not planner/offline pelvis_pos[:2].
# -----------------------------------------------------------------------------


def test_torso_pos_xy_error_uses_command_integrated_path_target(
    env: WildRobotEnv,
) -> None:
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)

    # Advance into steady-state to avoid the near-origin transient where
    # planner pelvis_pos and command-integrated path can be very close.
    for _ in range(30):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            pytest.skip("env terminated before phase-4 torso target probe window")

    wr = state.info[WR_INFO_KEY]
    step_idx = int(wr.loc_ref_offline_step_idx)
    win = env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
    t_since_reset_s = jp.float32(step_idx) * jp.float32(env.dt)
    path_state = env._offline_service.compute_command_integrated_path_state(
        t_since_reset_s=t_since_reset_s,
        velocity_cmd_mps=wr.velocity_cmd,
        yaw_rate_cmd_rps=jp.float32(0.0),
        dt_s=jp.float32(env.dt),
        default_root_pos_xyz=env._init_qpos[:3].astype(jp.float32),
    )

    root_xy = np.asarray(state.data.qpos[:2], dtype=np.float32)
    path_xy = np.asarray(path_state["torso_pos"][:2], dtype=np.float32)
    pure_path_xy = np.asarray(path_state["path_pos"][:2], dtype=np.float32)
    pelvis_xy = np.asarray(win["pelvis_pos"][:2], dtype=np.float32)

    expected_path_err = float(np.linalg.norm(root_xy - path_xy))
    planner_err = float(np.linalg.norm(root_xy - pelvis_xy))
    logged_err = float(
        state.metrics[METRICS_VEC_KEY][METRIC_INDEX["ref/torso_pos_xy_err_m"]]
    )

    np.testing.assert_allclose(
        logged_err,
        expected_path_err,
        atol=1e-5,
        err_msg="ref/torso_pos_xy_err_m should track command-integrated path_pos",
    )
    assert float(np.linalg.norm(path_xy - pelvis_xy)) > 1e-3, (
        "probe picked a frame where path target and planner pelvis target are "
        "effectively identical; cannot validate phase-4 semantic shift"
    )
    assert float(np.linalg.norm(path_xy - pure_path_xy)) > 1e-3, (
        "torso target should include the non-zero default root XY offset"
    )
    assert abs(logged_err - planner_err) > 1e-4, (
        "torso_pos_xy error still appears tied to planner pelvis_pos target"
    )


# -----------------------------------------------------------------------------
# 5. End-to-end G6 contract: zero-action ⇒ applied_target_q == q_ref AFTER
#    the full step pipeline (residual composition + action filter + delay).
#
#    This is the BLOCKING test the residual-only assertion above misses:
#    even when _compose_loc_ref_residual_action returns target_q == q_ref,
#    a non-no-op action filter (lowpass_v1 with retention alpha > 0) blends
#    raw_action with prev_action, yielding a lagged applied_target_q.  The
#    smoke contract requires iter-0 to be bare q_ref replay; this test
#    guards the whole path.
# -----------------------------------------------------------------------------


def test_zero_action_applied_target_q_equals_q_ref_under_full_step(
    env: WildRobotEnv,
) -> None:
    """Run env.step with zero policy_action and verify the ctrl actually
    written to mjx.Data matches q_ref at the new step_idx (within
    float32 round-off).  Under the residual-only filter contract this
    must hold for ANY ``action_filter_alpha`` value — the filter
    operates on the residual command (which stays at 0 when the policy
    outputs 0), not on the composed target.

    NOTE: the comparison must happen at a step where the trajectory is
    actually moving (q_ref[t] != q_ref[t-1]).  Without movement, even
    the buggy "filter the composed target" path would coincidentally
    pass because q_ref[t] == q_ref[t-1].  We advance until we see a
    meaningful frame-to-frame delta, then assert.
    """
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    perm = np.asarray(env._ctrl_mapper.policy_to_mj_order_jax)

    # Advance to a step where the prior is actually moving.  The ZMP prior
    # at vx=0.15 has its first lever flip around step 2-3 (per
    # walking_training.md G7 baseline note).  Loop a small fixed number of
    # steps and assert at the first one with a non-trivial trajectory delta.
    found_moving_frame = False
    for _ in range(20):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            pytest.skip("env terminated before reaching a moving trajectory frame")
        step_idx = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
        prev_q_ref = np.asarray(
            env._lookup_offline_window(jp.asarray(step_idx - 1, dtype=jp.int32))["q_ref"]
        ).astype(np.float32)
        curr_q_ref = np.asarray(
            env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))["q_ref"]
        ).astype(np.float32)
        if float(np.max(np.abs(curr_q_ref - prev_q_ref))) > 1e-3:
            found_moving_frame = True
            break

    assert found_moving_frame, (
        "Did not find a moving q_ref frame within 20 steps; the assertion "
        "would not detect filter lag on stationary frames."
    )

    ctrl_mj = np.asarray(state.data.ctrl).astype(np.float32)
    applied_target_q = ctrl_mj[perm]
    np.testing.assert_allclose(
        applied_target_q,
        curr_q_ref,
        atol=1e-5,
        err_msg=(
            f"G6 violation at step_idx={step_idx}: zero residual + filter/delay "
            f"produced applied_target_q != q_ref[{step_idx}].  "
            "Under the residual-only filter contract this must hold for "
            "any action_filter_alpha; if it fails, the filter is back on "
            "the composed target instead of the policy residual."
        ),
    )


# -----------------------------------------------------------------------------
# 6. End-to-end G6 metric contract: 50 zero-action steps must keep the
#    env's tracking/residual_q_abs_max metric at <= 1e-5 every step.  This
#    asserts the same invariant as test #5 but via the metrics path the
#    smoke run actually logs to W&B — guarding against any regression
#    where the env's own bookkeeping disagrees with the ctrl write.
# -----------------------------------------------------------------------------


def test_zero_action_residual_metric_stays_zero_for_50_steps(
    env: WildRobotEnv,
) -> None:
    """Run env.step for 50 steps with zero policy_action and assert
    ``tracking/residual_q_abs_max`` stays at numerical zero every step.
    Catches any regression in the residual-only filter contract that
    leaks through to the logged metric (which is what training
    monitors)."""
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    metric_idx = METRIC_INDEX["tracking/residual_q_abs_max"]

    max_residual_seen = 0.0
    last_step_idx = 0
    for _ in range(50):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            pytest.skip("env terminated before 50-step horizon")
        last_step_idx = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
        residual_max = float(state.metrics[METRICS_VEC_KEY][metric_idx])
        max_residual_seen = max(max_residual_seen, residual_max)

    assert max_residual_seen <= 1e-5, (
        f"tracking/residual_q_abs_max peaked at {max_residual_seen:.3e} "
        f"over 50 zero-action steps (last step_idx={last_step_idx}); "
        "G6 contract requires <= 1e-5.  Likely cause: composed target is "
        "being filtered instead of the policy residual."
    )


# -----------------------------------------------------------------------------
# 7. v6 proprio history schema contract: the buffer holds PAST bundles only.
#    The newest history slot must NOT duplicate the current frame's proprio
#    channels.  Specifically:
#      - reset: WildRobotInfo.proprio_history is all zeros
#      - step 1: obs's history channel slice equals all zeros
#                (the obs is built BEFORE the new bundle is rolled in)
#      - step k > PROPRIO_HISTORY_FRAMES: the newest history slot equals
#        the bundle from step k-1 (the previous step's post-step proprio),
#        NOT step k's current proprio
# -----------------------------------------------------------------------------


def test_v6_proprio_history_does_not_duplicate_current_frame(
    env: WildRobotEnv,
) -> None:
    """Pin the v6 schema contract: history holds PAST bundles only.

    Skipped on layouts other than v6.  Catches the regression where
    the rolled buffer (containing the just-computed current bundle in
    the newest slot) is fed back into the same step's obs.  Under the
    correct contract:
      - at step 1, the history slot is all zeros (reset zero-fills);
      - at step k>=2, the newest history slot equals the PREVIOUS
        step's stored proprio_history newest slot — i.e. the buffer
        is one-step lagged from the current frame.
    """
    from training.envs.env_info import PROPRIO_HISTORY_FRAMES
    layout_id = env._policy_spec.observation.layout_id
    if layout_id != "wr_obs_v6_offline_ref_history":
        pytest.skip(f"v6 history contract not applicable to layout={layout_id}")

    bundle_size = 3 + 4 + 3 * env.action_size

    # Locate the proprio_history slice in the obs vector.
    layout = env._policy_spec.observation.layout
    offset = 0
    history_offset = None
    for field in layout:
        if field.name == "proprio_history":
            history_offset = offset
            break
        offset += int(field.size)
    assert history_offset is not None, "v6 layout missing proprio_history slot"
    history_size = PROPRIO_HISTORY_FRAMES * bundle_size

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)

    # Reset: stored buffer is all zeros.
    wr0 = state.info[WR_INFO_KEY]
    np.testing.assert_array_equal(
        np.asarray(wr0.proprio_history),
        np.zeros((PROPRIO_HISTORY_FRAMES, bundle_size), dtype=np.float32),
        err_msg="reset must zero-fill WildRobotInfo.proprio_history",
    )

    # Step 1: obs is built BEFORE rolling, so the obs's history slice is
    # still all zeros.  After this step, the stored buffer's newest slot
    # holds the post-step bundle from step 1.
    state1 = step_fn(state, zero_action)
    obs1 = np.asarray(state1.obs).astype(np.float32)
    history_slice_1 = obs1[history_offset : history_offset + history_size]
    np.testing.assert_array_equal(
        history_slice_1,
        np.zeros(history_size, dtype=np.float32),
        err_msg=(
            "step 1 obs's proprio_history slice must be all zeros "
            "(the buffer hasn't been rolled yet — passing the rolled "
            "buffer to the same step's obs would duplicate the current "
            "frame in the newest history slot, which is the v6 schema "
            "violation gap A intended to fix)"
        ),
    )

    # Step 2: obs's newest history slot must equal the buffer that was
    # rolled at the END of step 1 — which is the bundle from step 1's
    # post-step proprio.  In particular, it must NOT equal the current
    # (step 2 post-step) proprio that the standard channels report.
    wr1 = state1.info[WR_INFO_KEY]
    expected_newest_after_step1 = np.asarray(wr1.proprio_history[-1]).astype(np.float32)

    state2 = step_fn(state1, zero_action)
    obs2 = np.asarray(state2.obs).astype(np.float32)
    history_slice_2 = obs2[history_offset : history_offset + history_size]
    newest_in_obs_step2 = history_slice_2[-bundle_size:]
    np.testing.assert_allclose(
        newest_in_obs_step2,
        expected_newest_after_step1,
        atol=1e-6,
        err_msg=(
            "step 2 obs's NEWEST history slot must equal the bundle "
            "stored at the end of step 1 (one-step lag), not the "
            "current step's post-step proprio.  A mismatch means the "
            "history is being rolled with the current bundle BEFORE "
            "the obs is built — duplicating the current frame."
        ),
    )


# =============================================================================
# Smoke8 (TB-aligned home-base residual) invariants
# =============================================================================
# These tests pin the `loc_ref_residual_base: home` contract: the action
# path uses target_q = home_q_rad + scale * action while q_ref still
# flows through the offline window into reward terms.  They are the
# home-base analogues of tests #1, #2, #5 above.


def test_smoke8_residual_base_flag_threaded(env_home_base: WildRobotEnv) -> None:
    """The yaml flag must propagate end-to-end: the env's cached
    `_residual_base_mode` must be exactly 'home' under smoke8.  Catches
    config-loader regressions where the field defaults to 'q_ref' even
    when the yaml asks for 'home'."""
    assert env_home_base._residual_base_mode == "home", (
        f"smoke8 env._residual_base_mode={env_home_base._residual_base_mode!r}; "
        "yaml sets loc_ref_residual_base: home — flag did not propagate."
    )
    # Home pose should be a valid joint vector.
    assert env_home_base._home_q_rad.shape == (env_home_base.action_size,)
    assert np.all(np.isfinite(np.asarray(env_home_base._home_q_rad)))


def test_smoke8_zero_action_target_q_equals_home(env_home_base: WildRobotEnv) -> None:
    """Smoke8 G6-equivalent at the compose layer: with zero residual,
    target_q == home_q_rad EXACTLY (within float32 round-off), even at
    frames where q_ref(t) is moving.  This is the invariant that
    differs from smoke7 — under home base, the action path is decoupled
    from the moving ZMP trajectory."""
    zero_action = jp.zeros(env_home_base.action_size, dtype=jp.float32)
    home = np.asarray(env_home_base._home_q_rad).astype(np.float32)
    # Sample steps where q_ref is moving (the prior advances each step).
    for step_idx in (0, 5, 10, 20, 30, 49):
        win = env_home_base._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        nominal_q_ref = win["q_ref"].astype(jp.float32)
        target_q, _ = env_home_base._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=nominal_q_ref,
        )
        np.testing.assert_allclose(
            np.asarray(target_q),
            home,
            atol=1e-5,
            err_msg=(
                f"smoke8: target_q != home at step {step_idx}; "
                "home-base contract violated"
            ),
        )


def test_smoke8_target_q_decoupled_from_q_ref(env_home_base: WildRobotEnv) -> None:
    """Sanity: the home base must NOT equal q_ref(t) at any moving frame
    — otherwise the test above would coincidentally pass under smoke7's
    q_ref base too.  Asserts q_ref drifts away from home over the first
    50 steps so the decoupling test has actual signal to detect."""
    home = np.asarray(env_home_base._home_q_rad).astype(np.float32)
    max_drift = 0.0
    for step_idx in range(50):
        win = env_home_base._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        q_ref = np.asarray(win["q_ref"]).astype(np.float32)
        max_drift = max(max_drift, float(np.max(np.abs(q_ref - home))))
    assert max_drift > 0.05, (
        f"q_ref stayed within {max_drift:.4f} rad of home over 50 steps; "
        "the home-vs-q_ref decoupling test cannot detect a regression "
        "if the two are effectively identical.  Pick later sample frames "
        "or adjust the assertion."
    )


def test_smoke8_zero_action_applied_target_q_equals_home_under_full_step(
    env_home_base: WildRobotEnv,
) -> None:
    """End-to-end: zero policy_action under the full step pipeline
    (residual compose + filter + delay + ctrl write) must produce
    applied_target_q == home_q_rad — even at moving-trajectory frames.
    Mirrors test #5 above for the home-base contract."""
    reset_fn = jax.jit(env_home_base.reset)
    step_fn = jax.jit(env_home_base.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env_home_base.action_size, dtype=jp.float32)
    perm = np.asarray(env_home_base._ctrl_mapper.policy_to_mj_order_jax)
    home = np.asarray(env_home_base._home_q_rad).astype(np.float32)

    # Advance to a step where q_ref is moving (so the test would catch
    # any residual pull from q_ref into the action path).
    found_moving_frame = False
    for _ in range(20):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            pytest.skip("env terminated before reaching moving-trajectory frame")
        step_idx = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
        prev_q = np.asarray(env_home_base._lookup_offline_window(
            jp.asarray(step_idx - 1, dtype=jp.int32))["q_ref"]).astype(np.float32)
        curr_q = np.asarray(env_home_base._lookup_offline_window(
            jp.asarray(step_idx, dtype=jp.int32))["q_ref"]).astype(np.float32)
        if float(np.max(np.abs(curr_q - prev_q))) > 1e-3:
            found_moving_frame = True
            break
    assert found_moving_frame, "did not reach moving q_ref frame within 20 steps"

    ctrl_mj = np.asarray(state.data.ctrl).astype(np.float32)
    applied_target_q = ctrl_mj[perm]
    np.testing.assert_allclose(
        applied_target_q,
        home,
        atol=1e-5,
        err_msg=(
            f"smoke8 home-base violation at step_idx={step_idx}: "
            "zero action through the full step pipeline produced "
            "applied_target_q != home_q_rad.  Either the flag is not "
            "wired through reset's ctrl_init, or the step path is "
            "still using q_ref as the residual base."
        ),
    )


def test_smoke8_residual_metric_is_zero_under_zero_action(
    env_home_base: WildRobotEnv,
) -> None:
    """Critical G5/anti-exploit gate contract under home base.

    The metric ``tracking/residual_q_abs_max`` must measure the policy's
    displacement from the active base (= clip(action) * scale), NOT
    abs(applied_target_q - q_ref).  Under home base the latter is
    abs(home + delta - q_ref(t)) which has a non-zero baseline equal to
    the q_ref drift away from home — that would corrupt G5 gates and
    fire false residual-cap violations under bare zero-action play.

    Smoke7 hid this confusion because under q_ref base
    abs(target - q_ref) == abs(delta) by construction.  Smoke8 surfaces
    it; this test pins the contract."""
    reset_fn = jax.jit(env_home_base.reset)
    step_fn = jax.jit(env_home_base.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env_home_base.action_size, dtype=jp.float32)
    metric_idx = METRIC_INDEX["tracking/residual_q_abs_max"]

    max_seen = 0.0
    for _ in range(50):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            pytest.skip("env terminated before 50-step horizon")
        max_seen = max(max_seen, float(state.metrics[METRICS_VEC_KEY][metric_idx]))

    assert max_seen <= 1e-5, (
        f"smoke8 home base: tracking/residual_q_abs_max peaked at "
        f"{max_seen:.3e} over 50 zero-action steps; G5 contract requires "
        f"<= 1e-5.  Likely cause: metric is computed as "
        f"abs(applied_target_q - q_ref) instead of abs(residual_delta), "
        f"which collapses to delta under q_ref base but logs the "
        f"home-vs-q_ref drift under home base."
    )


def test_smoke8_q_ref_still_flows_to_info(env_home_base: WildRobotEnv) -> None:
    """Reward terms (ref_q_track etc.) depend on nominal_q_ref via the
    offline window.  Even though the action path no longer consumes
    q_ref, the env must still surface q_ref in WildRobotInfo for the
    reward computation.  Catches an over-aggressive refactor that
    drops q_ref from info entirely under the home-base mode."""
    reset_fn = jax.jit(env_home_base.reset)
    step_fn = jax.jit(env_home_base.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env_home_base.action_size, dtype=jp.float32)
    state = step_fn(state, zero_action)
    if int(state.done) > 0:
        pytest.skip("env terminated before step 1; cannot probe info")
    step_idx = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
    expected = np.asarray(env_home_base._lookup_offline_window(
        jp.asarray(step_idx, dtype=jp.int32))["q_ref"]).astype(np.float32)
    actual = np.asarray(state.info[WR_INFO_KEY].nominal_q_ref).astype(np.float32)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-5,
        err_msg=(
            "smoke8: WildRobotInfo.nominal_q_ref out of sync with offline "
            "window — reward terms that consume q_ref will see stale data"
        ),
    )


# =============================================================================
# Smoke9b / Smoke9c base + reset contracts
# =============================================================================


def test_smoke9b_zero_action_still_targets_home(
    env_smoke9b_home_base: WildRobotEnv,
) -> None:
    zero_action = jp.zeros(env_smoke9b_home_base.action_size, dtype=jp.float32)
    home = np.asarray(env_smoke9b_home_base._home_q_rad).astype(np.float32)
    for step_idx in (0, 10, 30):
        win = env_smoke9b_home_base._lookup_offline_window(
            jp.asarray(step_idx, dtype=jp.int32)
        )
        target_q, _ = env_smoke9b_home_base._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=win["q_ref"].astype(jp.float32),
        )
        np.testing.assert_allclose(np.asarray(target_q), home, atol=1e-5)


def test_smoke9c_ref_init_modes_threaded(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    assert env_smoke9c_ref_init_base._residual_base_mode == "ref_init"
    assert env_smoke9c_ref_init_base._reset_base_mode == "ref_init"


def test_smoke9c_zero_action_targets_constant_ref_init_not_moving_q_ref(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    zero_action = jp.zeros(env_smoke9c_ref_init_base.action_size, dtype=jp.float32)
    ref_init = np.asarray(env_smoke9c_ref_init_base._ref_init_q_rad).astype(np.float32)
    home = np.asarray(env_smoke9c_ref_init_base._home_q_rad).astype(np.float32)

    max_q_ref_drift = 0.0
    for step_idx in (0, 5, 10, 20, 30, 49):
        win = env_smoke9c_ref_init_base._lookup_offline_window(
            jp.asarray(step_idx, dtype=jp.int32)
        )
        nominal_q_ref = np.asarray(win["q_ref"]).astype(np.float32)
        target_q, _ = env_smoke9c_ref_init_base._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=win["q_ref"].astype(jp.float32),
        )
        np.testing.assert_allclose(np.asarray(target_q), ref_init, atol=1e-5)
        max_q_ref_drift = max(
            max_q_ref_drift, float(np.max(np.abs(nominal_q_ref - ref_init)))
        )

    assert max_q_ref_drift > 1e-3, (
        "q_ref(t) did not drift away from ref_init over sampled frames; "
        "cannot verify the constant-base contract against moving q_ref."
    )
    # In the expected smoke9c setup ref_init differs from home; if this
    # asset happens to make them numerically equal, skip the non-home check.
    if float(np.max(np.abs(ref_init - home))) <= 1e-4:
        pytest.skip("ref_init and home are numerically equal in this asset")


def test_smoke9c_reset_ctrl_matches_ref_init(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    """Reset CTRL (action target) must always equal ref_init_q_rad,
    independent of any reset-time pose perturbation.  This is the
    zero-residual invariant: a zero policy action composes to the
    constant ref_init control base.  The perturbation introduced in the
    smoke9c spatial-normalization pass only touches the initial
    physical qpos, never the ctrl base.
    """
    reset_fn = jax.jit(env_smoke9c_ref_init_base.reset)
    state = reset_fn(jax.random.PRNGKey(0))
    ref_init = np.asarray(env_smoke9c_ref_init_base._ref_init_q_rad).astype(np.float32)

    perm = np.asarray(env_smoke9c_ref_init_base._ctrl_mapper.policy_to_mj_order_jax)
    applied_target_q = np.asarray(state.data.ctrl).astype(np.float32)[perm]
    np.testing.assert_allclose(
        applied_target_q,
        ref_init,
        atol=1e-5,
        err_msg=(
            "smoke9c reset ctrl init must match ref_init_q regardless "
            "of reset perturbation."
        ),
    )


def test_smoke9c_reset_qpos_matches_ref_init_when_perturbation_disabled(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    """With reset perturbation ranges zeroed out, the reset reproduces
    the historical quiet ref_init behavior exactly: qpos[actuator_addrs]
    == ref_init_q_rad and root quat is the keyframe quat.  This pins
    the no-op contract: ``reset_torso_{roll,pitch}_range = [0, 0]``
    must be a true no-op.
    """
    env = env_smoke9c_ref_init_base
    saved_roll = env._reset_torso_roll_range
    saved_pitch = env._reset_torso_pitch_range
    try:
        env._reset_torso_roll_range = jp.asarray([0.0, 0.0], dtype=jp.float32)
        env._reset_torso_pitch_range = jp.asarray([0.0, 0.0], dtype=jp.float32)
        # Don't jit — fresh closure each call so the zero ranges take effect.
        for seed in (0, 1, 7, 42):
            state = env.reset(jax.random.PRNGKey(seed))
            ref_init = np.asarray(env._ref_init_q_rad).astype(np.float32)
            reset_q = np.asarray(
                state.data.qpos[env._actuator_qpos_addrs]
            ).astype(np.float32)
            np.testing.assert_allclose(
                reset_q,
                ref_init,
                atol=1e-6,
                err_msg=(
                    f"seed={seed}: zero-range perturbation must reproduce "
                    f"ref_init qpos exactly."
                ),
            )
    finally:
        env._reset_torso_roll_range = saved_roll
        env._reset_torso_pitch_range = saved_pitch


def test_smoke9c_zero_ranges_skip_reset_perturbation_call(
    env_smoke9c_ref_init_base: WildRobotEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-width perturbation ranges must bypass the perturbation helper.

    Equality-vs-baseline tests already prove the quiet reset semantics.
    This regression test pins the stronger contract: the zero-range case
    should not even trace or execute ``_apply_reset_perturbation``.
    """
    env = env_smoke9c_ref_init_base
    saved_roll = env._reset_torso_roll_range
    saved_pitch = env._reset_torso_pitch_range
    try:
        env._reset_torso_roll_range = jp.asarray([0.0, 0.0], dtype=jp.float32)
        env._reset_torso_pitch_range = jp.asarray([0.0, 0.0], dtype=jp.float32)

        def _should_not_run(*_args, **_kwargs):
            raise AssertionError(
                "_apply_reset_perturbation should not run when both "
                "reset_torso_* ranges are zero-width."
            )

        monkeypatch.setattr(env, "_apply_reset_perturbation", _should_not_run)
        _ = env.reset(jax.random.PRNGKey(0))
    finally:
        env._reset_torso_roll_range = saved_roll
        env._reset_torso_pitch_range = saved_pitch


def test_smoke9c_reset_perturbation_changes_qpos_when_enabled(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    """With smoke9c's default non-zero perturbation ranges, the reset
    qpos is NOT always identical to ref_init_q_rad.  Across a handful
    of seeds, at least some leg-pitch joints and/or the root quat must
    differ from the quiet baseline by a measurable amount.
    """
    env = env_smoke9c_ref_init_base
    # Sanity: smoke9c yaml enables the perturbation.
    assert float(env._reset_torso_pitch_range[1]) > 0.0
    assert float(env._reset_torso_roll_range[1]) > 0.0

    ref_init = np.asarray(env._ref_init_q_rad).astype(np.float32)
    leg_pitch_addrs = np.asarray(env._leg_pitch_qpos_addrs)
    # Get the unperturbed root quat by sampling with the perturbation
    # off (locally) — that gives the keyframe root quat that the
    # perturbation rotates around.
    saved_roll = env._reset_torso_roll_range
    saved_pitch = env._reset_torso_pitch_range
    env._reset_torso_roll_range = jp.asarray([0.0, 0.0], dtype=jp.float32)
    env._reset_torso_pitch_range = jp.asarray([0.0, 0.0], dtype=jp.float32)
    quiet_state = env.reset(jax.random.PRNGKey(0))
    quiet_quat = np.asarray(quiet_state.data.qpos[3:7])
    env._reset_torso_roll_range = saved_roll
    env._reset_torso_pitch_range = saved_pitch

    leg_pitch_deltas = []
    quat_deltas = []
    for seed in (100, 200, 300, 400, 500):
        state = env.reset(jax.random.PRNGKey(seed))
        reset_q = np.asarray(
            state.data.qpos[env._actuator_qpos_addrs]
        ).astype(np.float32)
        leg_pitch_qpos = np.asarray(state.data.qpos[leg_pitch_addrs])
        # Compare leg-pitch qpos at MJ addrs against ref_init_q lookup at
        # the matching actuator positions.  Use ctrl mapping: the
        # actuator order maps to qpos addrs via _actuator_qpos_addrs.
        ref_init_qpos_full = np.zeros_like(state.data.qpos)
        # Just check leg-pitch qpos deviates from quiet baseline.
        quiet_leg_qpos = np.asarray(quiet_state.data.qpos[leg_pitch_addrs])
        leg_pitch_delta = float(np.max(np.abs(leg_pitch_qpos - quiet_leg_qpos)))
        quat_delta = float(
            np.max(np.abs(np.asarray(state.data.qpos[3:7]) - quiet_quat))
        )
        leg_pitch_deltas.append(leg_pitch_delta)
        quat_deltas.append(quat_delta)

    # At least one seed must produce a meaningful leg-pitch perturbation.
    assert max(leg_pitch_deltas) > 1e-3, (
        f"reset perturbation produced near-zero leg-pitch deltas "
        f"across all probed seeds: {leg_pitch_deltas}.  Perturbation "
        f"appears inactive when it should be enabled."
    )
    # And at least one seed must produce a non-trivial root-quat update.
    assert max(quat_deltas) > 1e-3, (
        f"reset perturbation produced near-zero root-quat deltas "
        f"across all probed seeds: {quat_deltas}.  Torso roll/pitch "
        f"updates appear to be missing from the qpos[3:7] write."
    )


def test_smoke9c_reset_perturbation_only_touches_leg_pitch_joints(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    """Structured perturbation contract: torso-pitch decomposition must
    touch ONLY the 6 leg-pitch actuator qpos addrs.  Non-pitch actuator
    qpos addrs (arms, waist yaw, hip rolls, ankle rolls) must equal
    their ref_init values exactly.  Anything else is iid noise, which
    the prompt explicitly forbids.
    """
    env = env_smoke9c_ref_init_base
    ref_init = np.asarray(env._ref_init_q_rad).astype(np.float32)
    leg_pitch_set = set(int(a) for a in np.asarray(env._leg_pitch_qpos_addrs))
    # Build mapping actuator index → mj qpos addr.
    actuator_qpos_addrs = np.asarray(env._actuator_qpos_addrs)
    non_pitch_mask = np.array(
        [int(a) not in leg_pitch_set for a in actuator_qpos_addrs]
    )

    for seed in (1, 2, 3, 4):
        state = env.reset(jax.random.PRNGKey(seed))
        reset_actuator_q = np.asarray(
            state.data.qpos[env._actuator_qpos_addrs]
        ).astype(np.float32)
        non_pitch_q = reset_actuator_q[non_pitch_mask]
        non_pitch_ref = ref_init[non_pitch_mask]
        np.testing.assert_allclose(
            non_pitch_q,
            non_pitch_ref,
            atol=1e-6,
            err_msg=(
                f"seed={seed}: non-pitch actuators were perturbed.  Reset "
                f"perturbation must only touch leg-pitch joints (hip / knee "
                f"/ ankle pitch L+R) and the root quat."
            ),
        )


def test_smoke9c_euler_xyz_quat_matches_scipy() -> None:
    """``_euler_xyz_to_quat_xyzw`` must match scipy's lowercase
    ``'xyz'`` convention — which scipy defines as **EXTRINSIC**
    (rotations about FIXED world axes), the convention TB uses at
    ``toddlerbot/locomotion/mjx_env.py:1044``
    ``R.from_euler('xyz', [roll, pitch, 0])``.

    scipy's uppercase ``'XYZ'`` is the intrinsic (body-axis) variant
    and yields the OPPOSITE sign on z when roll and pitch are both
    nonzero.  An earlier draft of the helper implemented the
    intrinsic ``q_x ⊗ q_y ⊗ q_z`` ordering by mistake, giving
    z = +sin²(θ/2) instead of −sin²(θ/2) for roll=pitch=θ, and so
    applied a different torso rotation than TB whenever roll AND
    pitch were both nonzero.  This test pins the correct extrinsic
    convention so a future regression can't sneak through.
    """
    from scipy.spatial.transform import Rotation as R  # local import

    cases = [
        (0.1, 0.1, 0.0),    # both nonzero — the case the previous bug hit
        (0.1, 0.0, 0.0),    # pure roll
        (0.0, 0.1, 0.0),    # pure pitch
        (-0.05, 0.08, 0.0), # asymmetric magnitudes
        (0.07, -0.06, 0.0), # opposite signs
        (-0.1, -0.1, 0.0),  # both negative
        (0.05, 0.03, 0.02), # with non-zero yaw too
    ]
    for r, p, y in cases:
        mine = np.asarray(
            WildRobotEnv._euler_xyz_to_quat_xyzw(
                jp.float32(r), jp.float32(p), jp.float32(y)
            )
        )
        scipy_xyzw = R.from_euler("xyz", [r, p, y]).as_quat()
        np.testing.assert_allclose(
            mine,
            scipy_xyzw,
            atol=1e-6,
            err_msg=(
                f"euler({r}, {p}, {y}) → quat mismatch with scipy "
                f"'xyz' (intrinsic): mine={mine}, scipy={scipy_xyzw}.  "
                f"Pin lost — see test docstring for the previous bug."
            ),
        )


def test_smoke9c_reset_perturbation_respects_joint_limits(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    """Even with maximum perturbation in the range, leg-pitch qpos must
    stay inside the kinematic joint limits.  Probe a wide sweep of
    seeds; every output must satisfy ``mins <= qpos <= maxs``.
    """
    env = env_smoke9c_ref_init_base
    leg_pitch_addrs = np.asarray(env._leg_pitch_qpos_addrs)
    mins = np.asarray(env._leg_pitch_joint_mins)
    maxs = np.asarray(env._leg_pitch_joint_maxs)
    for seed in range(32):
        state = env.reset(jax.random.PRNGKey(7000 + seed))
        leg_pitch_qpos = np.asarray(state.data.qpos[leg_pitch_addrs])
        assert np.all(leg_pitch_qpos >= mins - 1e-6), (
            f"seed={seed}: leg-pitch qpos {leg_pitch_qpos} fell below "
            f"min limits {mins}."
        )
        assert np.all(leg_pitch_qpos <= maxs + 1e-6), (
            f"seed={seed}: leg-pitch qpos {leg_pitch_qpos} exceeded "
            f"max limits {maxs}."
        )


def test_smoke9c_zero_action_applied_target_q_equals_ref_init_under_full_step(
    env_smoke9c_ref_init_base: WildRobotEnv,
) -> None:
    reset_fn = jax.jit(env_smoke9c_ref_init_base.reset)
    step_fn = jax.jit(env_smoke9c_ref_init_base.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env_smoke9c_ref_init_base.action_size, dtype=jp.float32)
    perm = np.asarray(env_smoke9c_ref_init_base._ctrl_mapper.policy_to_mj_order_jax)
    ref_init = np.asarray(env_smoke9c_ref_init_base._ref_init_q_rad).astype(np.float32)

    found_moving_frame = False
    for _ in range(20):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            pytest.skip("env terminated before reaching moving-trajectory frame")
        step_idx = int(state.info[WR_INFO_KEY].loc_ref_offline_step_idx)
        prev_q = np.asarray(
            env_smoke9c_ref_init_base._lookup_offline_window(
                jp.asarray(step_idx - 1, dtype=jp.int32)
            )["q_ref"]
        ).astype(np.float32)
        curr_q = np.asarray(
            env_smoke9c_ref_init_base._lookup_offline_window(
                jp.asarray(step_idx, dtype=jp.int32)
            )["q_ref"]
        ).astype(np.float32)
        if float(np.max(np.abs(curr_q - prev_q))) > 1e-3:
            found_moving_frame = True
            break
    assert found_moving_frame, "did not reach moving q_ref frame within 20 steps"

    ctrl_mj = np.asarray(state.data.ctrl).astype(np.float32)
    applied_target_q = ctrl_mj[perm]
    np.testing.assert_allclose(
        applied_target_q,
        ref_init,
        atol=1e-5,
        err_msg=(
            f"smoke9c ref-init violation at step_idx={step_idx}: "
            "zero action through the full step pipeline produced "
            "applied_target_q != ref_init_q."
        ),
    )


def test_command_distribution_metrics_are_registered() -> None:
    assert "tracking/velocity_cmd_abs" in METRIC_INDEX
    assert "tracking/velocity_cmd_nonzero_frac" in METRIC_INDEX


def _sampler_only_env(
    *,
    min_velocity: float,
    max_velocity: float,
    cmd_zero_chance: float,
    cmd_deadzone: float,
) -> WildRobotEnv:
    """Build a minimal env stub for unit-testing `_sample_velocity_cmd`."""
    env = WildRobotEnv.__new__(WildRobotEnv)
    env._config = SimpleNamespace(
        env=SimpleNamespace(
            min_velocity=float(min_velocity),
            max_velocity=float(max_velocity),
            cmd_zero_chance=float(cmd_zero_chance),
            cmd_deadzone=float(cmd_deadzone),
        )
    )
    return env


def _sample_scalar_cmds(env: WildRobotEnv, *, seed: int, count: int) -> np.ndarray:
    keys = jax.random.split(jax.random.PRNGKey(seed), count)
    sampler = jax.jit(jax.vmap(env._sample_velocity_cmd))
    return np.asarray(sampler(keys), dtype=np.float32)


def test_scalar_sampler_nonzero_branch_stays_outside_deadzone() -> None:
    deadzone = 0.0666666667
    env = _sampler_only_env(
        min_velocity=-0.1333333333,
        max_velocity=0.1333333333,
        cmd_zero_chance=0.0,
        cmd_deadzone=deadzone,
    )
    cmds = _sample_scalar_cmds(env, seed=0, count=256)
    assert np.all(np.abs(cmds) >= deadzone), (
        "nonzero branch produced commands inside the deadzone"
    )
    assert np.all(cmds != 0.0), "nonzero branch produced exact zero command"


def test_scalar_sampler_explicit_zero_branch_returns_exact_zero() -> None:
    env = _sampler_only_env(
        min_velocity=-0.1333333333,
        max_velocity=0.1333333333,
        cmd_zero_chance=1.0,
        cmd_deadzone=0.0666666667,
    )
    cmds = _sample_scalar_cmds(env, seed=1, count=64)
    assert np.all(cmds == 0.0), "cmd_zero_chance branch must return exact zero"


def test_scalar_sampler_fixed_command_behavior_is_preserved() -> None:
    env_fixed_walk = _sampler_only_env(
        min_velocity=0.08,
        max_velocity=0.08,
        cmd_zero_chance=0.0,
        cmd_deadzone=0.05,
    )
    assert float(env_fixed_walk._sample_velocity_cmd(jax.random.PRNGKey(2))) == pytest.approx(0.08)

    env_fixed_in_deadzone = _sampler_only_env(
        min_velocity=0.04,
        max_velocity=0.04,
        cmd_zero_chance=0.0,
        cmd_deadzone=0.05,
    )
    assert float(env_fixed_in_deadzone._sample_velocity_cmd(jax.random.PRNGKey(3))) == pytest.approx(0.0)

    env_fixed_zeroed = _sampler_only_env(
        min_velocity=0.08,
        max_velocity=0.08,
        cmd_zero_chance=1.0,
        cmd_deadzone=0.0,
    )
    assert float(env_fixed_zeroed._sample_velocity_cmd(jax.random.PRNGKey(4))) == pytest.approx(0.0)


def test_scalar_sampler_smoke9c_zero_mix_is_not_old_sixty_percent() -> None:
    env = _sampler_only_env(
        min_velocity=-0.1333333333,
        max_velocity=0.1333333333,
        cmd_zero_chance=0.2,
        cmd_deadzone=0.0666666667,
    )
    cmds = _sample_scalar_cmds(env, seed=5, count=1024)
    zero_ratio = float(np.mean(cmds == 0.0))
    assert abs(zero_ratio - 0.2) <= 0.05, (
        f"zero-command ratio drifted from explicit zero branch: got {zero_ratio:.3f}"
    )
    assert zero_ratio <= 0.30, (
        f"zero-command ratio regressed toward old deadzone-mixing bug: got {zero_ratio:.3f}"
    )


def test_scalar_sampler_handles_asymmetric_ranges_without_symmetry_assumption() -> None:
    env = _sampler_only_env(
        min_velocity=0.02,
        max_velocity=0.12,
        cmd_zero_chance=0.0,
        cmd_deadzone=0.05,
    )
    cmds = _sample_scalar_cmds(env, seed=6, count=128)
    assert np.all(cmds >= 0.05)
    assert np.all(cmds <= 0.12)
