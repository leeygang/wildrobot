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

import jax
import jax.numpy as jp
import numpy as np
import pytest

_SMOKE_CFG = Path("training/configs/ppo_walking_v0201_smoke.yaml")
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
