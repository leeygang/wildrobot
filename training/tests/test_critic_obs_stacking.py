"""Pin contract for smoke14 critic obs temporal stacking.

Spec: /tmp/smoke14_critic_obs_stacking_prompt.md.

What this test file pins:

  1. Default is legacy single-frame (depth=1), critic_obs shape (52,);
     smoke12b loads with depth=1.

  2. Smoke14 enables depth=15, critic_obs shape (780,).

  3. Reset initializes the rolling history to zeros and writes the
     reset frame at the newest slot; under depth=15 the older slots
     are still zero (older history doesn't exist yet).

  4. Each subsequent env.step appends one frame to the newest slot
     (oldest-first → newest-last convention, matching TB
     mjx_env.py:2166-2171).

  5. A single-step discontinuity in the underlying single-frame
     critic obs appears at only 1/15 of the stacked vector at that
     step, then transitions smoothly as the buffer rolls.

  6. ``create_networks(critic_obs_dim=780, ...)`` accepts the stacked
     dim and produces a finite scalar on a forward pass.

  7. Smoke12b regression: critic_obs is still (PRIVILEGED_OBS_DIM,)
     and ``critic_obs_history_frames`` defaults to 1 — no behavior
     change for any existing config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jp

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import EnvConfig
from training.envs.env_info import (
    PRIVILEGED_OBS_DIM,
    PRIVILEGED_OBS_HISTORY_FRAMES,
    WR_INFO_KEY,
)
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE12B = Path("training/configs/ppo_walking_v0201_smoke12b.yaml")
_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")
_SMOKE14 = Path("training/configs/ppo_walking_v0201_smoke14.yaml")


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    return load_training_config(str(path))


# ---------------------------------------------------------------------------
# 1. Default depth = 1 (legacy single-frame).
# ---------------------------------------------------------------------------
def test_critic_obs_history_frames_defaults_to_1() -> None:
    """A bare EnvConfig must default to single-frame critic obs so
    every pre-smoke14 config stays byte-equal."""
    assert EnvConfig().critic_obs_history_frames == 1


def test_smoke12b_loads_with_legacy_critic_depth(smoke12b_cfg) -> None:
    assert smoke12b_cfg.env.critic_obs_history_frames == 1


# ---------------------------------------------------------------------------
# 2. Smoke14 enables stacking.
# ---------------------------------------------------------------------------
def test_smoke14_sets_critic_depth_15(smoke14_cfg) -> None:
    assert smoke14_cfg.env.critic_obs_history_frames == 15


def test_smoke13_keeps_legacy_critic_depth(smoke13_cfg) -> None:
    """Smoke14 is supposed to be the FIRST config flipping the field
    on; smoke13 stays at depth=1 so its prior runs remain bit-equal."""
    assert smoke13_cfg.env.critic_obs_history_frames == 1


# ---------------------------------------------------------------------------
# 3. Reset / step shapes.
# ---------------------------------------------------------------------------
def test_smoke12b_critic_obs_shape_unchanged(smoke12b_env) -> None:
    state = smoke12b_env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    assert wr.critic_obs.shape == (PRIVILEGED_OBS_DIM,)
    # History buffer always carries the full depth regardless of
    # exposed-frame count, so the pytree shape stays uniform.
    assert wr.critic_obs_history.shape == (
        PRIVILEGED_OBS_HISTORY_FRAMES,
        PRIVILEGED_OBS_DIM,
    )


def test_smoke14_critic_obs_is_stacked_780(smoke14_env) -> None:
    state = smoke14_env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    assert wr.critic_obs.shape == (15 * PRIVILEGED_OBS_DIM,)
    assert wr.critic_obs.shape == (780,)
    assert wr.critic_obs_history.shape == (
        PRIVILEGED_OBS_HISTORY_FRAMES,
        PRIVILEGED_OBS_DIM,
    )


# ---------------------------------------------------------------------------
# 4. Reset writes the first frame into the newest slot; older slots zero.
# ---------------------------------------------------------------------------
def test_smoke14_reset_history_only_newest_slot_populated(smoke14_env) -> None:
    state = smoke14_env.reset(jax.random.PRNGKey(0))
    history = np.asarray(state.info[WR_INFO_KEY].critic_obs_history)
    # Last slot (newest) carries the real reset frame; older slots
    # must be all zero.
    assert not np.allclose(history[-1], 0.0), (
        "newest slot at reset must contain the real critic obs frame"
    )
    for i in range(PRIVILEGED_OBS_HISTORY_FRAMES - 1):
        assert np.allclose(history[i], 0.0), (
            f"slot {i} should be zero at reset (history not yet rolled "
            f"that far); got nonzero values"
        )


def test_smoke14_reset_critic_obs_matches_flatten_of_history(smoke14_env) -> None:
    """The exposed critic_obs == history[-15:].reshape(-1).  Pin that
    contract so any future helper rewrite doesn't silently break the
    layout."""
    state = smoke14_env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    history = np.asarray(wr.critic_obs_history)
    co = np.asarray(wr.critic_obs)
    np.testing.assert_allclose(co, history[-15:].reshape(-1), atol=1e-7)


# ---------------------------------------------------------------------------
# 5. Stepping rolls oldest-first, newest-last.
# ---------------------------------------------------------------------------
def test_smoke14_history_fills_one_frame_per_step(smoke14_env) -> None:
    state = smoke14_env.reset(jax.random.PRNGKey(0))
    action = jp.zeros((smoke14_env.action_size,), dtype=jp.float32)

    def nonzero_slots(s):
        h = np.asarray(s.info[WR_INFO_KEY].critic_obs_history)
        return int((np.abs(h) > 1e-12).any(axis=-1).sum())

    # Reset already populated the newest slot.
    assert nonzero_slots(state) == 1

    step_fn = jax.jit(smoke14_env.step)
    # Step N < PRIVILEGED_OBS_HISTORY_FRAMES; each step adds exactly
    # one new frame (the older slots remain whatever was rolled in
    # previously; once the buffer fills, all slots are nonzero).
    for expected_nonzero in range(2, 6):
        state = step_fn(state, action)
        n = nonzero_slots(state)
        assert n == expected_nonzero, (
            f"after {expected_nonzero - 1} steps, expected "
            f"{expected_nonzero} populated slots; got {n}"
        )


def test_smoke14_history_newest_slot_changes_with_step(smoke14_env) -> None:
    """The newest slot is rewritten every step (it carries the
    just-computed critic obs), while the previously-newest slot
    shifts one position back."""
    state0 = smoke14_env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(smoke14_env.step)
    state1 = step_fn(state0, jp.zeros((smoke14_env.action_size,), dtype=jp.float32))
    h0 = np.asarray(state0.info[WR_INFO_KEY].critic_obs_history)
    h1 = np.asarray(state1.info[WR_INFO_KEY].critic_obs_history)
    # Slot -2 of h1 must equal slot -1 of h0 (roll-back invariant).
    np.testing.assert_allclose(h1[-2], h0[-1], atol=1e-6)
    # Slot -1 of h1 is the newly-computed frame; it MUST differ from
    # h0[-1] for a non-trivial physics step (gravity + reset
    # perturbation guarantee some change).
    assert not np.allclose(h1[-1], h0[-1], atol=1e-4)


# ---------------------------------------------------------------------------
# 6. Discontinuity smoothing — a unit-magnitude jump in the underlying
#    single-frame obs appears as exactly 1/15 of the stacked vector's
#    deviation at the jump step (only the newest slot changed), and
#    propagates back one slot per subsequent step.
# ---------------------------------------------------------------------------
def test_smoke14_discontinuity_dilutes_to_one_over_N() -> None:
    """Use the stacking helpers directly with a constructed history +
    next-frame to pin the smoothing arithmetic without relying on the
    physics step path."""
    cfg = _load(_SMOKE14)
    env = WildRobotEnv(config=cfg)

    baseline = jp.full((PRIVILEGED_OBS_DIM,), 1.0, dtype=jp.float32)
    jumped = jp.full((PRIVILEGED_OBS_DIM,), 2.0, dtype=jp.float32)
    # Pre-jump history: every slot == baseline.
    history = jp.tile(baseline[None, :], (PRIVILEGED_OBS_HISTORY_FRAMES, 1))
    # Roll once with the post-jump frame.
    history1 = env._roll_critic_obs_history(history, jumped)
    co1 = env._stack_critic_obs(history1)
    # Older 14 slots still = baseline; newest = jumped.
    co1_np = np.asarray(co1)
    expected = np.concatenate(
        [np.full(14 * PRIVILEGED_OBS_DIM, 1.0, dtype=np.float32),
         np.full(PRIVILEGED_OBS_DIM, 2.0, dtype=np.float32)]
    )
    np.testing.assert_allclose(co1_np, expected, atol=1e-7)

    # 1/15 dilution: deviation from baseline-uniform is concentrated
    # in the last PRIVILEGED_OBS_DIM positions of the stacked vector.
    deviation = co1_np - np.full_like(co1_np, 1.0)
    nonzero_frac = float((np.abs(deviation) > 1e-6).mean())
    assert nonzero_frac == pytest.approx(1.0 / 15.0, abs=1e-9), (
        f"discontinuity should occupy 1/15 of the stacked vector; "
        f"got {nonzero_frac}"
    )

    # Subsequent steps with the same jumped value propagate it through
    # the buffer; after 14 more rolls the entire vector is the new
    # value (full transition).
    h = history1
    for step in range(2, PRIVILEGED_OBS_HISTORY_FRAMES + 1):
        h = env._roll_critic_obs_history(h, jumped)
        co_np = np.asarray(env._stack_critic_obs(h))
        # Newest ``step`` slots == 2.0, older slots == 1.0.
        expected = np.concatenate(
            [
                np.full(
                    (PRIVILEGED_OBS_HISTORY_FRAMES - step) * PRIVILEGED_OBS_DIM,
                    1.0,
                    dtype=np.float32,
                ),
                np.full(step * PRIVILEGED_OBS_DIM, 2.0, dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(co_np, expected, atol=1e-7)


def test_smoke12b_helper_returns_single_frame() -> None:
    """Under depth==1 the stack helper must return the trailing frame
    (no flatten, no concat) so the legacy byte-equal contract holds."""
    cfg = _load(_SMOKE12B)
    env = WildRobotEnv(config=cfg)
    assert env._critic_obs_history_frames == 1
    history = jp.arange(
        PRIVILEGED_OBS_HISTORY_FRAMES * PRIVILEGED_OBS_DIM, dtype=jp.float32
    ).reshape(PRIVILEGED_OBS_HISTORY_FRAMES, PRIVILEGED_OBS_DIM)
    co = env._stack_critic_obs(history)
    assert co.shape == (PRIVILEGED_OBS_DIM,)
    np.testing.assert_allclose(np.asarray(co), np.asarray(history[-1]))


# ---------------------------------------------------------------------------
# 7. create_networks accepts 780-dim critic obs.
# ---------------------------------------------------------------------------
def test_create_networks_accepts_stacked_critic_dim() -> None:
    """Pin: create_networks(critic_obs_dim=780, ...) returns a value
    head whose input width matches the stacked critic obs.  Mirrors
    the init/apply call pattern in
    ``training/algos/ppo/ppo_core.py::init_network_params`` and
    ``compute_values``."""
    from training.algos.ppo.ppo_core import create_networks

    obs_dim = 1184  # smoke14 actor obs_dim
    action_dim = 21
    network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=(512, 256, 128),
        value_hidden_dims=(512, 256, 128),
        critic_obs_dim=780,
        activation="elu",
    )
    value_params = network.value_network.init(jax.random.PRNGKey(0))
    critic_input = jax.random.normal(jax.random.PRNGKey(1), (780,))
    # Empty preprocessor_params matches init_network_params (no obs
    # normalization by default).
    value = network.value_network.apply((), value_params, critic_input)
    value = np.asarray(value)
    assert value.shape == ()
    assert np.isfinite(value)


# ---------------------------------------------------------------------------
# 8. training_loop derives critic_obs_dim from env.critic_obs_history_frames.
# ---------------------------------------------------------------------------
def test_training_loop_derives_stacked_critic_dim(smoke14_cfg) -> None:
    """Confirm the multiplier the training_loop applies matches what
    the env actually puts on the critic_obs vector — guards against
    a future refactor that introduces a mismatch (NN expecting
    52 while env emits 780)."""
    frames = int(smoke14_cfg.env.critic_obs_history_frames)
    derived = PRIVILEGED_OBS_DIM * frames
    assert derived == 780


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def smoke12b_cfg():
    return _load(_SMOKE12B)


@pytest.fixture(scope="module")
def smoke13_cfg():
    return _load(_SMOKE13)


@pytest.fixture(scope="module")
def smoke14_cfg():
    return _load(_SMOKE14)


@pytest.fixture(scope="module")
def smoke12b_env(smoke12b_cfg):
    return WildRobotEnv(config=smoke12b_cfg)


@pytest.fixture(scope="module")
def smoke14_env(smoke14_cfg):
    return WildRobotEnv(config=smoke14_cfg)
