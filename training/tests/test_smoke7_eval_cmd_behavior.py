"""smoke7 eval cmd plumbing — behavioral contract tests.

Goes beyond the YAML round-trip pinning in
``test_config_load_smoke6.py``: this file exercises the runtime env
behavior that the round-trip test cannot reach.

Two contracts under test:

1. **Eval cmd survives auto-reset** (issue 1 in the smoke7-prep1
   review).  ``WildRobotEnv.step`` auto-resets internally on episode
   termination via the ``_do_reset`` closure.  Without the smoke7
   fix, that closure called ``self.reset(rng_step)`` regardless of
   whether the step was an eval step — so any episode terminating
   before the eval horizon (~5% of episodes under smoke6-prep3
   distributions, likely higher under smoke7's added DR) would
   silently corrupt the post-reset window with a sampled cmd,
   diluting the load-bearing ``Evaluate/*`` metrics.

2. **Sentinel semantics** (issue 2 in the smoke7-prep1 review).
   ``eval_velocity_cmd < 0`` is documented as "eval samples like
   training."  ``reset_for_eval`` must be a no-op in this mode
   (delegate to ``reset``), and the post-reset cmd must vary across
   seeds (proving it's sampled, not pinned).

Spec: training/docs/walking_training.md v0.20.1-smoke7 §.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import pytest

_SMOKE_CFG = Path("training/configs/ppo_walking_v0201_smoke.yaml")
if not _SMOKE_CFG.exists():
    pytest.skip(
        f"{_SMOKE_CFG.name} not found",
        allow_module_level=True,
    )

from training.configs.training_config import load_training_config
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv

EPS = 1e-5


@pytest.fixture(scope="module")
def env_pinned():
    """Env constructed from the smoke7 YAML (eval_velocity_cmd = 0.15)."""
    cfg = load_training_config(str(_SMOKE_CFG))
    assert cfg.env.eval_velocity_cmd >= 0.0, (
        f"This test fixture assumes the smoke7 YAML pins eval_velocity_cmd; "
        f"got {cfg.env.eval_velocity_cmd}.  If smoke7 has been re-scoped, "
        f"update this fixture or the test."
    )
    return WildRobotEnv(cfg), float(cfg.env.eval_velocity_cmd)


@pytest.fixture(scope="module")
def env_sentinel():
    """Env with eval_velocity_cmd forced to -1.0 (sentinel: 'eval = training')."""
    cfg = load_training_config(str(_SMOKE_CFG))
    cfg.env.eval_velocity_cmd = -1.0
    return WildRobotEnv(cfg)


def test_eval_cmd_pinned_at_initial_reset(env_pinned) -> None:
    """Sanity baseline: ``reset_for_eval`` pins cmd to the configured value."""
    env, expected_cmd = env_pinned
    for seed in range(4):
        state = env.reset_for_eval(jax.random.PRNGKey(seed))
        cmd = float(state.info[WR_INFO_KEY].velocity_cmd)
        assert abs(cmd - expected_cmd) < EPS, (
            f"seed={seed}: expected eval cmd {expected_cmd}, got {cmd}"
        )


def test_eval_cmd_survives_auto_reset(env_pinned) -> None:
    """Forced termination must not let the auto-reset path swap the cmd.

    Setup: build an eval state (cmd=0.15), clobber the root z to a value
    well below ``min_height`` (forces height-based termination on the
    next step), then step in eval mode and verify the post-reset cmd
    is still pinned.

    Without the smoke7 fix, ``_do_reset`` calls ``self.reset(rng_step)``
    which samples cmd uniformly across [min_velocity, max_velocity] —
    and the test would fail with a randomly-sampled cmd in the new
    episode.

    With the fix, ``_do_reset`` calls ``self.reset_for_eval(rng_step)``
    when ``disable_cmd_resample=True``, which preserves the pin.
    """
    env, expected_cmd = env_pinned
    state = env.reset_for_eval(jax.random.PRNGKey(0))
    initial_cmd = float(state.info[WR_INFO_KEY].velocity_cmd)
    assert abs(initial_cmd - expected_cmd) < EPS

    # Force termination by clobbering root z below min_height (typically 0.30).
    # The MJX physics step will compute height_too_low → terminated → done.
    min_height = float(env._config.env.min_height)
    new_qpos = state.data.qpos.at[2].set(min_height - 0.10)
    new_data = state.data.replace(qpos=new_qpos)
    state = state.replace(data=new_data, pipeline_state=new_data)

    # Step in eval mode → auto-reset fires → new episode begins.
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    next_state = env.step(state, zero_action, disable_cmd_resample=True)

    # Confirm done fired (this is what triggers the auto-reset path we're
    # testing; if it didn't, the test setup is wrong).
    assert float(next_state.done) > 0.5, (
        "Test setup failed: clobbering root z below min_height did not "
        "trigger termination.  The auto-reset path was not exercised."
    )

    # The new cmd should still be pinned to the eval override, not a
    # freshly-sampled value.
    post_reset_cmd = float(next_state.info[WR_INFO_KEY].velocity_cmd)
    assert abs(post_reset_cmd - expected_cmd) < EPS, (
        f"Auto-reset corrupted the eval cmd pin: expected {expected_cmd}, "
        f"got {post_reset_cmd}.  The _do_reset closure in WildRobotEnv.step "
        f"is using reset() instead of reset_for_eval() when "
        f"disable_cmd_resample=True."
    )


def test_eval_cmd_sentinel_falls_back_to_sampled(env_sentinel) -> None:
    """``eval_velocity_cmd < 0`` must mean ``reset_for_eval`` ≡ ``reset``.

    Required by the documented sentinel semantics: when the override
    is unset (sentinel value), eval rollouts must behave like training
    rollouts (sample cmd from the configured range).
    """
    env = env_sentinel

    # If reset_for_eval correctly delegates to reset under sentinel mode,
    # the cmd should vary across seeds (like training reset does).
    cmds = []
    for seed in range(8):
        state = env.reset_for_eval(jax.random.PRNGKey(seed))
        cmds.append(float(state.info[WR_INFO_KEY].velocity_cmd))

    # Expect non-trivial variation across seeds.  With smoke7 YAML's
    # min_velocity=0.0, max_velocity=0.20, zero_chance=0.2, deadzone=0.05,
    # typical 8-seed sample shows 4-7 distinct values.  A pinned value
    # (the bug we're guarding against) would yield exactly one distinct
    # value — fail with a clear message.
    distinct = len(set(cmds))
    assert distinct >= 3, (
        f"Sentinel semantics broken: with eval_velocity_cmd=-1.0, "
        f"reset_for_eval should sample cmd like training, but 8 seeds "
        f"produced only {distinct} distinct cmd value(s): {cmds}.  "
        f"reset_for_eval is incorrectly pinning the cmd in sentinel mode."
    )

    # Sanity: all sampled cmds must be in the configured range.
    for seed_idx, cmd in enumerate(cmds):
        assert (
            env._config.env.min_velocity <= cmd <= env._config.env.max_velocity
        ), (
            f"Sampled eval cmd {cmd} from seed {seed_idx} is outside the "
            f"configured range "
            f"[{env._config.env.min_velocity}, {env._config.env.max_velocity}]."
        )
