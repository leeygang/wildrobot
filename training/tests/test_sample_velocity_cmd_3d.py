"""v0.21.0 P4.3-P4.7 — branched 3D command sampler tests.

Mirrors ToddlerBot's branched (zero / pure-turn / ellipse-walk)
``_sample_command`` (toddlerbot/locomotion/walk_env.py:235-321).

Divergences from the spec body in
``training/docs/.tmp_p4_original_spec.md``:

* The spec's ``_env_3d`` helper loads
  ``ppo_walking_v0210_smoke1_lateral_yaw.yaml`` (created later in P8).
  This file does not exist yet, so we use ``smoke14.yaml`` as the
  baseline and ``dataclasses.replace`` to inject the vy / wz ranges
  and per-branch chances that each test needs.

* P4.5 (walk-branch ellipse) — the spec asserts
  ``(cmds[:, 0] >= 0.0).all()``.  That assertion is wrong for any
  config with ``min_velocity > 0``: TB's walk sampler uses
  ``sin(theta) * uniform(deadzone, |x_max|)`` where the ``x_max``
  switches sign with ``sin(theta)``, so vx sweeps ``[-|min_velocity|,
  -deadzone] ∪ [deadzone, |max_velocity|]`` even when
  ``min_velocity > 0``.  See TB ``walk_env.py:267-272``.  We instead
  assert vx stays within ``[-max(|min_velocity|, |max_velocity|),
  +max(|min_velocity|, |max_velocity|)]`` (the elliptical envelope).

* P4.6 (turn-branch) — config object dataclass is frozen on the env
  copy, so we mutate via ``object.__setattr__`` on the existing
  EnvConfig rather than rebuilding the env (which would re-tracedly
  rebuild the offline reference library).  We also override
  ``cmd_zero_chance=0.0`` and ``cmd_turn_chance=1.0`` to isolate the
  turn branch.

* P4.7 (branch marginals) — smoke14 sets ``cmd_zero_chance=0.0`` and
  ``cmd_turn_chance=0.0``, so we explicitly override the env config to
  ``cmd_zero_chance=0.2, cmd_turn_chance=0.2`` to match the spec's
  intended 0.2/0.2/0.6 marginal distribution.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jp
import pytest

from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv


_BASE_YAML = "training/configs/ppo_walking_v0201_smoke14.yaml"


def _env_3d(
    *,
    min_velocity_y: float = -0.13,
    max_velocity_y: float = 0.13,
    max_yaw_rate: float = 0.25,
    cmd_zero_chance: float | None = None,
    cmd_turn_chance: float | None = None,
) -> WildRobotEnv:
    """Build a WildRobotEnv with vy / wz ranges injected on top of smoke14.

    Optional overrides for ``cmd_zero_chance`` / ``cmd_turn_chance``
    isolate individual branches in their dedicated tests.  When left as
    ``None`` the smoke14 defaults (both 0.0) are kept.

    P4-fix: ``cmd_sampler_3d_branched=True`` opts in to the TB-mirrored
    branched sampler.  Legacy YAML defaults to ``False`` (forward-only
    scalar-vx sampler) so smoke14 / smoke12b / smoke7 are unchanged.
    These tests EXERCISE the branched sampler, so the helper sets the
    opt-in flag.
    """
    cfg = load_training_config(_BASE_YAML)
    env_kwargs: dict[str, object] = dict(
        cmd_sampler_3d_branched=True,
        min_velocity_y=min_velocity_y,
        max_velocity_y=max_velocity_y,
        max_yaw_rate=max_yaw_rate,
    )
    if cmd_zero_chance is not None:
        env_kwargs["cmd_zero_chance"] = cmd_zero_chance
    if cmd_turn_chance is not None:
        env_kwargs["cmd_turn_chance"] = cmd_turn_chance
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(cfg.env, **env_kwargs),
    )
    return WildRobotEnv(cfg)


def test_sample_velocity_cmd_returns_3_vector() -> None:
    env = _env_3d()
    key = jax.random.PRNGKey(0)
    cmd = env._sample_velocity_cmd(key)
    assert cmd.shape == (3,)
    assert cmd.dtype == jp.float32


def test_walk_branch_samples_within_ellipse() -> None:
    """Walk branch alone — disable zero / turn paths.

    See the module docstring for the spec-divergence rationale on the
    vx sign assertion: TB's walk sampler allows negative vx through the
    ``sin(theta) > 0 ? max : -min`` switch, so we cannot require
    ``vx >= 0``.
    """
    env = _env_3d(cmd_zero_chance=0.0, cmd_turn_chance=0.0)
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)

    vx_envelope = max(
        abs(float(env._config.env.min_velocity)),
        abs(float(env._config.env.max_velocity)),
    )
    vy_envelope = max(
        abs(float(env._config.env.min_velocity_y)),
        abs(float(env._config.env.max_velocity_y)),
    )

    assert (cmds[:, 0] >= -vx_envelope - 1e-6).all()
    assert (cmds[:, 0] <= vx_envelope + 1e-6).all()
    assert (jp.abs(cmds[:, 1]) <= vy_envelope + 1e-6).all()
    # wz axis always zero in walk branch.
    assert (jp.abs(cmds[:, 2]) < 1e-6).all()


def test_turn_branch_samples_yaw_within_range() -> None:
    """Pure-turn branch — vx == vy == 0, |wz| <= max_yaw_rate."""
    env = _env_3d(cmd_zero_chance=0.0, cmd_turn_chance=1.0)
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)

    assert (jp.abs(cmds[:, 0]) < 1e-6).all()
    assert (jp.abs(cmds[:, 1]) < 1e-6).all()
    assert (jp.abs(cmds[:, 2]) <= env._config.env.max_yaw_rate + 1e-6).all()
    # Sanity: non-trivial yaw distribution (not all zero).
    assert float(jp.abs(cmds[:, 2]).mean()) > 0.0


def test_branch_marginal_distribution() -> None:
    """0.2 / 0.2 / 0.6 marginal split across zero / turn / walk branches."""
    env = _env_3d(cmd_zero_chance=0.2, cmd_turn_chance=0.2)
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)

    is_zero = jp.all(jp.abs(cmds) < 1e-6, axis=-1)
    is_turn = (
        (jp.abs(cmds[:, 2]) > 1e-6)
        & (jp.abs(cmds[:, 0]) < 1e-6)
        & (jp.abs(cmds[:, 1]) < 1e-6)
    )
    is_walk = (jp.abs(cmds[:, 0]) > 1e-6) | (jp.abs(cmds[:, 1]) > 1e-6)

    assert abs(float(is_zero.mean()) - 0.20) < 0.03
    assert abs(float(is_turn.mean()) - 0.20) < 0.03
    assert abs(float(is_walk.mean()) - 0.60) < 0.03


# -----------------------------------------------------------------------------
# P4-fix: gating contract — opt-in flag picks between the v0.20.x scalar
# sampler (default) and the TB-branched 3D sampler.
# -----------------------------------------------------------------------------


def test_default_sampler_is_v0_20x_scalar_for_smoke14() -> None:
    """smoke14 baseline must produce vx in [min_velocity, max_velocity]
    only (forward-only) when cmd_sampler_3d_branched is unset/False.
    Reverts the P4 behavioral change for legacy configs."""
    cfg = load_training_config(_BASE_YAML)
    assert cfg.env.cmd_sampler_3d_branched is False
    env = WildRobotEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    # vy and wz must be exactly zero on the scalar-vx path.
    assert (jp.abs(cmds[:, 1]) < 1e-7).all()
    assert (jp.abs(cmds[:, 2]) < 1e-7).all()
    # vx must NOT go negative under positive-only config.
    nonzero_vx = cmds[jp.abs(cmds[:, 0]) > 1e-7, 0]
    assert (nonzero_vx >= 0.0).all(), (
        f"v0.20.x scalar sampler produced negative vx: {nonzero_vx.min()}"
    )


def test_branched_sampler_opt_in_via_flag() -> None:
    """Setting cmd_sampler_3d_branched=True activates the TB-mirrored
    branched sampler.  With smoke14 + opt-in + the new vy/wz ranges,
    the sampler can emit nonzero vy / wz."""
    cfg = load_training_config(_BASE_YAML)
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env,
            cmd_sampler_3d_branched=True,
            min_velocity_y=-0.13,
            max_velocity_y=0.13,
            max_yaw_rate=0.25,
            cmd_zero_chance=0.0,
            cmd_turn_chance=1.0,  # pure-turn branch
        ),
    )
    env = WildRobotEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    # Pure-turn branch: vx == vy == 0, |wz| up to 0.25.
    assert (jp.abs(cmds[:, 0]) < 1e-6).all()
    assert (jp.abs(cmds[:, 1]) < 1e-6).all()
    assert (jp.abs(cmds[:, 2]) > 1e-6).any()
