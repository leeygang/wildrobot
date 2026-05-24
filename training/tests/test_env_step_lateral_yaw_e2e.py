"""v0.21.0 P9 — end-to-end env-step sanity for lateral and yaw cmds.

Note on what these tests prove: with zero action the policy doesn't move
the robot.  The actor's ``qpos`` is physics-driven (gravity + contact),
NOT cmd-driven — velocity commands do NOT directly inject motion into
the free joint.  The velocity cmd integrates into ``WildRobotInfo``'s
incremental path-state target (``path_state_torso_pos`` /
``path_state_path_rot``) per H7.  These tests assert that the env-step
pipeline updates that path-state target consistent with the cmd
direction under both lateral and yaw inputs.

Why we only step the env ONCE (not 50 times like the -final.md draft):
``training/tests/test_velocity_cmd_3d_migration.py``'s
``test_path_state_continuity_under_cmd_resample`` previously built a
full ``WildRobotEnv`` and stepped 50 times under CPU JAX, which
JIT-recompiled the entire env step on every iteration and hung the
test suite (see that file's review-fix C docstring).  The H7 invariant
under test is purely the ``incremental_path_state_step`` recurrence;
multi-step integration is already locked in by
``test_incremental_path_state_matches_closed_form_for_constant_cmd``.
P9's job is the env-step wiring — one step is sufficient to prove the
env reads ``velocity_cmd`` and writes a non-zero increment to the
path-state target in the correct axis.

The 50-step rollout shape from the draft is unsafe under non-jitted
env.step at the WR scale (smoke1 builds a 25-bin 3D library and each
re-trace takes minutes); collapsing to a single env step keeps the
test fast while still exercising the env-step → WildRobotInfo
pipeline that P9 needs to guard.

Full cmd-tracking coverage:
  * ``training/tests/test_velocity_cmd_3d_migration.py`` (3-vec shape
    + reset/step plumbing + N-step closed-form parity via pure helper)
  * ``training/tests/test_cmd_lateral_velocity_tracking.py`` (P6 reward
    consumes real vy_cmd from the 3-vec).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jp

from training.configs.training_config import load_training_config
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE1 = "training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml"
_SMOKE14 = "training/configs/ppo_walking_v0201_smoke14.yaml"


def _yaw_from_wxyz(quat) -> float:
    """Decode yaw (rad) from a wxyz quaternion via the standard
    ZYX-Euler closed form.  Casting components to ``float`` first
    keeps the return as a python float even if the input is a JAX
    tracer."""
    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _env(yaml_path: str) -> WildRobotEnv:
    cfg = load_training_config(yaml_path)
    return WildRobotEnv(config=cfg)


# ---------------------------------------------------------------------------
# P9.1 — env-step pipeline writes the cmd-integrated path-state target
# ---------------------------------------------------------------------------
def test_env_step_with_lateral_cmd_advances_path_state_y() -> None:
    """One env step under ``vy=0.10 m/s`` must write a positive y
    increment to ``wr.path_state_torso_pos`` (cmd → integrator wired).

    Expected per-step y increment: ``vy * dt = 0.10 * 0.02 = 0.002 m``.
    We assert > 0.0005 m to allow numerical slack while still catching
    a sign flip / zeroed-out integrator.

    Note: ``qpos[1]`` is NOT expected to track the cmd (zero action →
    no joint motion → no body-frame propulsion).  Multi-step
    integration parity vs the closed form is covered by
    ``test_incremental_path_state_matches_closed_form_for_constant_cmd``
    in ``test_velocity_cmd_3d_migration.py``.
    """
    env = _env(_SMOKE1)
    state = env.reset(jax.random.PRNGKey(0))
    wr0 = state.info[WR_INFO_KEY]
    # Sanity: reset puts the path-state at the world origin.
    assert float(wr0.path_state_torso_pos[1]) == 0.0
    new_wr = wr0.replace(
        velocity_cmd=jp.array([0.20, 0.10, 0.0], dtype=jp.float32)
    )
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})

    state1 = env.step(
        state,
        jp.zeros(env.action_size),
        disable_pushes=True,
        disable_cmd_resample=True,
    )

    wr1 = state1.info[WR_INFO_KEY]
    # Per-step y increment = vy_cmd * dt = 0.10 * 0.02 = 0.002 m.
    # >0.0005 catches a zeroed-out integrator without being noise-sensitive.
    assert float(wr1.path_state_torso_pos[1]) > 0.0005, (
        "path-state target y did not advance under lateral cmd: "
        f"path_state_torso_pos[1]={float(wr1.path_state_torso_pos[1])}"
    )
    # And x should also advance from vx_cmd=0.20 (sanity that we
    # didn't accidentally swap axes anywhere).
    assert float(wr1.path_state_torso_pos[0]) > 0.0005, (
        "path-state target x did not advance under vx cmd: "
        f"path_state_torso_pos[0]={float(wr1.path_state_torso_pos[0])}"
    )
    # Physics sanity: robot did not fall through the floor on one step.
    assert float(state1.pipeline_state.qpos[2]) > 0.2, (
        "physics blew up after one zero-action env step: "
        f"qpos[2]={float(state1.pipeline_state.qpos[2])}"
    )


def test_env_step_with_yaw_cmd_advances_path_state_yaw() -> None:
    """One env step under ``wz=0.20 rad/s`` must rotate
    ``wr.path_state_path_rot`` around +z (cmd → integrator wired).

    Expected per-step yaw increment: ``wz * dt = 0.20 * 0.02 = 0.004 rad``.
    We assert > 0.001 rad to allow slack while still catching a sign
    flip / zeroed-out integrator.

    Note: the actor's free-joint quat is NOT expected to track the cmd
    (zero action → no joint motion).  Multi-step yaw integration is
    covered by the pure-helper test
    ``test_incremental_step_rotates_around_z_under_pure_yaw_cmd``.
    """
    env = _env(_SMOKE1)
    state = env.reset(jax.random.PRNGKey(1))
    wr0 = state.info[WR_INFO_KEY]
    yaw_init = _yaw_from_wxyz(wr0.path_state_path_rot)
    new_wr = wr0.replace(
        velocity_cmd=jp.array([0.0, 0.0, 0.20], dtype=jp.float32)
    )
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})

    state1 = env.step(
        state,
        jp.zeros(env.action_size),
        disable_pushes=True,
        disable_cmd_resample=True,
    )

    wr1 = state1.info[WR_INFO_KEY]
    yaw_final = _yaw_from_wxyz(wr1.path_state_path_rot)
    delta = yaw_final - yaw_init
    # Per-step yaw delta = wz_cmd * dt = 0.20 * 0.02 = 0.004 rad,
    # positive under +wz.  >0.001 with the sign check catches a sign
    # flip or a zeroed-out integrator.
    assert delta > 0.001, (
        "path-state yaw did not advance under +wz cmd: "
        f"yaw_init={yaw_init}, yaw_final={yaw_final}, delta={delta}"
    )
    # Physics sanity.
    assert float(state1.pipeline_state.qpos[2]) > 0.2, (
        "physics blew up after one zero-action env step: "
        f"qpos[2]={float(state1.pipeline_state.qpos[2])}"
    )


# ---------------------------------------------------------------------------
# P9.2 — v0.20.1 smoke14 regression guard under v0.21 code
# ---------------------------------------------------------------------------
def test_smoke14_env_still_initializes_under_v7() -> None:
    """The v0.20.1 baseline smoke must still initialize after all v0.21
    code lands: obs layout id stays ``wr_obs_v7_phase_proprio`` and
    ``WildRobotInfo.velocity_cmd`` has shape (3,) (the migration from
    the legacy scalar happens unconditionally; smoke14's eval cmd
    broadcasts (s,) → (s, 0, 0) so only ``vx`` is non-zero)."""
    env = _env(_SMOKE14)
    state = env.reset(jax.random.PRNGKey(0))
    assert env._policy_spec.observation.layout_id == "wr_obs_v7_phase_proprio"
    wr = state.info[WR_INFO_KEY]
    assert wr.velocity_cmd.shape == (3,)
