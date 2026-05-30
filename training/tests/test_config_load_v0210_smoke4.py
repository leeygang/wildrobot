"""Pin the v0.21.0 smoke4A (forward-first) YAML contract.

``ppo_walking_v0210_smoke4_forward_first.yaml`` is the forward-first
control after smoke3 (run psbmkbds) failed.  smoke3 used
``cmd_velocity_track_dim=2`` and drifted laterally (visual: lat_vel ->
0.55 m/s at cmd (0.26,0,0)); the vy term saturated the 2D reward to ~0,
killing the forward gradient.  smoke4A asks the decisive question: can
the v0.21 plumbing recover a forward gait with EVERY lateral/yaw DOF
removed?

smoke4A changes vs smoke3:
  * cmd_sampler_3d_branched: true -> False  (legacy scalar (vx,0,0))
  * cmd_turn_chance: 0.05 -> 0.0
  * min/max_velocity_y: -/+0.13 -> 0.0      (no lateral command)
  * max_yaw_rate: 0.25 -> 0.0
  * reward_weights.cmd_velocity_track_dim: 2 -> 1   (the main fix)
  * reward_weights.cmd_yaw_rate_track: 1.5 -> 0.0
  * eval_velocity_cmd: [0.20, 0.065, 0] -> [0.13, 0.0, 0.0]

Kept from smoke3: floor 0.065 / max 0.26 / interval 0.065 (0.26 ref bin),
alpha 562.5, action_rate -1.0, feet_phase, home anchor, obs_v8,
loc_ref_command_axes_3d (3D library still built), critic actor+privileged.

Reference: training/CHANGELOG.md v0.21.0-smoke4A-plan + the smoke3 result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


_SMOKE4 = Path("training/configs/ppo_walking_v0210_smoke4_forward_first.yaml")


@pytest.fixture(scope="module")
def cfg():
    if not _SMOKE4.exists():
        pytest.skip(f"{_SMOKE4.name} not found")
    return load_training_config(str(_SMOKE4))


# ---------------------------------------------------------------------------
# Loads under the strict validator
# ---------------------------------------------------------------------------
def test_smoke4_loads(cfg) -> None:
    assert cfg.version == "0.21.0"


# ---------------------------------------------------------------------------
# v0.21 plumbing kept (this is NOT a v0.20 rollback)
# ---------------------------------------------------------------------------
def test_smoke4_keeps_v021_plumbing(cfg) -> None:
    assert cfg.env.actor_obs_layout_id == "wr_obs_v8_cmd3d"
    assert cfg.env.loc_ref_command_axes_3d is True
    assert cfg.ppo.critic_includes_actor_obs is True


# ---------------------------------------------------------------------------
# Forward-first knobs (the lever set)
# ---------------------------------------------------------------------------
def test_smoke4_is_1d_legacy_sampler(cfg) -> None:
    """cmd_sampler_3d_branched=False -> legacy scalar (vx,0,0) sampler."""
    assert cfg.env.cmd_sampler_3d_branched is False


def test_smoke4_forward_only_reward(cfg) -> None:
    """The main fix: dim=1 forward-only reward, no yaw pressure."""
    assert cfg.reward_weights.cmd_velocity_track_dim == 1
    assert cfg.reward_weights.cmd_yaw_rate_track == pytest.approx(0.0)


def test_smoke4_no_lateral_or_yaw_command(cfg) -> None:
    assert cfg.env.cmd_turn_chance == pytest.approx(0.0)
    assert cfg.env.max_yaw_rate == pytest.approx(0.0)
    assert cfg.env.min_velocity_y == pytest.approx(0.0)
    assert cfg.env.max_velocity_y == pytest.approx(0.0)


def test_smoke4_eval_is_forward_first_exact_bin(cfg) -> None:
    """Eval at a reachable forward, exact-bin command; vy=wz=0 (smoke3's
    [0.20, 0.065, 0] would be OOD/dead for this 1D forward run)."""
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx((0.13, 0.0, 0.0))


# ---------------------------------------------------------------------------
# smoke3 floor / range / binning preserved
# ---------------------------------------------------------------------------
def test_smoke4_keeps_smoke3_floor_and_binning(cfg) -> None:
    assert cfg.env.min_velocity == pytest.approx(0.065)
    assert cfg.env.max_velocity == pytest.approx(0.26)
    assert cfg.env.loc_ref_command_grid_interval == pytest.approx(0.065)
    assert cfg.reward_weights.cmd_forward_velocity_alpha == pytest.approx(562.5)
    # prior-free reward block + smoke2 basin-break values still intact
    assert cfg.reward_weights.action_rate == pytest.approx(-1.0)
    assert cfg.reward_weights.feet_phase == pytest.approx(7.5)
    assert cfg.reward_weights.ref_q_track == pytest.approx(0.0)
    assert cfg.env.loc_ref_residual_base == "home"


# ---------------------------------------------------------------------------
# Real blast radius: env-init pins the actual library bins
# (loc_ref_command_axes_3d=true still builds the 3D library off the vx grid;
# the 0.26 top-speed bin must survive)
# ---------------------------------------------------------------------------
def test_smoke4_env_init_pins_reference_vx_bins(cfg) -> None:
    import numpy as np
    from training.envs.wildrobot_env import WildRobotEnv

    env = WildRobotEnv(config=cfg)
    keys = np.asarray(env._offline_cmd_keys)
    vx = sorted({round(float(v), 4) for v in keys[:, 0]})
    assert vx == [0.0, 0.065, 0.13, 0.195, 0.2, 0.26]
    assert len(keys) == 30

    def has(v):
        return any(np.allclose(k, v, atol=1e-6) for k in keys)

    assert has([0.065, 0.0, 0.0])
    assert has([0.13, 0.0, 0.0])   # eval command is an exact bin
    assert has([0.26, 0.0, 0.0])   # top-speed bin retained


# ---------------------------------------------------------------------------
# Sampler sanity: legacy path emits 5% stand + 95% (vx in [0.065,0.26], 0, 0)
# ---------------------------------------------------------------------------
def test_smoke4_legacy_sampler_is_forward_only(cfg) -> None:
    import numpy as np
    import jax
    from training.envs.wildrobot_env import WildRobotEnv

    env = WildRobotEnv(config=cfg)
    ks = jax.random.split(jax.random.PRNGKey(0), 20000)
    cmds = np.asarray(jax.vmap(env._sample_velocity_cmd)(ks))
    vx, vy, wz = cmds[:, 0], cmds[:, 1], cmds[:, 2]
    zero = (np.abs(vx) < 1e-9) & (np.abs(vy) < 1e-9) & (np.abs(wz) < 1e-9)
    walk = ~zero
    # ~5% exact-stand commands
    assert 0.02 < zero.mean() < 0.09
    # walk branch: forward-only vx in [0.065, 0.26], no negatives, no turn
    assert vx[walk].min() >= 0.065 - 1e-6
    assert vx[walk].max() <= 0.26 + 1e-6
    assert not (vx[walk] < 0).any()
    # vy and wz are structurally zero on the legacy path
    assert np.allclose(vy, 0.0)
    assert np.allclose(wz, 0.0)
