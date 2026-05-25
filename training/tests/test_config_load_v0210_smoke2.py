"""Pin the v0.21.0 lateral+yaw smoke2 YAML contract.

``ppo_walking_v0210_smoke2.yaml`` is the basin-break retry of smoke1
after the lblub15y (30M) post-mortem.  Inherits all 4 v0.21.0 gates
from smoke1, then applies 6 targeted changes:

  (A) Reward kernel: cmd_forward_velocity_alpha 40 -> 562.5
      Closes Problem A (standing-during-walk-cmd partial credit leak).
  (A) Action smoothness: action_rate -2.0 -> -1.0
      Restores smoke12b basin-break value (stride exploration enabler).
  (B) Sampler: cmd_zero_chance 0.20 -> 0.05, cmd_turn_chance 0.20 -> 0.05
      Shrinks Problem B (zero-cmd basin-formation exposure).
  (C) Sampler: cmd_sampler_walk_vx_positive_only = True (new flag)
      Closes negative-vx contract mismatch with positive-only library.
  (D) Horizon: max_episode_steps 500 -> 1000
      Longer productive walking horizon post cold-start.
  (E) Eval: eval_velocity_cmd [0.18, 0.04, 0.0] -> [0.20, 0.065, 0.0]
      Aligns to library bins (vy=0.04 was off-grid); tests fwd+lateral.

These tests pin the YAML contract so a later edit can't silently revert
one of the 6 basin-break changes.

Reference: /tmp/wildrobot_v021_smoke2_plan.md (final).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


_SMOKE2 = Path("training/configs/ppo_walking_v0210_smoke2.yaml")
_SMOKE1 = Path("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")


@pytest.fixture(scope="module")
def cfg_smoke2():
    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    return load_training_config(str(_SMOKE2))


# ---------------------------------------------------------------------------
# Inheritance: smoke2 keeps the 4 v0.21.0 gates from smoke1
# ---------------------------------------------------------------------------
def test_smoke2_inherits_v021_gates(cfg_smoke2) -> None:
    env = cfg_smoke2.env
    rw = cfg_smoke2.reward_weights
    assert env.actor_obs_layout_id == "wr_obs_v8_cmd3d"
    assert env.cmd_sampler_3d_branched is True
    assert env.loc_ref_command_axes_3d is True
    assert rw.cmd_yaw_rate_track == pytest.approx(1.5)
    assert rw.cmd_velocity_track_dim == 2
    # Plan-locked ranges (v0.21.0 plan):
    assert env.min_velocity_y == pytest.approx(-0.13)
    assert env.max_velocity_y == pytest.approx(0.13)
    assert env.max_yaw_rate == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Smoke2 change (A): reward kernel + action smoothness
# ---------------------------------------------------------------------------
def test_smoke2_change_A_cmd_forward_velocity_alpha_reverted_to_smoke12b(
    cfg_smoke2,
) -> None:
    """smoke2 (A): closes Problem A leak by reverting alpha 40 -> 562.5.
    smoke12b/13/14 all used 562.5; smoke1's 40 paid +0.011/step standing
    credit during walk-cmd."""
    assert cfg_smoke2.reward_weights.cmd_forward_velocity_alpha == pytest.approx(562.5)


def test_smoke2_change_A_action_rate_reverted_to_smoke12b_basin_break(
    cfg_smoke2,
) -> None:
    """smoke2 (A): action_rate -2.0 -> -1.0 restores smoke12b's
    basin-break value.  smoke12b shipped at -1.0 and broke through to
    gait; smoke13/14/1 reverted to -2.0 (TB-faithful) and stalled."""
    assert cfg_smoke2.reward_weights.action_rate == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Smoke2 change (B): sampler bias toward forward
# ---------------------------------------------------------------------------
def test_smoke2_change_B_cmd_zero_and_turn_chance_reduced(cfg_smoke2) -> None:
    """smoke2 (B): drop zero/turn chance from 0.20 each to 0.05 each.
    Shrinks zero-cmd episode exposure from 40% -> 10%, reducing
    Problem B basin formation."""
    assert cfg_smoke2.env.cmd_zero_chance == pytest.approx(0.05)
    assert cfg_smoke2.env.cmd_turn_chance == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Smoke2 change (C): vx-clamp
# ---------------------------------------------------------------------------
def test_smoke2_change_C_walk_vx_positive_only_enabled(cfg_smoke2) -> None:
    """smoke2 (C): opt into the walk-branch vx-clamp.  Closes the
    contract mismatch between TB's symmetric ellipse (emits negative vx)
    and WR's positive-only reference library."""
    assert cfg_smoke2.env.cmd_sampler_walk_vx_positive_only is True


# ---------------------------------------------------------------------------
# Smoke2 change (D): episode horizon
# ---------------------------------------------------------------------------
def test_smoke2_change_D_episode_horizon_doubled(cfg_smoke2) -> None:
    """smoke2 (D): max_episode_steps 500 -> 1000.  Walking episode that
    survives cold-start now gets 900 productive steps vs 400."""
    assert cfg_smoke2.env.max_episode_steps == 1000


# ---------------------------------------------------------------------------
# Smoke2 change (E): eval cmd aligned to library bins
# ---------------------------------------------------------------------------
def test_smoke2_change_E_eval_velocity_cmd_on_grid(cfg_smoke2) -> None:
    """smoke2 (E): eval cmd [0.18, 0.04, 0.0] -> [0.20, 0.065, 0.0].
    Both axes now match library bin centers exactly:
      - vx=0.20 matches loc_ref_offline_command_vx union anchor
      - vy=0.065 is an exact library bin (grid {-0.13,-0.065,0,0.065,0.13})
    smoke1's vy=0.04 was off-grid -> snap-induced phantom err."""
    eval_cmd = tuple(cfg_smoke2.env.eval_velocity_cmd)
    assert eval_cmd == pytest.approx((0.20, 0.065, 0.00))

    # vx anchor matches eval vx
    assert cfg_smoke2.env.loc_ref_offline_command_vx == pytest.approx(0.20)

    # vy on grid
    vy_grid = tuple(cfg_smoke2.env.loc_ref_offline_command_vy_grid)
    assert eval_cmd[1] in vy_grid, (
        f"eval vy={eval_cmd[1]} not in library grid {vy_grid}"
    )


def test_smoke2_eval_is_walk_branch_in_distribution(cfg_smoke2) -> None:
    """smoke2 primary eval must not have simultaneous nonzero vy AND wz
    (TB-split contract).  Inherited invariant from smoke1's review."""
    vx, vy, wz = (float(c) for c in cfg_smoke2.env.eval_velocity_cmd)
    vy_active = abs(vy) >= 0.02
    wz_active = abs(wz) >= 0.05
    assert not (vy_active and wz_active), (
        f"primary eval_velocity_cmd=({vx:.3f}, {vy:.3f}, {wz:.3f}) violates "
        "the TB-split walk-branch contract (mixed walk + turn)."
    )


# ---------------------------------------------------------------------------
# Probes still present + plan-locked
# ---------------------------------------------------------------------------
def test_smoke2_probes_still_configured(cfg_smoke2) -> None:
    """Probes inherited from smoke1; smoke2 must keep both Appendix C
    pure-axis probes."""
    probes = tuple(cfg_smoke2.env.eval_velocity_cmd_probes)
    assert len(probes) >= 2
    # Pure-lateral
    lateral = [p for p in probes if abs(p[1]) >= 0.02 and abs(p[2]) < 0.05]
    assert lateral, "smoke2 must keep at least one pure-lateral probe"
    # Pure-yaw
    yaw = [p for p in probes if abs(p[2]) >= 0.05 and abs(p[1]) < 0.02]
    assert yaw, "smoke2 must keep at least one pure-yaw probe"


# ---------------------------------------------------------------------------
# Env init smoke test
# ---------------------------------------------------------------------------
def test_smoke2_env_init_succeeds(cfg_smoke2) -> None:
    """Sanity: env constructs end-to-end with all smoke2 changes."""
    from training.envs.wildrobot_env import WildRobotEnv
    env = WildRobotEnv(config=cfg_smoke2)
    assert env is not None
    assert env._config.env.max_episode_steps == 1000
    assert env._config.env.cmd_sampler_walk_vx_positive_only is True
