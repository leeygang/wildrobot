"""Tests for v0.21.0 smoke2 Lever 7: PPOConfig.critic_includes_actor_obs.

Lever 7 (Option A): make the value-network input be
concat([actor_obs_history, privileged_obs_history]) instead of the
privileged_obs_history alone.  Standard asymmetric actor-critic pattern;
mirrors TB whose num_single_privileged_obs (151) ALREADY includes the
noiseless actor obs inline.

WR's pre-Lever-7 critic saw ONLY the 52-dim privileged_obs (with 15
history frames = 780 dims), missing cmd / phase / quat / abs motor_pos
that the actor sees.  At the standing local minimum that V(s) under-
estimation contributed to the V(stand) > V(walk) inversion smoke1
lblub15y converged to.

Tests:
  * Default False (back-compat: smoke1 / smoke14 / smoke12b unaffected).
  * Smoke2 YAML opts in.
  * compute_values input dim under each branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import PPOConfig


_SMOKE1 = Path("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
_SMOKE2 = Path("training/configs/ppo_walking_v0210_smoke2.yaml")


# ---------------------------------------------------------------------------
# Dataclass default + back-compat
# ---------------------------------------------------------------------------
def test_critic_includes_actor_obs_default_false() -> None:
    """Default must be False so legacy YAMLs (smoke1, smoke14, ...)
    are byte-for-byte identical."""
    ppo_cfg = PPOConfig()
    assert ppo_cfg.critic_includes_actor_obs is False


def test_smoke1_yaml_does_not_set_lever7() -> None:
    """smoke1 reproducibility: must keep default False so the pre-
    Lever-7 critic behavior is preserved."""
    if not _SMOKE1.exists():
        pytest.skip(f"{_SMOKE1.name} not found")
    cfg = load_training_config(str(_SMOKE1))
    assert cfg.ppo.critic_includes_actor_obs is False


def test_smoke2_yaml_opts_in_to_lever7() -> None:
    """smoke2 must opt in to the actor-obs concat into the critic."""
    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    cfg = load_training_config(str(_SMOKE2))
    assert cfg.ppo.critic_includes_actor_obs is True
    # Lever 7 is only meaningful with the privileged critic enabled.
    assert cfg.ppo.critic_privileged_enabled is True


# ---------------------------------------------------------------------------
# End-to-end: rollout concats correctly under each branch
# ---------------------------------------------------------------------------
def test_rollout_critic_obs_shape_when_lever7_off() -> None:
    """When critic_includes_actor_obs=False (default), rollout's
    per-step critic_obs has shape = privileged-only (PRIVILEGED_OBS_DIM
    * critic_history_frames)."""
    import dataclasses
    import jax
    import jax.numpy as jp

    from training.envs.env_info import PRIVILEGED_OBS_DIM, WR_INFO_KEY
    from training.envs.wildrobot_env import WildRobotEnv
    from training.algos.ppo.ppo_core import create_networks, init_network_params
    from training.core.rollout import collect_rollout

    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    cfg = load_training_config(str(_SMOKE2))
    # Force Lever 7 OFF for this test.
    cfg = dataclasses.replace(
        cfg,
        ppo=dataclasses.replace(cfg.ppo, critic_includes_actor_obs=False),
    )

    env = WildRobotEnv(config=cfg)
    obs_dim = env.observation_size
    action_dim = env.action_size
    critic_history_frames = int(cfg.env.critic_obs_history_frames)
    privileged_dim = PRIVILEGED_OBS_DIM * critic_history_frames

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=(64,),
        value_hidden_dims=(64,),
        critic_obs_dim=privileged_dim,
    )
    processor_params, policy_params, value_params = init_network_params(
        ppo_network, obs_dim=obs_dim, action_dim=action_dim, seed=0,
    )

    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    env_state = env.reset(reset_rng)

    _, trajectory = collect_rollout(
        env_step_fn=env.step,
        env_state=env_state,
        policy_params=policy_params,
        value_params=value_params,
        processor_params=processor_params,
        ppo_network=ppo_network,
        rng=rng,
        num_steps=2,
        use_privileged_critic=True,
        critic_includes_actor_obs=False,
    )
    # trajectory.critic_obs shape: (num_steps, privileged_dim) for a single env.
    assert trajectory.critic_obs.shape[-1] == privileged_dim, (
        f"expected critic_obs last-dim {privileged_dim} (privileged only); "
        f"got {trajectory.critic_obs.shape[-1]}"
    )


def test_rollout_critic_obs_shape_when_lever7_on() -> None:
    """When critic_includes_actor_obs=True (smoke2), rollout's per-step
    critic_obs has shape = obs_dim + privileged_dim (concatenated)."""
    import dataclasses
    import jax

    from training.envs.env_info import PRIVILEGED_OBS_DIM
    from training.envs.wildrobot_env import WildRobotEnv
    from training.algos.ppo.ppo_core import create_networks, init_network_params
    from training.core.rollout import collect_rollout

    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    cfg = load_training_config(str(_SMOKE2))
    assert cfg.ppo.critic_includes_actor_obs is True

    env = WildRobotEnv(config=cfg)
    obs_dim = env.observation_size
    action_dim = env.action_size
    critic_history_frames = int(cfg.env.critic_obs_history_frames)
    privileged_dim = PRIVILEGED_OBS_DIM * critic_history_frames
    expected_critic_dim = obs_dim + privileged_dim

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=(64,),
        value_hidden_dims=(64,),
        critic_obs_dim=expected_critic_dim,
    )
    processor_params, policy_params, value_params = init_network_params(
        ppo_network, obs_dim=obs_dim, action_dim=action_dim, seed=0,
    )

    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    env_state = env.reset(reset_rng)

    _, trajectory = collect_rollout(
        env_step_fn=env.step,
        env_state=env_state,
        policy_params=policy_params,
        value_params=value_params,
        processor_params=processor_params,
        ppo_network=ppo_network,
        rng=rng,
        num_steps=2,
        use_privileged_critic=True,
        critic_includes_actor_obs=True,
    )
    assert trajectory.critic_obs.shape[-1] == expected_critic_dim, (
        f"expected critic_obs last-dim {expected_critic_dim} = "
        f"obs_dim({obs_dim}) + privileged_dim({privileged_dim}); "
        f"got {trajectory.critic_obs.shape[-1]}"
    )
