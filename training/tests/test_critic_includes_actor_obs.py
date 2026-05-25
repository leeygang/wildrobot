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


# ---------------------------------------------------------------------------
# Resume validator: hard-fail when critic_includes_actor_obs differs
# ---------------------------------------------------------------------------
def test_resume_validator_hard_fails_on_lever7_mismatch(tmp_path) -> None:
    """A pre-Lever-7 ckpt (critic_includes_actor_obs=False) under a
    Lever-7 config (True) would silently load wrong-shape value_params
    and crash later in value_network.apply.  The resume validator
    MUST hard-fail this combination."""
    from training.core.training_loop import _validate_resume_checkpoint

    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    cfg = load_training_config(str(_SMOKE2))
    assert cfg.ppo.critic_includes_actor_obs is True  # smoke2 opts in

    # Simulate loading a pre-Lever-7 ckpt (flag absent in stored config
    # OR explicitly False).  Resume validator must raise.
    # policy_spec_hash must be present + matching so the validator reaches
    # the hard_mismatch raise (the spec-hash check raises earlier on its own).
    fake_hash = "deadbeef"
    ckpt_config_pre_lever7 = {
        "num_envs": cfg.ppo.num_envs,
        "rollout_steps": cfg.ppo.rollout_steps,
        "num_minibatches": cfg.ppo.num_minibatches,
        "epochs": cfg.ppo.epochs,
        "max_grad_norm": cfg.ppo.max_grad_norm,
        "gamma": cfg.ppo.gamma,
        "gae_lambda": cfg.ppo.gae_lambda,
        "clip_epsilon": cfg.ppo.clip_epsilon,
        "value_loss_coef": cfg.ppo.value_loss_coef,
        "critic_privileged_enabled": cfg.ppo.critic_privileged_enabled,
        "critic_includes_actor_obs": False,  # pre-Lever-7 / smoke1
        "actor_hidden_sizes": cfg.networks.actor.hidden_sizes,
        "critic_hidden_sizes": cfg.networks.critic.hidden_sizes,
        "policy_spec_hash": fake_hash,
    }
    with pytest.raises(ValueError, match="critic_includes_actor_obs"):
        _validate_resume_checkpoint(
            ckpt_metrics={},
            ckpt_config=ckpt_config_pre_lever7,
            config=cfg,
            current_policy_spec_hash=fake_hash,
        )


def test_resume_validator_accepts_lever7_match() -> None:
    """Sanity: matching Lever-7 flag passes validation cleanly."""
    from training.core.training_loop import _validate_resume_checkpoint

    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    cfg = load_training_config(str(_SMOKE2))

    fake_hash = "deadbeef"
    ckpt_config_matching = {
        "num_envs": cfg.ppo.num_envs,
        "rollout_steps": cfg.ppo.rollout_steps,
        "num_minibatches": cfg.ppo.num_minibatches,
        "epochs": cfg.ppo.epochs,
        "max_grad_norm": cfg.ppo.max_grad_norm,
        "gamma": cfg.ppo.gamma,
        "gae_lambda": cfg.ppo.gae_lambda,
        "clip_epsilon": cfg.ppo.clip_epsilon,
        "value_loss_coef": cfg.ppo.value_loss_coef,
        "critic_privileged_enabled": cfg.ppo.critic_privileged_enabled,
        "critic_includes_actor_obs": cfg.ppo.critic_includes_actor_obs,  # matches
        "actor_hidden_sizes": cfg.networks.actor.hidden_sizes,
        "critic_hidden_sizes": cfg.networks.critic.hidden_sizes,
        "policy_spec_hash": fake_hash,
    }
    # Should not raise.
    _validate_resume_checkpoint(
        ckpt_metrics={},
        ckpt_config=ckpt_config_matching,
        config=cfg,
        current_policy_spec_hash=fake_hash,
    )


def test_checkpoint_save_includes_critic_includes_actor_obs() -> None:
    """The checkpoint config metadata MUST carry critic_includes_actor_obs
    so the resume validator can hard-check it.  Inline source check
    guards against a refactor silently dropping the field."""
    import inspect
    from training.core import checkpoint

    src = inspect.getsource(checkpoint)
    assert '"critic_includes_actor_obs": config.ppo.critic_includes_actor_obs' in src, (
        'checkpoint save MUST persist "critic_includes_actor_obs" in the '
        "config dict so resume validation can hard-fail an incompatible "
        "value-head shape (pre-Lever-7 ckpt vs Lever-7 config or vice "
        "versa)."
    )
