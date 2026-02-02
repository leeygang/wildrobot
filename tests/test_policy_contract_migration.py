from __future__ import annotations

from pathlib import Path

import pytest

from policy_contract.spec import PolicySpec, validate_runtime_compat, validate_spec
from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from wr_runtime.utils.mjcf import load_mjcf_model_info


def test_env_observation_size_matches_contract() -> None:
    load_robot_config("assets/v1/robot_config.yaml")
    config = load_training_config("training/configs/ppo_walking.yaml")
    config.freeze()
    env = WildRobotEnv(config)

    assert env.observation_size == env._policy_spec.model.obs_dim  # noqa: SLF001


def test_bundle_validates_against_mjcf() -> None:
    spec_path = Path("policy_contract/policy_spec.json")
    spec = PolicySpec.from_json(spec_path)
    validate_spec(spec)

    mjcf_info = load_mjcf_model_info(Path("assets/v1/wildrobot.xml"))
    validate_runtime_compat(
        spec=spec,
        mjcf_actuator_names=mjcf_info.actuator_names,
        onnx_obs_dim=spec.model.obs_dim,
        onnx_action_dim=spec.model.action_dim,
    )


def test_sim_foot_switches_nonzero() -> None:
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv
    import jax
    import jax.numpy as jnp

    load_robot_config("assets/v1/robot_config.yaml")
    config = load_training_config("training/configs/ppo_walking.yaml")
    config.freeze()
    env = WildRobotEnv(config)

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    signals = env._signals_adapter.read(state.data)  # noqa: SLF001
    foot_switches = signals.foot_switches
    assert float(jnp.max(foot_switches)) > 0.0
