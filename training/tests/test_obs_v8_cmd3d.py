"""v0.21.0 P7: ``wr_obs_v8_cmd3d`` actor obs layout tests.

P7 adds a NEW size-2 ``velocity_cmd_lateral_yaw`` slot to the actor obs so
the policy can see ``(vy_cmd, wz_cmd)`` on top of the existing scalar
``velocity_cmd`` slot (which still carries ``vx_cmd``).  v1-v7 must stay
BYTE-IDENTICAL: the shared ``velocity_cmd`` slot keeps size=1, sliced to
``[..., :1]`` so v1-v7 callers can feed it the new (3,) command without
observable change.

The new slot is APPENDED to the v7 layout — it is not a rename or widening
of an existing slot.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from policy_contract.spec import SUPPORTED_LAYOUT_IDS
from policy_contract.spec_builder import build_policy_spec


def _spec(layout_id: str, action_dim: int = 13):
    return build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {
                "name": f"j{i}",
                "range": [-1.0, 1.0],
                "policy_action_sign": 1.0,
                "max_velocity_rad_s": 10.0,
            }
            for i in range(action_dim)
        ],
        action_filter_alpha=0.5,
        layout_id=layout_id,
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * action_dim,
    )


def test_v8_layout_id_in_supported_set() -> None:
    assert "wr_obs_v8_cmd3d" in SUPPORTED_LAYOUT_IDS


def test_v8_obs_dim_equals_v7_plus_two() -> None:
    spec_v7 = _spec("wr_obs_v7_phase_proprio")
    spec_v8 = _spec("wr_obs_v8_cmd3d")
    assert spec_v8.model.obs_dim == spec_v7.model.obs_dim + 2


def test_jax_build_observation_v8_appends_lateral_yaw_slot() -> None:
    from policy_contract.jax.obs import build_observation_from_components

    spec = _spec("wr_obs_v8_cmd3d")
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    obs = build_observation_from_components(
        spec=spec,
        gravity_local=jnp.zeros(3, dtype=jnp.float32),
        angvel_heading_local=jnp.zeros(3, dtype=jnp.float32),
        joint_pos_normalized=jnp.zeros(action_dim, dtype=jnp.float32),
        joint_vel_normalized=jnp.zeros(action_dim, dtype=jnp.float32),
        foot_switches=jnp.zeros(4, dtype=jnp.float32),
        prev_action=jnp.zeros(action_dim, dtype=jnp.float32),
        velocity_cmd=jnp.array([0.20, 0.05, 0.10], dtype=jnp.float32),
        velocity_cmd_lateral_yaw=jnp.array([0.05, 0.10], dtype=jnp.float32),
        loc_ref_phase_sin_cos=jnp.zeros(2, dtype=jnp.float32),
        proprio_history=jnp.zeros(15 * proprio_bundle, dtype=jnp.float32),
    )
    assert obs.shape == (spec.model.obs_dim,)
    # The new (vy, wz) slot must be the LAST two values when the v8
    # branch appends it after the v7 base (per the impl note: the
    # ``velocity_cmd_lateral_yaw`` ObsFieldSpec is appended right
    # before the trailing ``padding`` slot).
    # padding=1 sits at index -1; lateral_yaw=(2,) at indices -3..-2.
    assert float(obs[-3]) == pytest.approx(0.05)
    assert float(obs[-2]) == pytest.approx(0.10)


def test_jax_v7_obs_byte_identical_when_velocity_cmd_passed_as_3vec() -> None:
    """v1-v7 must stay byte-identical when fed a (3,) cmd from the v0.21
    env (sampler now returns (3,) regardless of layout)."""
    from policy_contract.jax.obs import build_observation_from_components

    spec_v7 = _spec("wr_obs_v7_phase_proprio")
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    kw = dict(
        spec=spec_v7,
        gravity_local=jnp.zeros(3, dtype=jnp.float32),
        angvel_heading_local=jnp.zeros(3, dtype=jnp.float32),
        joint_pos_normalized=jnp.zeros(action_dim, dtype=jnp.float32),
        joint_vel_normalized=jnp.zeros(action_dim, dtype=jnp.float32),
        foot_switches=jnp.zeros(4, dtype=jnp.float32),
        prev_action=jnp.zeros(action_dim, dtype=jnp.float32),
        loc_ref_phase_sin_cos=jnp.zeros(2, dtype=jnp.float32),
        proprio_history=jnp.zeros(15 * proprio_bundle, dtype=jnp.float32),
    )
    obs_scalar = build_observation_from_components(
        velocity_cmd=jnp.array([0.20], dtype=jnp.float32),
        **kw,
    )
    obs_3vec = build_observation_from_components(
        velocity_cmd=jnp.array([0.20, 0.05, 0.10], dtype=jnp.float32),
        velocity_cmd_lateral_yaw=jnp.array([0.05, 0.10], dtype=jnp.float32),
        **kw,
    )
    np.testing.assert_array_equal(obs_scalar, obs_3vec)


def test_env_with_v8_layout_passes_lateral_yaw_to_obs_builder() -> None:
    """E2E: env._get_obs under v8 layout must forward velocity_cmd[1:] to
    the obs builder so the resulting obs contains (vy, wz) at the
    indices of the new ``velocity_cmd_lateral_yaw`` slot."""
    from training.configs.training_config import load_training_config
    from training.envs.env_info import WR_INFO_KEY
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env,
            actor_obs_layout_id="wr_obs_v8_cmd3d",
        ),
    )
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(
        velocity_cmd=jnp.array([0.20, 0.05, 0.10], dtype=jnp.float32)
    )
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    # Trigger one step so the env reconstructs the obs with the new cmd.
    state2 = env.step(state, jnp.zeros(env.action_size))
    obs = state2.obs  # shape (obs_dim,)
    # padding=1 at -1; lateral_yaw=(2,) at indices -3..-2.
    np.testing.assert_allclose(float(obs[-3]), 0.05, atol=1e-5)
    np.testing.assert_allclose(float(obs[-2]), 0.10, atol=1e-5)
