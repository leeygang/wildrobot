from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import _load_positive_int_range, load_training_config
from training.envs.wildrobot_env import WildRobotEnv, _sample_hold_joint_feedback
from training.policy_spec_utils import build_policy_spec_from_training_config


def _native_17d_spec():
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke6_home_rsi.yaml"
    )
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    return build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )


def test_sample_hold_uses_cached_values_until_each_joint_refreshes() -> None:
    cached_pos = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    cached_vel = jnp.asarray([7.0, 8.0], dtype=jnp.float32)
    age = jnp.zeros((2,), dtype=jnp.int32)
    periods = jnp.asarray([2, 3], dtype=jnp.int32)
    phases = jnp.zeros((2,), dtype=jnp.int32)

    pos, vel, cached_pos, cached_vel, age = _sample_hold_joint_feedback(
        current_pos=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        current_vel=jnp.asarray([11.0, 12.0], dtype=jnp.float32),
        cached_pos=cached_pos,
        cached_vel=cached_vel,
        age_steps=age,
        period_steps=periods,
        phase_steps=phases,
        next_step_count=jnp.int32(1),
        ctrl_dt=0.02,
        enabled=True,
    )
    np.testing.assert_array_equal(pos, [0.0, 0.0])
    np.testing.assert_array_equal(vel, [7.0, 8.0])
    np.testing.assert_array_equal(age, [1, 1])

    pos, vel, cached_pos, cached_vel, age = _sample_hold_joint_feedback(
        current_pos=jnp.asarray([2.0, 4.0], dtype=jnp.float32),
        current_vel=jnp.asarray([21.0, 22.0], dtype=jnp.float32),
        cached_pos=cached_pos,
        cached_vel=cached_vel,
        age_steps=age,
        period_steps=periods,
        phase_steps=phases,
        next_step_count=jnp.int32(2),
        ctrl_dt=0.02,
        enabled=True,
    )
    np.testing.assert_allclose(pos, [2.0, 0.0])
    np.testing.assert_allclose(vel, [50.0, 8.0], rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(age, [0, 2])


def test_sample_hold_disabled_returns_raw_feedback() -> None:
    current_pos = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    current_vel = jnp.asarray([3.0, 4.0], dtype=jnp.float32)
    pos, vel, cache_pos, cache_vel, age = _sample_hold_joint_feedback(
        current_pos=current_pos,
        current_vel=current_vel,
        cached_pos=jnp.zeros((2,), dtype=jnp.float32),
        cached_vel=jnp.zeros((2,), dtype=jnp.float32),
        age_steps=jnp.asarray([4, 5], dtype=jnp.int32),
        period_steps=jnp.asarray([6, 7], dtype=jnp.int32),
        phase_steps=jnp.asarray([1, 2], dtype=jnp.int32),
        next_step_count=jnp.int32(9),
        ctrl_dt=0.02,
        enabled=False,
    )
    np.testing.assert_array_equal(pos, current_pos)
    np.testing.assert_array_equal(vel, current_vel)
    np.testing.assert_array_equal(cache_pos, current_pos)
    np.testing.assert_array_equal(cache_vel, current_vel)
    np.testing.assert_array_equal(age, [0, 0])


@pytest.mark.parametrize("bad", ([0, 1], [3, 2], [1], "1,2"))
def test_joint_feedback_period_range_rejects_invalid_values(bad) -> None:
    with pytest.raises(ValueError):
        _load_positive_int_range(
            bad, default=(1, 1), field_name="env.feedback_period"
        )


def test_native_17d_policy_contract_has_expected_dimensions() -> None:
    spec = _native_17d_spec()

    assert spec.model.action_dim == 17
    assert spec.model.obs_dim == 937
    assert not any("wrist" in name for name in spec.robot.actuator_names)


class _ReplaceableModel:
    def replace(self, **kwargs):
        return SimpleNamespace(**kwargs)


def test_subset_domain_randomization_changes_only_policy_actuator_rows() -> None:
    base_gain = jnp.arange(30, dtype=jnp.float32).reshape(5, 6)
    base_bias = -base_gain
    dummy = SimpleNamespace(
        _config=SimpleNamespace(
            env=SimpleNamespace(domain_randomization_enabled=True)
        ),
        _base_geom_friction=jnp.ones((2, 3), dtype=jnp.float32),
        _base_body_mass=jnp.ones((2,), dtype=jnp.float32),
        _base_actuator_gainprm=base_gain,
        _base_actuator_biasprm=base_bias,
        _base_dof_frictionloss=jnp.ones((5,), dtype=jnp.float32),
        _ctrl_mapper=SimpleNamespace(
            policy_to_mj_order_jax=jnp.asarray([0, 2, 4], dtype=jnp.int32)
        ),
        _actuator_dof_addrs=jnp.asarray([0, 2, 4], dtype=jnp.int32),
        _mjx_model=_ReplaceableModel(),
    )
    randomized = WildRobotEnv._get_randomized_mjx_model(
        dummy,
        {
            "friction_scale": jnp.float32(1.0),
            "mass_scales": jnp.ones((2,), dtype=jnp.float32),
            "kp_scales": jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32),
            "frictionloss_scales": jnp.ones((3,), dtype=jnp.float32),
        },
    )

    randomized_gain = np.asarray(randomized.actuator_gainprm)
    randomized_bias = np.asarray(randomized.actuator_biasprm)
    base_gain_np = np.asarray(base_gain)
    base_bias_np = np.asarray(base_bias)
    np.testing.assert_array_equal(randomized_gain[[1, 3]], base_gain_np[[1, 3]])
    np.testing.assert_array_equal(randomized_bias[[1, 3]], base_bias_np[[1, 3]])
    np.testing.assert_allclose(
        randomized_gain[[0, 2, 4], 0],
        base_gain_np[[0, 2, 4], 0] * [2.0, 3.0, 4.0],
    )
