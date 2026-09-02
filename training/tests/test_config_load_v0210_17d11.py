from __future__ import annotations

from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml

from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.v6_eval_adapter import V6EvalAdapter
from training.exports.runtime_metadata import build_runtime_policy_config


CONFIG = Path("training/configs/ppo_walking_v0210_17d11_native_stance_stage1.yaml")
ROLL_NAMES = (
    "left_hip_roll",
    "left_ankle_roll",
    "right_hip_roll",
    "right_ankle_roll",
)


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(CONFIG)


@pytest.fixture(scope="module")
def env(cfg):
    cfg.freeze()
    return WildRobotEnv(config=cfg)


def test_17d11_is_a_guarded_asymmetric_finetune(cfg) -> None:
    assert cfg.version == "0.21.0-17d11-safe"
    assert cfg.ppo.iterations == 10
    assert cfg.checkpoints.interval == 1
    assert cfg.ppo.critic_warmup_iterations == 2
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.source_policy_kl_limit == pytest.approx(0.003)
    assert cfg.ppo.learning_rate == pytest.approx(2.0e-6)
    assert cfg.ppo.epochs == 2
    assert cfg.ppo.target_kl == pytest.approx(0.003)
    assert cfg.ppo.entropy_coef == pytest.approx(0.0)
    assert cfg.ppo.eval.enabled is True
    assert cfg.ppo.eval.interval == 1
    assert cfg.ppo.eval.num_envs == 64
    assert cfg.ppo.eval.num_steps == 1000
    assert cfg.ppo.eval.post_training_top_k == 10
    assert cfg.ppo.rollback.enabled is True
    assert cfg.ppo.rollback.patience == 1
    assert cfg.ppo.rollback.success_rate_drop_threshold == pytest.approx(0.01)
    assert cfg.ppo.rollback.stable_saturation_increase_threshold == pytest.approx(
        0.02
    )
    assert cfg.reward_weights.saturation == pytest.approx(-0.025)
    assert cfg.env.torque_saturation_soft_limit_ratio == pytest.approx(0.8)
    assert cfg.env.torque_saturation_weight_default == pytest.approx(0.0)
    assert cfg.env.torque_saturation_weights_per_joint == {"left_hip_roll": 1.0}
    assert cfg.env.loc_ref_default_stance_width_m == pytest.approx(0.0495)
    assert cfg.env.loc_ref_walking_base_from_ref_init_roll is True
    assert cfg.env.loc_ref_walking_joint_offsets_rad == {}
    assert cfg.env.loc_ref_residual_base == "home"
    assert cfg.env.loc_ref_reset_base == "home"
    assert cfg.env.loc_ref_rsi_enabled is True
    assert cfg.env.loc_ref_rsi_probability == pytest.approx(0.75)


def test_17d11_generated_roll_reference_and_action_base_are_coherent(env) -> None:
    names = tuple(env._policy_spec.robot.actuator_names)
    home = np.asarray(env._home_q_rad)
    walking_base = np.asarray(env._walking_home_q_rad)
    ref_init = np.asarray(env._ref_init_q_rad)

    for name in ROLL_NAMES:
        idx = names.index(name)
        assert walking_base[idx] == pytest.approx(ref_init[idx], abs=1e-7)

    non_roll = [i for i, name in enumerate(names) if name not in ROLL_NAMES]
    np.testing.assert_allclose(walking_base[non_roll], home[non_roll], atol=1e-7)

    left_hip = ref_init[names.index("left_hip_roll")]
    left_ankle = ref_init[names.index("left_ankle_roll")]
    right_hip = ref_init[names.index("right_hip_roll")]
    right_ankle = ref_init[names.index("right_ankle_roll")]
    assert left_hip == pytest.approx(0.01129, abs=2e-4)
    assert right_hip == pytest.approx(-left_hip, abs=1e-7)
    assert left_ankle == pytest.approx(-left_hip, abs=1e-7)
    assert right_ankle == pytest.approx(left_hip, abs=1e-7)

    target, residual = env._compose_target_q_from_residual(
        policy_action=jp.zeros(env.action_size, dtype=jp.float32),
        nominal_q_ref=jp.asarray(ref_init),
    )
    np.testing.assert_allclose(target, walking_base, atol=1e-7)
    np.testing.assert_allclose(residual, 0.0, atol=1e-7)


def test_17d11_native_eval_uses_the_same_walking_base(env) -> None:
    adapter = V6EvalAdapter(
        training_cfg=env._config,
        mj_model=env._mj_model,
        policy_spec=env._policy_spec,
        signals_adapter=env._signals_adapter,
        action_dim=env.action_size,
    )
    np.testing.assert_allclose(
        adapter._walking_home_q_rad,
        np.asarray(env._walking_home_q_rad),
        atol=1e-7,
    )

    data = mujoco.MjData(env._mj_model)
    mujoco.mj_resetDataKeyframe(env._mj_model, data, 0)
    adapter.apply_action(data, np.zeros(env.action_size, dtype=np.float32))
    target_policy_order = np.asarray(data.ctrl)[
        np.asarray(env._ctrl_mapper.policy_to_mj_order_jax)
    ]
    np.testing.assert_allclose(
        target_policy_order,
        env._walking_home_q_rad,
        atol=1e-6,
    )


def test_17d11_export_freezes_the_same_walking_base(env) -> None:
    raw_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    metadata = build_runtime_policy_config(
        env=raw_config["env"],
        spec=env._policy_spec,
    )
    exported_offsets = np.asarray(
        metadata["residual_base_offset_per_actuator"], dtype=np.float32
    )
    np.testing.assert_allclose(
        np.asarray(env._home_q_rad) + exported_offsets,
        np.asarray(env._walking_home_q_rad),
        atol=1e-7,
    )
    assert metadata["loc_ref_default_stance_width_m"] == pytest.approx(0.0495)
    assert metadata["loc_ref_walking_base_from_ref_init_roll"] is True
