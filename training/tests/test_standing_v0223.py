from __future__ import annotations

import numpy as np


_CONFIG = "training/configs/ppo_standing_home_stabilizer_v0223.yaml"


def _load_config_and_spec():
    from assets.robot_config import get_robot_config, load_robot_config
    from training.configs.training_config import load_training_config
    from training.policy_spec_utils import build_policy_spec_from_training_config

    cfg = load_training_config(_CONFIG)
    load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=get_robot_config(),
    )
    return cfg, spec


def test_v0223_contract_is_fresh_contact_privileged_standing() -> None:
    cfg, spec = _load_config_and_spec()

    assert cfg.version == "0.22.3"
    assert spec.observation.layout_id == "wr_obs_v9_standing"
    assert spec.model.obs_dim == 59
    assert spec.model.action_dim == 17
    assert "foot_switches" not in {
        field.name for field in spec.observation.layout
    }
    assert cfg.ppo.critic_privileged_enabled is True
    assert cfg.ppo.critic_includes_actor_obs is False
    assert cfg.reward_weights.ref_contact_match == 0.0
    assert cfg.reward_weights.standing_support_balance == 1.0
    assert cfg.ppo.eval.post_training_enabled is True
    assert cfg.ppo.eval.post_training_task == "standing"
    assert cfg.ppo.eval.post_training_num_steps == 1000


def test_v0223_actor_observation_ignores_foot_switches() -> None:
    import jax.numpy as jnp

    from policy_contract.jax.obs import build_observation as build_jax_observation
    from policy_contract.jax.signals import Signals as JaxSignals
    from policy_contract.jax.state import PolicyState as JaxPolicyState
    from policy_contract.numpy.obs import build_observation
    from policy_contract.numpy.signals import Signals
    from policy_contract.numpy.state import PolicyState

    _, spec = _load_config_and_spec()
    action_dim = spec.model.action_dim
    home = np.asarray(spec.robot.home_ctrl_rad, dtype=np.float32)
    common = dict(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        joint_pos_rad=home,
        joint_vel_rad_s=np.zeros(action_dim, dtype=np.float32),
    )
    state = PolicyState(prev_action=np.zeros(action_dim, dtype=np.float32))
    cmd = np.zeros(3, dtype=np.float32)

    open_obs = build_observation(
        spec=spec,
        state=state,
        signals=Signals(
            **common,
            foot_switches=np.zeros(4, dtype=np.float32),
        ),
        velocity_cmd=cmd,
    )
    pressed_obs = build_observation(
        spec=spec,
        state=state,
        signals=Signals(
            **common,
            foot_switches=np.ones(4, dtype=np.float32),
        ),
        velocity_cmd=cmd,
    )

    assert open_obs.shape == (59,)
    np.testing.assert_array_equal(open_obs, pressed_obs)

    jax_obs = build_jax_observation(
        spec=spec,
        state=JaxPolicyState(prev_action=jnp.zeros(action_dim, dtype=jnp.float32)),
        signals=JaxSignals(
            quat_xyzw=jnp.asarray(common["quat_xyzw"]),
            gyro_rad_s=jnp.asarray(common["gyro_rad_s"]),
            joint_pos_rad=jnp.asarray(common["joint_pos_rad"]),
            joint_vel_rad_s=jnp.asarray(common["joint_vel_rad_s"]),
            foot_switches=jnp.zeros(4, dtype=jnp.float32),
        ),
        velocity_cmd=jnp.zeros(3, dtype=jnp.float32),
    )
    np.testing.assert_allclose(np.asarray(jax_obs), open_obs, atol=1e-6)


def test_v0223_standing_runtime_accepts_contact_free_actor_contract() -> None:
    from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
    from runtime.wr_runtime.control.standing_policy_runner import StandingPolicyRunner

    _, spec = _load_config_and_spec()

    class _ZeroPolicy:
        last_obs: np.ndarray | None = None

        def predict(self, obs: np.ndarray) -> np.ndarray:
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return np.zeros(spec.model.action_dim, dtype=np.float32)

    policy = _ZeroPolicy()
    robot_io = MockRobotIO(
        actuator_names=list(spec.robot.actuator_names),
        control_dt=0.02,
        home_q_rad=np.asarray(spec.robot.home_ctrl_rad, dtype=np.float32),
    )
    runner = StandingPolicyRunner(
        spec=spec,
        policy=policy,
        robot_io=robot_io,
    )

    runner.step(np.zeros(3, dtype=np.float32))

    assert policy.last_obs is not None
    assert policy.last_obs.shape == (59,)
    assert len(robot_io.written) == 1


def test_v0223_standing_deterministic_gate_checks_support() -> None:
    from training.core.post_training_eval import deterministic_standing_eval_gate

    metrics = {
        "mean_episode_length": 1000.0,
        "left_loaded": 0.99,
        "right_loaded": 0.98,
        "both_loaded": 0.97,
        "load_imbalance": 0.10,
        "body_quat_err_deg": 3.0,
        "torque_sat_frac": 0.01,
        "action_sat_frac": 0.00,
    }
    passed = deterministic_standing_eval_gate(metrics, eval_num_steps=1000)
    assert passed.passed

    failed = deterministic_standing_eval_gate(
        {**metrics, "right_loaded": 0.20, "both_loaded": 0.20},
        eval_num_steps=1000,
    )
    assert not failed.passed
    assert failed.gates["right_loaded"] is False
    assert failed.gates["both_loaded"] is False


def test_v0223_checkpoint_ranking_uses_standing_support_metrics() -> None:
    from training.core.post_training_eval import (
        CheckpointMetricCandidate,
        rank_checkpoint_candidates,
    )

    candidates = [
        CheckpointMetricCandidate(
            checkpoint_path="high_reward_unstable.pkl",
            iteration=10,
            total_steps=1000,
            metrics={
                "episode_reward": 500.0,
                "episode_length": 1200.0,
                "support/both_loaded": 0.50,
                "support/load_imbalance": 0.60,
            },
        ),
        CheckpointMetricCandidate(
            checkpoint_path="stable.pkl",
            iteration=20,
            total_steps=2000,
            metrics={
                "episode_reward": 250.0,
                "episode_length": 995.0,
                "support/both_loaded": 0.98,
                "support/load_imbalance": 0.08,
            },
        ),
    ]

    ranked, used_filter_fallback = rank_checkpoint_candidates(
        candidates,
        top_k=1,
        task="standing",
        episode_length_target=1000,
    )

    assert used_filter_fallback is False
    assert ranked[0].checkpoint_path == "stable.pkl"
    assert ranked[0].train_both_loaded == 0.98


def test_v0223_support_metrics_are_registered() -> None:
    from training.core.metrics_registry import METRIC_NAMES

    assert "reward/standing_support_balance" in METRIC_NAMES
    assert "support/left_loaded" in METRIC_NAMES
    assert "support/right_loaded" in METRIC_NAMES
    assert "support/both_loaded" in METRIC_NAMES
    assert "support/load_imbalance" in METRIC_NAMES
