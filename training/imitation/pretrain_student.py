#!/usr/bin/env python3
"""Behavior-cloning warm start for v0.16.1 student handoff."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_contract.spec import policy_spec_hash
from policy_contract.spec_builder import build_policy_spec
from training.algos.ppo.ppo_core import create_networks, init_network_params
from training.configs.training_config import (
    get_robot_config,
    load_robot_config,
    load_training_config,
)
from training.imitation.dataset import (
    list_shards,
    load_metadata,
    load_rollout_shard,
)


def _load_dataset_arrays(dataset_dir: str | Path) -> Dict[str, np.ndarray]:
    dataset_dir = Path(dataset_dir)
    metadata = load_metadata(dataset_dir / "metadata.json")
    shards = list_shards(dataset_dir)
    if not shards:
        raise ValueError(f"No rollout shards found in {dataset_dir}")

    merged: Dict[str, list[np.ndarray]] = {}
    for shard_path in shards:
        shard = load_rollout_shard(shard_path)
        for key, value in shard.items():
            merged.setdefault(key, []).append(value)
    arrays = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    if arrays["obs"].shape[0] != metadata.num_samples:
        raise ValueError(
            f"Dataset sample mismatch: metadata={metadata.num_samples}, "
            f"loaded={arrays['obs'].shape[0]}"
        )
    return arrays


def pretrain_student_behavior_cloning(
    *,
    training_config_path: str | Path,
    dataset_dir: str | Path,
    output_checkpoint_path: str | Path,
    steps: int = 200,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a compact BC pretrain and export a PPO-compatible warm-start checkpoint."""
    cfg = load_training_config(training_config_path)
    load_robot_config(cfg.env.robot_config_path)
    robot_cfg = get_robot_config()
    spec = build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=robot_cfg.actuated_joints,
        action_filter_alpha=float(cfg.env.action_filter_alpha),
        layout_id=str(cfg.env.actor_obs_layout_id),
    )
    spec_hash = policy_spec_hash(spec)
    data = _load_dataset_arrays(dataset_dir)
    obs = jnp.asarray(data["obs"], dtype=jnp.float32)
    actions = jnp.asarray(data["actions"], dtype=jnp.float32)
    done = np.asarray(data["done"], dtype=np.float32)
    forward_velocity = np.asarray(data["forward_velocity"], dtype=np.float32)
    root_pitch = np.asarray(data.get("root_pitch", np.zeros_like(done)), dtype=np.float32)
    root_pitch_rate = np.asarray(data.get("root_pitch_rate", np.zeros_like(done)), dtype=np.float32)

    ppo_network = create_networks(
        obs_dim=spec.model.obs_dim,
        action_dim=spec.model.action_dim,
        policy_hidden_dims=cfg.networks.actor.hidden_sizes,
        value_hidden_dims=cfg.networks.critic.hidden_sizes,
    )
    processor_params, policy_params, value_params = init_network_params(
        ppo_network,
        spec.model.obs_dim,
        spec.model.action_dim,
        seed=seed,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(policy_params)
    sample_count = int(obs.shape[0])
    if sample_count == 0:
        raise ValueError("Dataset is empty")

    @jax.jit
    def _update_step(
        params: Any,
        state: optax.OptState,
        obs_batch: jnp.ndarray,
        action_batch: jnp.ndarray,
    ) -> tuple[Any, optax.OptState, jnp.ndarray]:
        def _loss_fn(p):
            logits = ppo_network.policy_network.apply(processor_params, p, obs_batch)
            pred_actions = ppo_network.parametric_action_distribution.mode(logits)
            return jnp.mean(jnp.square(pred_actions - action_batch))

        loss, grads = jax.value_and_grad(_loss_fn)(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    rng = np.random.default_rng(seed)
    last_loss = np.float32(0.0)
    for _ in range(int(steps)):
        idx = rng.integers(0, sample_count, size=(min(batch_size, sample_count),))
        obs_batch = obs[idx]
        action_batch = actions[idx]
        policy_params, opt_state, loss = _update_step(
            policy_params, opt_state, obs_batch, action_batch
        )
        last_loss = np.asarray(loss, dtype=np.float32)

    pitch_failure_rate = float(np.mean((np.abs(root_pitch) > 0.6).astype(np.float32)))
    pitch_rate_failure = float(np.mean((np.abs(root_pitch_rate) > 1.5).astype(np.float32)))
    survival_rate = float(np.mean(1.0 - done))
    forward_velocity_mean = float(np.mean(forward_velocity))
    checkpoint = {
        "checkpoint_type": "student_pretrain_v0.16.1",
        "policy_params": jax.device_get(policy_params),
        "value_params": jax.device_get(value_params),
        "processor_params": jax.device_get(processor_params),
        "policy_spec_hash": spec_hash,
        "metrics": {
            "imitation_loss": float(last_loss),
            "forward_velocity": forward_velocity_mean,
            "pitch_failure_rate": pitch_failure_rate,
            "pitch_rate_failure_rate": pitch_rate_failure,
            "survival_rate": survival_rate,
        },
        "config": {
            "actor_hidden_sizes": cfg.networks.actor.hidden_sizes,
            "critic_hidden_sizes": cfg.networks.critic.hidden_sizes,
            "observation_layout": cfg.env.actor_obs_layout_id,
        },
    }

    output_path = Path(output_checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(checkpoint, f)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--output-checkpoint", required=True, type=str)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = pretrain_student_behavior_cloning(
        training_config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_checkpoint_path=args.output_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print("Student pretraining complete:")
    print(f"  checkpoint: {args.output_checkpoint}")
    print(f"  imitation_loss: {ckpt['metrics']['imitation_loss']:.6f}")
    print(f"  forward_velocity: {ckpt['metrics']['forward_velocity']:.4f}")
    print(f"  survival_rate: {ckpt['metrics']['survival_rate']:.4f}")


if __name__ == "__main__":
    main()
