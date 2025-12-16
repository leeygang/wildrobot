"""AMP reward wrapper for Brax environments.

Wraps a Brax environment to add AMP (Adversarial Motion Priors) discriminator
rewards and handle discriminator training during rollouts.
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import State

from playground_amp.amp.discriminator import (
    create_discriminator,
    discriminator_loss,
    compute_amp_reward,
    create_discriminator_optimizer,
)
from playground_amp.amp.ref_buffer import create_reference_buffer


class AMPRewardWrapper(envs.Env):
    """Wraps a Brax environment to add AMP discriminator rewards.

    This wrapper:
    1. Intercepts step() calls to inject AMP reward
    2. Collects policy observations for discriminator training
    3. Periodically trains the discriminator

    The discriminator learns to distinguish reference motion (real) from
    policy rollouts (fake), encouraging natural motion.
    """

    def __init__(
        self,
        base_env: envs.Env,
        amp_weight: float = 1.0,
        discriminator_lr: float = 1e-4,
        discriminator_update_freq: int = 100,
        gradient_penalty_weight: float = 5.0,
        obs_dim: int = 44,
        ref_motion_mode: str = "walking",
        ref_motion_num_sequences: int = 200,
        seed: int = 0,
    ):
        """Initialize AMP reward wrapper.

        Args:
            base_env: Base Brax environment to wrap
            amp_weight: Weight for AMP reward (1.0 = equal to task reward)
            discriminator_lr: Learning rate for discriminator
            discriminator_update_freq: Train discriminator every N steps
            gradient_penalty_weight: Weight for WGAN-GP gradient penalty
            obs_dim: Observation dimension
            ref_motion_mode: "walking", "standing", or "file"
            ref_motion_num_sequences: Number of synthetic sequences to generate
            seed: Random seed
        """
        self._base_env = base_env
        self._amp_weight = amp_weight
        self._discriminator_update_freq = discriminator_update_freq
        self._gradient_penalty_weight = gradient_penalty_weight
        self._obs_dim = obs_dim

        # Create discriminator
        self._disc_model, self._disc_params = create_discriminator(obs_dim, seed=seed)
        self._disc_optimizer = create_discriminator_optimizer(discriminator_lr)
        self._disc_opt_state = self._disc_optimizer.init(self._disc_params)

        # Create reference motion buffer
        self._ref_buffer = create_reference_buffer(
            obs_dim=obs_dim,
            mode=ref_motion_mode,
            num_sequences=ref_motion_num_sequences,
        )

        # RNG key for discriminator training
        self._rng = jax.random.PRNGKey(seed + 1000)

        # Metrics tracking
        self._step_count = 0
        self._disc_train_count = 0
        self._recent_disc_loss = 0.0
        self._recent_disc_accuracy = 0.5

        # Observation buffer for discriminator training (collect fake samples)
        # We'll store observations from recent rollouts
        self._obs_buffer = []
        self._max_obs_buffer_size = 10000

        print(f"✓ AMPRewardWrapper initialized")
        print(f"  AMP weight: {amp_weight}")
        print(f"  Discriminator LR: {discriminator_lr}")
        print(f"  Update frequency: every {discriminator_update_freq} steps")
        print(f"  Reference buffer: {len(self._ref_buffer)} sequences")

    @property
    def observation_size(self) -> int:
        return self._base_env.observation_size

    @property
    def action_size(self) -> int:
        return self._base_env.action_size

    @property
    def backend(self) -> str:
        return self._base_env.backend

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment."""
        return self._base_env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment and add AMP reward.

        Args:
            state: Current environment state
            action: Action to take

        Returns:
            New state with AMP reward added
        """
        # Step base environment
        new_state = self._base_env.step(state, action)

        # Compute AMP reward
        # Note: We use the observation from the new state
        obs = new_state.obs

        # Compute AMP reward (vectorized over batch if needed)
        if obs.ndim == 1:
            obs_batch = obs[None, :]  # (1, obs_dim)
        else:
            obs_batch = obs  # Already batched

        amp_reward_batch = compute_amp_reward(
            self._disc_params,
            self._disc_model,
            obs_batch
        )

        # Extract scalar or match original shape
        if obs.ndim == 1:
            amp_reward = amp_reward_batch[0]
        else:
            amp_reward = amp_reward_batch

        # Combine task reward + AMP reward
        combined_reward = new_state.reward + self._amp_weight * amp_reward

        # Update state with combined reward
        new_state = new_state.replace(reward=combined_reward)

        # Increment step count (for tracking)
        self._step_count += 1

        # Add AMP metrics to state.info (but keep it JIT-compatible)
        # We'll use simple scalar values that can be logged
        # NOTE: Don't modify info dict - it causes JAX tracing issues in Brax PPO
        # The metrics will be logged through progress_fn instead

        return new_state

    def train_discriminator(self, policy_observations: jnp.ndarray) -> Dict[str, float]:
        """Train the discriminator on collected observations.

        This should be called periodically (e.g., after each PPO update)
        with observations from policy rollouts.

        Args:
            policy_observations: Observations from policy rollouts (batch, obs_dim)

        Returns:
            Dictionary of training metrics
        """
        if self._ref_buffer.is_empty():
            return {
                "discriminator_loss": 0.0,
                "discriminator_accuracy": 0.5,
                "discriminator_real_mean": 0.5,
                "discriminator_fake_mean": 0.5,
            }

        # Sample reference motion (real)
        batch_size = min(512, policy_observations.shape[0])
        real_obs = self._ref_buffer.sample_single_frames(batch_size)

        # Sample policy motion (fake) - use provided observations
        if policy_observations.shape[0] >= batch_size:
            # Randomly sample from policy observations
            indices = jax.random.choice(
                self._rng,
                policy_observations.shape[0],
                shape=(batch_size,),
                replace=False
            )
            fake_obs = policy_observations[indices]
        else:
            fake_obs = policy_observations

        # Convert to JAX arrays
        real_obs_jax = jnp.asarray(real_obs, dtype=jnp.float32)
        fake_obs_jax = jnp.asarray(fake_obs, dtype=jnp.float32)

        # Split RNG
        self._rng, loss_rng = jax.random.split(self._rng)

        # Compute loss and gradients
        def loss_fn(params):
            return discriminator_loss(
                params,
                self._disc_model,
                real_obs_jax,
                fake_obs_jax,
                loss_rng,
                self._gradient_penalty_weight
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self._disc_params)

        # Apply updates
        updates, self._disc_opt_state = self._disc_optimizer.update(
            grads, self._disc_opt_state
        )
        import optax
        self._disc_params = optax.apply_updates(self._disc_params, updates)

        # Update tracking
        self._disc_train_count += 1
        self._recent_disc_loss = float(loss)
        self._recent_disc_accuracy = float(metrics['discriminator_accuracy'])

        # Convert metrics to Python floats for logging
        metrics_float = {k: float(v) for k, v in metrics.items()}

        return metrics_float

    def get_metrics(self) -> Dict[str, float]:
        """Get current AMP metrics."""
        return {
            "amp/discriminator_loss": self._recent_disc_loss,
            "amp/discriminator_accuracy": self._recent_disc_accuracy,
            "amp/train_count": float(self._disc_train_count),
            "amp/step_count": float(self._step_count),
        }
