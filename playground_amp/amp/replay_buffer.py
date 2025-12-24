"""Policy Replay Buffer for AMP discriminator training.

v0.6.0: Introduced to prevent "catastrophic forgetting" in the discriminator.

Problem:
When training the discriminator only on the current rollout, it can "forget"
how to penalize old bad behaviors that the policy has already learned to avoid.
This creates a cyclic pattern where the policy regresses to old failure modes.

Solution:
Store historical policy samples in a replay buffer. Train the discriminator
on a mix of current rollout + historical samples. This provides more stable
gradients and prevents the discriminator from overfitting to recent behavior.

Industry Practice:
- DeepMind AMP: Uses replay buffer with ~200K samples
- NVIDIA IsaacGym: Stores last 10-20 policy iterations
- ETH Zurich: ~100K sample buffer

Usage:
    buffer = PolicyReplayBuffer(max_size=200000, feature_dim=29)

    # During training
    buffer.add(current_rollout_features)  # Add new samples

    # For discriminator training
    historical_samples = buffer.sample(rng, batch_size=256)
    # Mix with current samples for training
"""

from typing import Tuple
import jax
import jax.numpy as jnp


class PolicyReplayBuffer:
    """GPU-resident replay buffer for policy samples.

    Stores historical policy features for discriminator training.
    All operations are JAX-compatible for use in JIT-compiled training.

    Attributes:
        max_size: Maximum number of samples to store
        feature_dim: Dimension of each feature vector
        data: JAX array storing samples (max_size, feature_dim)
        ptr: Current write position (circular buffer)
        size: Number of valid samples in buffer
    """

    def __init__(self, max_size: int, feature_dim: int):
        """Initialize empty replay buffer.

        Args:
            max_size: Maximum number of samples to store
            feature_dim: Dimension of each feature vector (e.g., 29 for AMP features)
        """
        self.max_size = max_size
        self.feature_dim = feature_dim
        # Pre-allocate GPU memory
        self.data = jnp.zeros((max_size, feature_dim), dtype=jnp.float32)
        self.ptr = 0
        self.size = 0

    def add(self, samples: jnp.ndarray) -> None:
        """Add samples to buffer (circular, overwrites oldest).

        Args:
            samples: New samples to add, shape (num_samples, feature_dim)
        """
        num_samples = samples.shape[0]

        if num_samples >= self.max_size:
            # If adding more samples than buffer size, just keep the last max_size
            self.data = samples[-self.max_size:]
            self.ptr = 0
            self.size = self.max_size
            return

        # Calculate indices for circular insertion
        end_ptr = self.ptr + num_samples

        if end_ptr <= self.max_size:
            # Simple case: no wraparound
            self.data = self.data.at[self.ptr:end_ptr].set(samples)
            self.ptr = end_ptr % self.max_size
        else:
            # Wraparound case
            first_chunk = self.max_size - self.ptr
            self.data = self.data.at[self.ptr:].set(samples[:first_chunk])
            self.data = self.data.at[:end_ptr - self.max_size].set(samples[first_chunk:])
            self.ptr = end_ptr - self.max_size

        self.size = min(self.size + num_samples, self.max_size)

    def sample(self, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample random batch from buffer.

        Args:
            rng: JAX random key
            batch_size: Number of samples to return

        Returns:
            Sampled features, shape (batch_size, feature_dim)
        """
        if self.size == 0:
            # Return zeros if buffer is empty (shouldn't happen in practice)
            return jnp.zeros((batch_size, self.feature_dim), dtype=jnp.float32)

        # Sample with replacement from valid portion of buffer
        indices = jax.random.choice(
            rng, self.size, shape=(batch_size,), replace=True
        )
        return self.data[indices]

    def is_ready(self, min_samples: int = 1000) -> bool:
        """Check if buffer has enough samples for stable training.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            True if buffer has at least min_samples
        """
        return self.size >= min_samples


class JITReplayBuffer:
    """Fully JIT-compatible replay buffer using functional updates.

    Unlike PolicyReplayBuffer which uses in-place mutations, this version
    returns new state for each operation, making it compatible with JAX's
    functional paradigm and jax.lax.scan.

    Use this version inside JIT-compiled training loops.
    """

    @staticmethod
    def create(max_size: int, feature_dim: int) -> dict:
        """Create initial buffer state.

        Args:
            max_size: Maximum number of samples
            feature_dim: Feature dimension

        Returns:
            Dict containing buffer state (data, ptr, size, max_size, feature_dim)
        """
        return {
            "data": jnp.zeros((max_size, feature_dim), dtype=jnp.float32),
            "ptr": jnp.array(0, dtype=jnp.int32),
            "size": jnp.array(0, dtype=jnp.int32),
            "max_size": max_size,
            "feature_dim": feature_dim,
        }

    @staticmethod
    def add(state: dict, samples: jnp.ndarray) -> dict:
        """Add samples to buffer (functional update).

        Args:
            state: Current buffer state
            samples: Samples to add, shape (num_samples, feature_dim)

        Returns:
            New buffer state
        """
        num_samples = samples.shape[0]
        max_size = state["max_size"]
        ptr = state["ptr"]

        # Calculate new indices
        indices = (jnp.arange(num_samples) + ptr) % max_size

        # Scatter samples into buffer
        new_data = state["data"].at[indices].set(samples)

        # Update pointer and size
        new_ptr = (ptr + num_samples) % max_size
        new_size = jnp.minimum(state["size"] + num_samples, max_size)

        return {
            **state,
            "data": new_data,
            "ptr": new_ptr,
            "size": new_size,
        }

    @staticmethod
    def sample(state: dict, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample from buffer.

        Args:
            state: Buffer state
            rng: Random key
            batch_size: Number of samples

        Returns:
            Sampled features, shape (batch_size, feature_dim)
        """
        # Sample indices from valid portion
        indices = jax.random.choice(
            rng,
            state["size"],
            shape=(batch_size,),
            replace=True,
        )
        return state["data"][indices]

    @staticmethod
    def is_ready(state: dict, min_samples: int = 1000) -> jnp.ndarray:
        """Check if buffer has enough samples.

        Args:
            state: Buffer state
            min_samples: Minimum required samples

        Returns:
            Boolean array indicating if ready
        """
        return state["size"] >= min_samples
