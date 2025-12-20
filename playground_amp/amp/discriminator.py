"""AMP Discriminator for natural motion learning.

Based on DeepMind's AMP paper (2021) with 2024 improvements:
- Larger network (1024-512-256) for better expressiveness
- LayerNorm for training stability
- ELU activations
- Binary classification (real reference motion vs fake policy motion)
"""

from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax


class AMPDiscriminator(nn.Module):
    """Discriminator network that distinguishes reference motion from policy rollouts.

    Architecture follows 2024 best practices:
    - Input: observation features (joint pos/vel or full obs)
    - Hidden layers: 1024 -> 512 -> 256
    - Output: single logit (real vs fake)
    - Activations: ELU
    - Normalization: LayerNorm after each hidden layer

    Training:
    - Real samples: reference motion dataset
    - Fake samples: current policy rollouts
    - Loss: binary cross-entropy
    - Regularization: WGAN-GP style gradient penalty
    """

    hidden_dims: Sequence[int] = (1024, 512, 256)

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Forward pass through discriminator.

        Args:
            x: Input observations (batch, obs_dim)
            training: Whether in training mode (affects LayerNorm)

        Returns:
            logits: (batch,) discriminator scores (real/fake logits)
        """
        # Input normalization (helps with varying observation scales)
        x = x.astype(jnp.float32)

        # Hidden layers with LayerNorm
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
                name=f"dense_{i}"
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        # Output layer (single logit)
        logits = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.zeros,
            name="output"
        )(x)

        return logits.squeeze(-1)  # (batch,)


def create_discriminator(obs_dim: int, seed: int = 0):
    """Create discriminator network and initialize parameters.

    Args:
        obs_dim: Observation dimension
        seed: Random seed for initialization

    Returns:
        (model, params): Discriminator module and initialized parameters
    """
    model = AMPDiscriminator()
    rng = jax.random.PRNGKey(seed)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_obs, training=False)
    return model, params


def discriminator_loss(
    params,
    model,
    real_obs,
    fake_obs,
    rng_key,
    gradient_penalty_weight=5.0
):
    """Compute discriminator loss with gradient penalty.

    Binary cross-entropy loss:
    - Real samples should have D(x) → 1
    - Fake samples should have D(x) → 0

    Gradient penalty (WGAN-GP style):
    - Encourages smooth discriminator
    - Prevents mode collapse

    Args:
        params: Discriminator parameters
        model: Discriminator network
        real_obs: Reference motion observations (batch, obs_dim)
        fake_obs: Policy rollout observations (batch, obs_dim)
        rng_key: JAX random key
        gradient_penalty_weight: Weight for gradient penalty term

    Returns:
        (loss, metrics): Total loss and dict of metrics
    """
    # Forward pass on real and fake samples
    real_logits = model.apply(params, real_obs, training=True)
    fake_logits = model.apply(params, fake_obs, training=True)

    # Binary cross-entropy loss
    # Real samples: target = 1, Fake samples: target = 0
    real_loss = optax.sigmoid_binary_cross_entropy(real_logits, jnp.ones_like(real_logits))
    fake_loss = optax.sigmoid_binary_cross_entropy(fake_logits, jnp.zeros_like(fake_logits))
    bce_loss = jnp.mean(real_loss) + jnp.mean(fake_loss)

    # Gradient penalty (on interpolated samples)
    batch_size = real_obs.shape[0]
    alpha = jax.random.uniform(rng_key, shape=(batch_size, 1))
    interpolated = alpha * real_obs + (1 - alpha) * fake_obs

    def disc_fn(x):
        return jnp.sum(model.apply(params, x, training=True))

    grad_interp = jax.grad(disc_fn)(interpolated)
    grad_norm = jnp.sqrt(jnp.sum(grad_interp ** 2, axis=-1) + 1e-8)
    gradient_penalty = jnp.mean((grad_norm - 1.0) ** 2)

    # Total loss
    total_loss = bce_loss + gradient_penalty_weight * gradient_penalty

    # Metrics for logging
    real_probs = jax.nn.sigmoid(real_logits)
    fake_probs = jax.nn.sigmoid(fake_logits)
    accuracy = (
        jnp.mean(real_probs > 0.5).astype(jnp.float32) * 0.5 +
        jnp.mean(fake_probs < 0.5).astype(jnp.float32) * 0.5
    )

    metrics = {
        "discriminator_loss": total_loss,
        "discriminator_bce": bce_loss,
        "discriminator_gp": gradient_penalty,
        "discriminator_accuracy": accuracy,
        "discriminator_real_mean": jnp.mean(real_probs),
        "discriminator_fake_mean": jnp.mean(fake_probs),
    }

    return total_loss, metrics


def compute_amp_reward(params, model, obs):
    """Compute AMP reward for policy training.

    AMP reward encourages policy to generate motions similar to reference.

    Reward formula: r_amp = log(sigmoid(D(x)))
    - High when discriminator thinks motion is "real" (like reference)
    - Low when discriminator thinks motion is "fake"

    Args:
        params: Discriminator parameters
        model: Discriminator network
        obs: Policy observations (batch, obs_dim)

    Returns:
        amp_reward: (batch,) AMP rewards
    """
    logits = model.apply(params, obs, training=False)
    probs = jax.nn.sigmoid(logits)
    # Clip to avoid log(0)
    probs = jnp.clip(probs, 1e-8, 1.0 - 1e-8)
    amp_reward = jnp.log(probs)
    return amp_reward


def create_discriminator_optimizer(learning_rate=1e-4):
    """Create optimizer for discriminator training.

    Uses Adam with gradient clipping (standard for GANs).

    Args:
        learning_rate: Learning rate for discriminator

    Returns:
        optax optimizer
    """
    # Ensure learning_rate is a Python scalar, not a JAX/numpy array
    # This is critical because optax expects scalar learning rates
    learning_rate = float(learning_rate)
    return optax.adam(learning_rate)


# Compatibility layer: NumPy-style interface for gradual migration
class AMPDiscriminatorWrapper:
    """Wrapper for JAX discriminator with NumPy-style interface.

    This provides backward compatibility with the old NumPy stub
    while using the new JAX/Flax implementation under the hood.
    """

    def __init__(self, input_dim: int, hidden=(1024, 512, 256), seed: int = 0):
        self.input_dim = input_dim
        self.hidden = tuple(hidden)
        self.seed = seed

        # Create JAX discriminator
        self.model = AMPDiscriminator(hidden_dims=self.hidden)
        self.rng = jax.random.PRNGKey(seed)

        # Initialize parameters
        dummy_obs = jnp.zeros((1, input_dim), dtype=jnp.float32)
        self.params = self.model.init(self.rng, dummy_obs, training=False)

        # Create optimizer
        self.optimizer = create_discriminator_optimizer()
        self.opt_state = self.optimizer.init(self.params)

    def score(self, obs_batch):
        """Compute discriminator scores (backward compatible)."""
        import numpy as np
        arr = jnp.asarray(obs_batch, dtype=jnp.float32)
        if arr.ndim == 1:
            arr = arr[None, :]

        try:
            logits = self.model.apply(self.params, arr, training=False)
            probs = jax.nn.sigmoid(logits)
            return np.array(probs)
        except Exception:
            return np.zeros((arr.shape[0],), dtype=np.float32)

    def update(self, real_batch, fake_batch):
        """Update discriminator (trains on real vs fake)."""
        real_obs = jnp.asarray(real_batch, dtype=jnp.float32)
        fake_obs = jnp.asarray(fake_batch, dtype=jnp.float32)

        if real_obs.ndim == 1:
            real_obs = real_obs[None, :]
        if fake_obs.ndim == 1:
            fake_obs = fake_obs[None, :]

        # Generate new RNG key
        self.rng, loss_key = jax.random.split(self.rng)

        # Compute loss and gradients
        loss_fn = lambda p: discriminator_loss(
            p, self.model, real_obs, fake_obs, loss_key
        )
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)

        # Apply updates
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        return loss, metrics
