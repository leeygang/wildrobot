"""AMP Discriminator for natural motion learning.

Based on the original AMP paper (Peng et al., 2021) using LSGAN formulation:
- Least Squares GAN loss (more stable than BCE)
- AMP reward as POSITIVE BONUS (not penalty)
- Larger network (1024-512-256) for better expressiveness
- LayerNorm for training stability
- ELU activations

LSGAN Formulation:
- Discriminator outputs raw scores (no sigmoid)
- D(real) → 1, D(fake) → 0
- Loss: (D(real) - 1)² + D(fake)²
- Reward: max(0, 1 - 0.25 * (D(s) - 1)²)
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

    hidden_dims: Sequence[int]

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


def create_discriminator(
    obs_dim: int,
    hidden_dims: Sequence[int],
    seed: int = 0,
):
    """Create discriminator network and initialize parameters.

    Args:
        obs_dim: Observation dimension
        hidden_dims: Hidden layer dimensions (REQUIRED - read from config)
        seed: Random seed for initialization

    Returns:
        (model, params): Discriminator module and initialized parameters
    """
    model = AMPDiscriminator(hidden_dims=tuple(hidden_dims))
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
    gradient_penalty_weight=5.0,
    input_noise_std=0.0,
):
    """Compute LSGAN discriminator loss with gradient penalty.

    LSGAN (Least Squares GAN) loss from the original AMP paper:
    - Real samples: (D(x) - 1)² (want D to output 1 for real)
    - Fake samples: (D(x) - 0)² (want D to output 0 for fake)

    Benefits of LSGAN over BCE:
    - More stable gradients (no vanishing gradients when D is confident)
    - Smoother loss landscape
    - Better convergence in adversarial training

    Gradient penalty (WGAN-GP style):
    - Encourages smooth discriminator
    - Prevents mode collapse

    Input noise (optional):
    - Adds Gaussian noise to observations before feeding to discriminator
    - Prevents overfitting to high-frequency jitter unique to simulation
    - Helps discriminator focus on general motion style

    Args:
        params: Discriminator parameters
        model: Discriminator network
        real_obs: Reference motion observations (batch, obs_dim)
        fake_obs: Policy rollout observations (batch, obs_dim)
        rng_key: JAX random key
        gradient_penalty_weight: Weight for gradient penalty term
        input_noise_std: Std of Gaussian noise to add to inputs (0.0 = no noise)

    Returns:
        (loss, metrics): Total loss and dict of metrics
    """
    # Add input noise to prevent discriminator overfitting to high-freq jitter
    if input_noise_std > 0.0:
        rng_key, noise_key1, noise_key2 = jax.random.split(rng_key, 3)
        real_noise = jax.random.normal(noise_key1, real_obs.shape) * input_noise_std
        fake_noise = jax.random.normal(noise_key2, fake_obs.shape) * input_noise_std
        real_obs = real_obs + real_noise
        fake_obs = fake_obs + fake_noise

    # Forward pass on real and fake samples (raw logits, no sigmoid)
    real_scores = model.apply(params, real_obs, training=True)
    fake_scores = model.apply(params, fake_obs, training=True)

    # LSGAN loss: least squares instead of BCE
    # Real samples: want D(x) → 1, so loss = (D(x) - 1)²
    # Fake samples: want D(x) → 0, so loss = D(x)²
    real_loss = jnp.mean((real_scores - 1.0) ** 2)
    fake_loss = jnp.mean(fake_scores ** 2)
    lsgan_loss = 0.5 * (real_loss + fake_loss)

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
    total_loss = lsgan_loss + gradient_penalty_weight * gradient_penalty

    # Metrics for logging
    # For LSGAN, "accuracy" = how well D separates real (→1) from fake (→0)
    # Use 0.5 threshold on raw scores
    accuracy = (
        jnp.mean(real_scores > 0.5).astype(jnp.float32) * 0.5 +
        jnp.mean(fake_scores < 0.5).astype(jnp.float32) * 0.5
    )

    metrics = {
        "discriminator_loss": total_loss,
        "discriminator_lsgan": lsgan_loss,
        "discriminator_gp": gradient_penalty,
        "discriminator_accuracy": accuracy,
        "discriminator_real_mean": jnp.mean(real_scores),  # Should → 1
        "discriminator_fake_mean": jnp.mean(fake_scores),  # Should → 0
    }

    return total_loss, metrics


def compute_amp_reward(params, model, obs):
    """Compute AMP reward for policy training (LSGAN formulation).

    Original AMP paper reward formula (Peng et al., 2021):
        r_amp = max(0, 1 - 0.25 * (D(s) - 1)²)

    This is a POSITIVE BONUS (not a penalty):
    - D(s) = 1.0 (looks like reference) → reward = 1.0 (maximum bonus)
    - D(s) = 0.5 → reward = 0.9375
    - D(s) = 0.0 (looks fake) → reward = 0.75
    - D(s) = -1.0 → reward = 0.0 (clipped)

    Benefits:
    - Always positive (0 to 1 range) - easier to balance with task reward
    - Smooth gradient everywhere (no log singularities)
    - Matches the LSGAN discriminator loss

    Args:
        params: Discriminator parameters
        model: Discriminator network
        obs: Policy observations (batch, obs_dim)

    Returns:
        amp_reward: (batch,) AMP rewards (positive values, 0 to 1)
    """
    # Get raw discriminator scores (not sigmoid - LSGAN uses raw outputs)
    scores = model.apply(params, obs, training=False)

    # LSGAN reward: max(0, 1 - 0.25 * (D(s) - 1)²)
    # When D(s) → 1 (real), reward → 1.0
    # When D(s) → 0 (fake), reward → 0.75
    # When D(s) → -1, reward → 0.0 (clipped)
    amp_reward = jnp.maximum(0.0, 1.0 - 0.25 * (scores - 1.0) ** 2)

    return amp_reward


def create_discriminator_optimizer(learning_rate: float):
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
