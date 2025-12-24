"""AMP Discriminator for natural motion learning.

Based on the original AMP paper (Peng et al., 2021) using LSGAN formulation:
- Least Squares GAN loss (more stable than BCE)
- AMP reward as POSITIVE BONUS (not penalty)
- Larger network (512-256 or 1024-512) for better expressiveness
- Spectral Normalization for Lipschitz control (v0.6.0)
- ELU activations

LSGAN Formulation:
- Discriminator outputs raw scores (no sigmoid)
- D(real) → 1, D(fake) → 0
- Loss: (D(real) - 1)² + D(fake)²
- Reward: clipped linear mapping (D=0 → 0, D=1 → 1)

Version History:
- v0.4.0: Reward formula changed from quadratic to clipped linear
- v0.4.1: Replaced WGAN-GP with R1 regularizer for loss consistency
- v0.6.0: Replaced LayerNorm with Spectral Normalization for strict Lipschitz control
"""

from typing import Sequence, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax


class SpectralNormDense(nn.Module):
    """Dense layer with Spectral Normalization.

    Spectral Normalization constrains the Lipschitz constant of the layer
    by normalizing the weight matrix by its largest singular value.

    This is the industry standard for GAN discriminators (Miyato et al., 2018)
    and prevents the discriminator from "overpowering" the generator/policy.
    """
    features: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Create the dense layer
        dense = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        # Wrap with spectral normalization
        # SpectralNorm normalizes weights by their largest singular value
        spectral_dense = nn.SpectralNorm(
            dense,
            collection_name="batch_stats",  # Store singular vectors here
        )

        return spectral_dense(x)


class AMPDiscriminator(nn.Module):
    """Discriminator network that distinguishes reference motion from policy rollouts.

    Architecture follows 2024 best practices:
    - Input: observation features (joint pos/vel or full obs)
    - Hidden layers: 512 -> 256 (configurable)
    - Output: single logit (real vs fake)
    - Activations: ELU
    - Normalization: Spectral Normalization (v0.6.0, was LayerNorm)

    v0.6.0 Change: Replaced LayerNorm with Spectral Normalization
    - LayerNorm: Normalizes activations (helps with internal covariate shift)
    - SpectralNorm: Normalizes weights by singular value (controls Lipschitz constant)

    Why Spectral Normalization:
    - Industry standard for GAN discriminators (Miyato et al., 2018)
    - Strictly controls Lipschitz constant of the discriminator
    - Prevents discriminator from "overpowering" the policy
    - More theoretically grounded than LayerNorm for adversarial training

    Training:
    - Real samples: reference motion dataset
    - Fake samples: current policy rollouts
    - Loss: LSGAN (least squares)
    - Regularization: R1 gradient penalty on real samples
    """

    hidden_dims: Sequence[int]
    use_spectral_norm: bool = True  # v0.6.0: Can disable for A/B testing

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Forward pass through discriminator.

        Args:
            x: Input observations (batch, obs_dim)
            training: Whether in training mode (affects SpectralNorm updates)

        Returns:
            logits: (batch,) discriminator scores (real/fake logits)
        """
        x = x.astype(jnp.float32)

        for i, hidden_dim in enumerate(self.hidden_dims):
            if self.use_spectral_norm:
                # v0.6.0: Spectral Normalization (industry standard)
                x = SpectralNormDense(
                    features=hidden_dim,
                    kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
                    bias_init=nn.initializers.zeros,
                    name=f"spectral_dense_{i}"
                )(x, training=training)
            else:
                # Legacy: LayerNorm (for A/B testing)
                x = nn.Dense(
                    hidden_dim,
                    kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
                    bias_init=nn.initializers.zeros,
                    name=f"dense_{i}"
                )(x)
                x = nn.LayerNorm(name=f"ln_{i}")(x)

            x = nn.elu(x)

        # Output layer (single logit) - no spectral norm on output
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
    r1_gamma=5.0,
    input_noise_std=0.0,
):
    """Compute LSGAN discriminator loss with R1 regularization.

    v0.4.1: Replaced WGAN-GP with R1 regularizer for consistency with LSGAN.

    LSGAN (Least Squares GAN) loss:
    - Real samples: (D(x) - 1)² (want D to output 1 for real)
    - Fake samples: (D(x) - 0)² (want D to output 0 for fake)

    R1 Regularization (Mescheder et al., 2018):
    - Penalizes gradient norm on REAL samples only
    - Formula: R1 = γ/2 * E[||∇D(x_real)||²]
    - More appropriate for classification-style losses than WGAN-GP
    - WGAN-GP was designed for Wasserstein critics (unbounded outputs)

    Why R1 over WGAN-GP:
    - LSGAN uses bounded targets [0, 1]
    - WGAN-GP uses interpolated samples (makes sense for Wasserstein, not LSGAN)
    - R1 only penalizes gradient on real samples (simpler, faster, more stable)

    Input noise (optional):
    - Adds Gaussian noise to observations before feeding to discriminator
    - Prevents overfitting to high-frequency jitter unique to simulation

    Args:
        params: Discriminator parameters
        model: Discriminator network
        real_obs: Reference motion observations (batch, obs_dim)
        fake_obs: Policy rollout observations (batch, obs_dim)
        rng_key: JAX random key
        r1_gamma: Weight for R1 gradient penalty (default 5.0)
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

    # R1 Regularization: gradient penalty on REAL samples only
    # This is more appropriate for LSGAN than WGAN-GP (which uses interpolated samples)
    def real_disc_sum(x):
        return jnp.sum(model.apply(params, x, training=True))

    grad_real = jax.grad(real_disc_sum)(real_obs)
    # R1 = (gamma/2) * E[||grad||^2]
    r1_penalty = jnp.mean(jnp.sum(grad_real ** 2, axis=-1))

    # Total loss
    total_loss = lsgan_loss + (r1_gamma / 2.0) * r1_penalty

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
        "discriminator_r1": r1_penalty,
        "discriminator_accuracy": accuracy,
        # Distribution metrics for debugging
        "discriminator_real_mean": jnp.mean(real_scores),  # Should → 1
        "discriminator_real_std": jnp.std(real_scores),
        "discriminator_fake_mean": jnp.mean(fake_scores),  # Should → 0
        "discriminator_fake_std": jnp.std(fake_scores),
        # Min/max for histogram-like understanding
        "discriminator_real_min": jnp.min(real_scores),
        "discriminator_real_max": jnp.max(real_scores),
        "discriminator_fake_min": jnp.min(fake_scores),
        "discriminator_fake_max": jnp.max(fake_scores),
    }

    return total_loss, metrics


def compute_amp_reward(params, model, obs):
    """Compute AMP reward for policy training (clipped linear formulation).

    v0.4.0 Fix: Changed from quadratic to clipped linear mapping.

    Old (buggy) formula: max(0, 1 - 0.25 * (D(s) - 1)²)
    - D(s) = 0.0 (fake) → reward = 0.75 (too high!)

    New (correct) formula: clip(D(s), 0, 1)
    - D(s) = 1.0 (looks like reference) → reward = 1.0
    - D(s) = 0.5 → reward = 0.5
    - D(s) = 0.0 (looks fake) → reward = 0.0
    - D(s) < 0 → reward = 0.0 (clipped)

    This matches industry standard (NVIDIA IsaacGym, DeepMind) where
    fake motion receives zero style reward, providing proper gradient
    signal for the policy to improve.

    Args:
        params: Discriminator parameters
        model: Discriminator network
        obs: Policy observations (batch, obs_dim)

    Returns:
        amp_reward: (batch,) AMP rewards (0 to 1 range)
    """
    # Get raw discriminator scores (not sigmoid - LSGAN uses raw outputs)
    scores = model.apply(params, obs, training=False)

    # Clipped linear reward: D=0 → 0, D=1 → 1
    # This ensures fake motion (D≈0) gets zero reward, not 0.75
    amp_reward = jnp.clip(scores, 0.0, 1.0)

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
