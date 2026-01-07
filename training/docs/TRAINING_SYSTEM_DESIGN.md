# WildRobot Training System Design

**Version:** 0.6.0
**Last Updated:** 2024-12-23
**Status:** Production Ready

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Environment](#2-environment)
3. [Observation & Action Space](#3-observation--action-space)
4. [PPO Training](#4-ppo-training)
5. [AMP Discriminator](#5-amp-discriminator)
6. [AMP Feature Extraction](#6-amp-feature-extraction)
7. [Training Pipeline](#7-training-pipeline)
8. [Configuration Reference](#8-configuration-reference)
9. [Metrics & Monitoring](#9-metrics--monitoring)
10. [Future Work](#10-future-work)

---

## 1. System Overview

### 1.1 Architecture

The WildRobot training system combines PPO (Proximal Policy Optimization) with AMP (Adversarial Motion Priors) for natural locomotion learning.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WildRobot Training System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│  │   Policy     │────▶│  Environment │────▶│   Rollout (Transitions)  │   │
│  │  π(a|s)      │     │  (MJX/Brax)  │     │  obs, action, reward...  │   │
│  └──────────────┘     └──────────────┘     └────────────┬─────────────┘   │
│        ▲                                                │                  │
│        │                                                ▼                  │
│        │                                   ┌──────────────────────────┐    │
│        │                                   │  AMP Feature Extraction  │    │
│        │                                   │  (29-dim per frame)      │    │
│        │                                   └────────────┬─────────────┘    │
│        │                                                │                  │
│  ┌─────┴──────┐                            ┌────────────▼─────────────┐    │
│  │    PPO     │◀──── Shaped Reward ◀───────│     Discriminator        │    │
│  │   Update   │      r = r_task +          │     D(s): Real vs Fake   │    │
│  └────────────┘      w_amp * r_amp         └────────────┬─────────────┘    │
│                                                         ▲                  │
│                                            ┌────────────┴─────────────┐    │
│                                            │   Reference Motion Data  │    │
│                                            │   (AMASS via GMR)        │    │
│                                            └──────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components

| Component | File | Description |
|-----------|------|-------------|
| Environment | `envs/wildrobot_env.py` | MJX-based bipedal robot simulation |
| PPO Networks | `training/algos/ppo/ppo_core.py` | Policy and value networks (via Brax) |
| Discriminator | `amp/discriminator.py` | LSGAN with Spectral Normalization |
| Feature Extraction | `amp/amp_features.py` | 29-dim AMP feature vector |
| Replay Buffer | `amp/replay_buffer.py` | Historical policy samples (v0.6.0) |
| Training Loop | `training/trainer_jit.py` | Fully JIT-compiled training |
| Entry Point | `train.py` | CLI and orchestration |

### 1.3 Design Principles

1. **Fully JIT-Compiled**: Entire training loop runs inside JAX/XLA with no Python for-loops
2. **GPU-Resident Data**: Reference motion buffer stays on GPU, no CPU↔GPU transfers
3. **Fixed Normalization**: Feature statistics computed once from reference data (prevents drift)
4. **Fail Fast**: No silent fallbacks - missing data raises immediate errors

---

## 2. Environment

### 2.1 WildRobotEnv

**File:** `training/envs/wildrobot_env.py`

Based on `mujoco_playground.mjx_env.MjxEnv`, providing a vectorized MuJoCo XLA environment.

```python
class WildRobotEnv(MjxEnv):
    """Bipedal walking environment for WildRobot."""
    
    # Timing
    ctrl_dt: float = 0.02      # Control frequency: 50 Hz
    sim_dt: float = 0.002      # Simulation frequency: 500 Hz
    
    # Episode limits
    max_episode_steps: int = 500  # 10 seconds per episode
    
    # Height constraints
    target_height: float = 0.45
    min_height: float = 0.2     # Termination threshold
    max_height: float = 0.7
```

### 2.2 Reward Function

The task reward encourages forward walking while penalizing undesirable behaviors:

```python
reward = (
    w_vel * forward_velocity_reward +    # Track commanded velocity
    w_height * healthy_reward +           # Maintain target height
    w_action * action_rate_penalty +      # Smooth actions
    w_torque * torque_penalty             # Energy efficiency
)
```

**Default Weights:**
| Weight | Value | Description |
|--------|-------|-------------|
| `tracking_lin_vel` | 5.0 | Forward velocity tracking |
| `base_height` | 0.3 | Height maintenance |
| `action_rate` | -0.01 | Action smoothness penalty |
| `torque_penalty` | -0.001 | Energy efficiency |

### 2.3 Termination Conditions

- **Fall Detection**: `height < min_height` (0.2m)
- **Time Limit**: `step_count >= max_episode_steps` (500 steps = 10s)

Episodes ending at time limit are marked as `truncated=True` (success).

---

## 3. Observation & Action Space

### 3.1 Observation Space (37-dim)

| Index | Feature | Dims | Description |
|-------|---------|------|-------------|
| 0-8 | Joint positions | 9 | All actuated joints (normalized) |
| 9-17 | Joint velocities | 9 | All actuated joints |
| 18-20 | Root linear velocity | 3 | Base velocity (world frame) |
| 21-23 | Root angular velocity | 3 | Base angular velocity |
| 24-26 | Gravity direction | 3 | Projected gravity vector |
| 27-29 | Command velocity | 3 | Target velocity command |
| 30-36 | Previous action | 7+ | Last action for smoothness |

### 3.2 Action Space (9-dim)

Continuous actions in [-1, 1] range, mapped to joint position targets:

| Index | Joint | Range |
|-------|-------|-------|
| 0 | waist_yaw | ±0.5 rad |
| 1-4 | left_hip_*, left_knee | ±1.0 rad |
| 5-8 | right_hip_*, right_knee | ±1.0 rad |

**Action Processing:**
```python
# Low-pass filter for smooth motion
action_filtered = α * action_new + (1 - α) * action_prev
# α = 0.7 (configurable)
```

---

## 4. PPO Training

### 4.1 Network Architecture

**File:** `training/algos/ppo/ppo_core.py`

Uses Brax's PPO network factory with separate policy and value networks:

```python
# Policy Network
policy_hidden_dims = [512, 256, 128]  # 3-layer MLP
# Input: obs (37-dim) → Hidden → Action distribution params

# Value Network  
value_hidden_dims = [512, 256, 128]   # 3-layer MLP
# Input: obs (37-dim) → Hidden → Scalar value
```

**Brax Network API:**
```python
# Forward pass
logits = policy_network.apply(processor_params, policy_params, obs)
value = value_network.apply(processor_params, value_params, obs)

# Action sampling
raw_actions = dist.sample_no_postprocessing(logits, rng)
actions = dist.postprocess(raw_actions)  # For environment
log_probs = dist.log_prob(logits, raw_actions)  # For PPO
```

### 4.2 GAE (Generalized Advantage Estimation)

Custom implementation for AMP reward shaping integration:

```python
def compute_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    """
    Computes advantages using GAE-λ.
    
    δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
    
    Args:
        rewards: Shaped rewards (task + AMP)
        gamma: Discount factor (0.99)
        gae_lambda: GAE lambda (0.95)
    """
```

### 4.3 PPO Loss

```python
# Clipped surrogate objective
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clip(ratio, 1 - ε, 1 + ε)
policy_loss = -mean(min(ratio * A, clipped_ratio * A))

# Value loss (MSE)
value_loss = 0.5 * mean((V(s) - returns)²)

# Entropy bonus
entropy_loss = -mean(entropy(π))

# Total loss
total_loss = policy_loss + c_value * value_loss + c_entropy * entropy_loss
```

**Hyperparameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `value_loss_coef` | 0.5 | Value loss weight |
| `entropy_coef` | 0.01 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `num_minibatches` | 8 | Minibatches per update |
| `update_epochs` | 4 | PPO epochs per iteration |

---

## 5. AMP Discriminator

### 5.1 Architecture

**File:** `training/amp/discriminator.py`

```python
class AMPDiscriminator(nn.Module):
    """LSGAN discriminator with Spectral Normalization."""
    
    hidden_dims: Sequence[int] = (1024, 512, 256)
    use_spectral_norm: bool = True
    
    def __call__(self, x, training=True):
        for dim in hidden_dims:
            if use_spectral_norm:
                x = SpectralNormDense(dim)(x)  # v0.6.0
            else:
                x = Dense(dim)(x)
                x = LayerNorm()(x)
            x = elu(x)
        
        return Dense(1)(x).squeeze(-1)  # Raw logit
```

**Architecture Details:**
| Component | Configuration |
|-----------|---------------|
| Input | 29-dim AMP features |
| Hidden | [1024, 512, 256] |
| Activation | ELU |
| Normalization | **Spectral Norm** (v0.6.0) |
| Output | Raw logit (no sigmoid) |
| Initialization | Orthogonal (√2 hidden, 0.01 output) |

### 5.2 Why Spectral Normalization?

**v0.6.0 Change:** Replaced LayerNorm with Spectral Normalization

| Property | LayerNorm | Spectral Norm |
|----------|-----------|---------------|
| Target | Activations | Weights |
| Effect | Reduces covariate shift | Controls Lipschitz constant |
| Guarantee | None for gradients | K-Lipschitz (K=1) |
| Industry Use | General | GAN standard |

**Benefits:**
- Prevents discriminator from "overpowering" the policy
- Stable training without hyperparameter tuning
- Industry standard (StyleGAN, BigGAN, etc.)

### 5.3 Loss Function (LSGAN + R1)

```python
def discriminator_loss(params, model, real_obs, fake_obs, r1_gamma=5.0):
    # Forward pass
    D_real = model.apply(params, real_obs)  # Should → 1
    D_fake = model.apply(params, fake_obs)  # Should → 0
    
    # LSGAN loss
    lsgan_loss = 0.5 * (mean((D_real - 1)²) + mean(D_fake²))
    
    # R1 Regularizer (gradient penalty on REAL samples only)
    grad_real = grad(sum(D(real_obs)))
    r1_penalty = mean(sum(grad_real², axis=-1))
    
    return lsgan_loss + (r1_gamma / 2) * r1_penalty
```

**Why R1 over WGAN-GP:**
- LSGAN uses bounded targets [0, 1]
- WGAN-GP designed for Wasserstein critics (unbounded)
- R1 penalizes gradient on real samples only (simpler, faster)

### 5.4 AMP Reward

```python
def compute_amp_reward(params, model, obs):
    """Clipped linear reward mapping."""
    scores = model.apply(params, obs, training=False)
    return clip(scores, 0.0, 1.0)
```

| D(s) Score | Reward | Interpretation |
|------------|--------|----------------|
| 1.0 | 1.0 | Looks like reference motion |
| 0.5 | 0.5 | Partially realistic |
| 0.0 | 0.0 | Clearly fake |

### 5.5 Policy Replay Buffer (v0.6.0)

**File:** `training/amp/replay_buffer.py`

Stores historical policy samples to prevent discriminator "catastrophic forgetting":

```python
class JITReplayBuffer:
    """Functional replay buffer for JAX jit compatibility."""
    
    @staticmethod
    def create(max_size: int, feature_dim: int) -> dict:
        return {
            'data': zeros((max_size, feature_dim)),
            'ptr': 0,
            'size': 0,
        }
    
    @staticmethod
    def add(state, samples) -> dict:
        # Functional update for jax.lax.scan
        ...
    
    @staticmethod
    def sample(state, rng, batch_size) -> array:
        # Mix fresh + historical samples
        ...
```

**Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `replay_buffer_size` | 0 | 0=disabled, >0=capacity |
| `replay_buffer_ratio` | 0.5 | Ratio of historical samples |

---

## 6. AMP Feature Extraction

### 6.1 Feature Vector (29-dim)

**File:** `training/amp/amp_features.py`

| Index | Feature | Dims | Notes |
|-------|---------|------|-------|
| 0-8 | Joint positions | 9 | All actuated joints |
| 9-17 | Joint velocities | 9 | All actuated joints |
| 18-20 | Root velocity **direction** | 3 | Unit vector (normalized) |
| 21-23 | Root angular velocity | 3 | World frame |
| 24 | Root height | 1 | Z-coordinate |
| 25-28 | Foot contacts | 4 | [L_toe, L_heel, R_toe, R_heel] |

### 6.2 Velocity Normalization

Root linear velocity is normalized to unit direction to remove speed information:

```python
linvel_norm = norm(root_linvel)
is_moving = linvel_norm > 0.1  # m/s threshold

root_linvel_dir = where(
    is_moving,
    root_linvel / (linvel_norm + 1e-8),  # Unit direction
    zeros_like(root_linvel)               # Zero if stationary
)
```

**Why:** Removes speed mismatch between reference data (~0.27 m/s) and policy commands (0.5-1.0 m/s).

### 6.3 Foot Contact Extraction

**v0.5.0:** Real foot contacts extracted from MJX simulation:

```python
# 4-point model
foot_geoms = ['left_toe', 'left_heel', 'right_toe', 'right_heel']

# Get contact forces from simulation
contact_forces = get_geom_contact_forces(state, foot_geom_ids)

# Soft thresholding for continuous values
foot_contacts = tanh(contact_forces / contact_scale)  # 0-1 range
```

### 6.4 Feature Normalization

Features are normalized using **fixed statistics** from reference data:

```python
# Computed ONCE at initialization (not updated during training)
feature_mean, feature_var = compute_normalization_stats(ref_data)

# Applied to both policy and reference features
normalized = (features - mean) / sqrt(var + 1e-8)
```

**Why Fixed Stats:** Prevents distribution drift where running statistics adapt to policy motion, causing reference data to appear "out of distribution".

---

## 7. Training Pipeline

### 7.1 Training Loop Structure

**File:** `training/core/trainer_jit.py`

```
jax.lax.scan(train_iteration, num_iterations)
    └── train_iteration (fully JIT-compiled)
          │
          ├── 1. Collect Rollout (jax.lax.scan over env.step)
          │     └── transitions: obs, action, reward, done, foot_contacts
          │
          ├── 2. Extract AMP Features (batched)
          │     └── 29-dim features from observations
          │
          ├── 3. Compute AMP Rewards (OLD disc_params) ⚠️ CRITICAL
          │     └── r_amp = clip(D(features), 0, 1)
          │
          ├── 4. Train Discriminator (then update params)
          │     └── LSGAN + R1 loss
          │
          ├── 5. Compute GAE Advantages
          │     └── r_shaped = r_task + w_amp * r_amp
          │
          └── 6. PPO Update (jax.lax.scan over epochs/minibatches)
                └── Policy and value network updates
```

### 7.2 Training Order (Critical)

**Correct Order:**
1. Compute AMP rewards using **OLD** discriminator params
2. **Then** train discriminator

```python
# Step 3: Compute AMP rewards FIRST (using OLD disc_params)
amp_rewards = compute_amp_reward(state.disc_params, ...)  # OLD params
#                                 ^^^^^^^^^^^^^^^^^^

# Step 4: THEN train discriminator
new_disc_params, ... = train_discriminator(state.disc_params, ...)
```

**Why:** Ensures policy is rewarded based on the discriminator active when samples were collected, not the updated one.

### 7.3 Training State

```python
class TrainingState(NamedTuple):
    # Networks
    policy_params: Any
    value_params: Any
    processor_params: Any
    disc_params: Any
    
    # Optimizers
    policy_opt_state: Any
    value_opt_state: Any
    disc_opt_state: Any
    
    # Feature normalization (fixed)
    feature_mean: jnp.ndarray
    feature_var: jnp.ndarray
    
    # Replay buffer (v0.6.0)
    replay_buffer_data: jnp.ndarray
    replay_buffer_ptr: jnp.ndarray
    replay_buffer_size: jnp.ndarray
    
    # Progress
    iteration: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.Array
```

### 7.4 JIT Compilation Strategy

Two-level compilation for maximum efficiency:

```python
# Level 1: Single iteration (always available)
train_iteration_fn = jax.jit(train_iteration)

# Level 2: Batch of N iterations (reduces Python overhead)
train_batch_fn = jax.jit(
    lambda state, env, ref: jax.lax.scan(
        train_iteration, (state, env), None, length=N
    )
)
```

**Pre-compilation:** Both functions are compiled before training starts to avoid mid-training compilation pauses.

---

## 8. Configuration Reference

### 8.1 Config File

**File:** `training/configs/ppo_amass_training.yaml`

```yaml
version: "0.6.0"
version_name: "Spectral Normalization + Policy Replay Buffer"

trainer:
  num_envs: 1024
  rollout_steps: 128
  iterations: 3000
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.01
  log_interval: 10

networks:
  policy_hidden_dims: [512, 256, 128]
  value_hidden_dims: [512, 256, 128]

amp:
  weight: 0.5
  disc_lr: 1e-4
  batch_size: 2048
  update_steps: 3
  discriminator_hidden: [1024, 512, 256]
  r1_gamma: 5.0
  disc_input_noise_std: 0.02
  # v0.6.0: Policy Replay Buffer
  replay_buffer_size: 100000  # 0 = disabled
  replay_buffer_ratio: 0.5

reward_weights:
  tracking_lin_vel: 5.0
  base_height: 0.3
  action_rate: -0.01
  torque_penalty: -0.001
```

### 8.2 CLI Arguments

```bash
python training/train.py \
    --config configs/ppo_amass_training.yaml \
    --iterations 5000 \
    --num-envs 2048 \
    --amp-weight 0.5 \
    --lr 3e-4 \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `ppo_amass_training.yaml` | Config file path |
| `--iterations` | From config | Training iterations |
| `--num-envs` | 1024 | Parallel environments |
| `--lr` | 3e-4 | Learning rate |
| `--amp-weight` | 0.5 | AMP reward weight |
| `--amp-data` | From config | Reference motion path |
| `--verify` | False | Quick smoke test mode |

---

## 9. Metrics & Monitoring

### 9.1 Key Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| `disc_acc` | 0.55-0.80 | Healthy: oscillating. Stuck at 0.50 = not learning |
| `amp_reward` | 0.3-0.8 | Style quality. Near 0 = fake, Near 1 = natural |
| `episode_reward` | Increasing | Total shaped reward |
| `success_rate` | >50% | Episodes reaching max steps without falling |
| `forward_velocity` | ~0.5-1.0 m/s | Tracking commanded velocity |

### 9.2 Discriminator Distribution Metrics

| Metric | Expected | Issue If |
|--------|----------|----------|
| `disc_real_mean` | → 1.0 | <0.7 = disc too weak |
| `disc_fake_mean` | → 0.0 | >0.5 = disc too weak |
| `disc_real_std` | 0.1-0.3 | High = unstable |
| `disc_fake_std` | 0.2-0.5 | Low = overconfident |

### 9.3 W&B Integration

Metrics are automatically logged to Weights & Biases when enabled:

```yaml
wandb:
  enabled: true
  project: "wildrobot-amp"
  entity: null  # Uses default
```

---

## 10. Future Work

### 10.1 Temporal Context (v0.7.0)

Pass 2-3 frame window instead of single frame:

```python
# Current: 29-dim single frame
# Proposed: 87-dim (3 frames × 29 features)
amp_features = concatenate([obs_t_minus_2, obs_t_minus_1, obs_t], axis=-1)
```

**Benefits:**
- Enables judging acceleration and jerk
- Better detection of "teleporting jitter" vs smooth motion
- Industry standard (DeepMind, NVIDIA)

**Infrastructure Ready (v0.5.0):**
- `TemporalFeatureConfig` dataclass
- `create_temporal_buffer()` function
- `update_temporal_buffer()` function

### 10.2 Multi-Speed Reference Data

Augment reference data with multiple walking speeds:
- Slow: 0.3-0.5 m/s
- Normal: 0.5-0.8 m/s
- Fast: 0.8-1.2 m/s

### 10.3 Curriculum Learning

Progressive velocity command curriculum:
1. Start with slow commands (0.3 m/s)
2. Gradually increase to target speed
3. Helps policy learn stable gaits first

---

## References

1. **Peng et al. (2021)** - "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
2. **Miyato et al. (2018)** - "Spectral Normalization for Generative Adversarial Networks"
3. **Mescheder et al. (2018)** - "Which Training Methods for GANs do actually Converge?" (R1 regularizer)
4. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
5. **Brax Documentation** - https://github.com/google/brax

---

**Document End**
