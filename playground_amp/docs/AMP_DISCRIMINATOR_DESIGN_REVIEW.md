# AMP Discriminator Design Review

**Document Version:** 2.1
**Date:** 2025-12-23
**Status:** ✅ Reviewed & Fixes Implemented (v0.4.1)
**Project:** WildRobot AMP Walking

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [External Review Feedback](#external-review-feedback)
3. [Background: AMP Overview](#background-amp-overview)
4. [Current Implementation](#current-implementation)
   - [Architecture](#architecture)
   - [Loss Function](#loss-function)
   - [Reward Computation](#reward-computation)
   - [Training Pipeline](#training-pipeline)
   - [Feature Extraction](#feature-extraction)
4. [Identified Issues](#identified-issues)
   - [Critical: Reward Formula Bug](#critical-reward-formula-bug)
   - [Critical: Training Order Bug](#critical-training-order-bug)
   - [Moderate: Network Capacity](#moderate-network-capacity)
   - [Minor: No Replay Buffer](#minor-no-replay-buffer)
5. [Proposed Improvements](#proposed-improvements)
6. [Implementation Plan](#implementation-plan)
7. [Validation Criteria](#validation-criteria)
8. [References](#references)

---

## Executive Summary

This document reviews the end-to-end design and implementation of the AMP (Adversarial Motion Priors) discriminator for the WildRobot locomotion project. The review identifies **two critical bugs** and **two moderate issues** that explain the observed training instability (`disc_acc` stuck at 0.50).

| Category | Status | Impact |
|----------|--------|--------|
| Reward Formula | 🔴 Critical Bug | Policy gets 75% reward even when fully fake |
| Training Order | 🔴 Critical Bug | Reward computed with already-updated discriminator |
| Network Size | ⚠️ Moderate | May limit discriminator expressiveness |
| Replay Buffer | ⚠️ Minor | No historical policy samples for training |

**Recommendation:** Fix critical bugs immediately before further hyperparameter tuning.

---

## External Review Feedback

**Review Date:** 2025-12-23
**Reviewer:** External Expert

### Confirmation of Critical Bugs

The external review confirmed both critical issues identified in this document:

1. **Reward Formula Bug ("The Participation Trophy"):** Confirmed that giving **0.75 reward** when the discriminator is 100% sure the motion is fake violates industry standards. In the original Peng et al. 2021 paper, fake motion should receive **0.0 reward**.

2. **Training Order Bug ("The Hindsight Bias"):** Computing rewards using the *updated* discriminator parameters for a rollout collected with the *old* policy creates a "moving target" that destabilizes the PPO value function.

> **Industry Gold Rule:** Always compute rewards using the discriminator parameters that were active when the samples were collected.

### Additional Recommendations from External Review

| Feature | Current Setup | Industry "Gold Rule" | Recommendation |
| --- | --- | --- | --- |
| **Capacity** | `[256, 128]` | `[512, 256]` or `[1024, 512]` | **Increase.** Current network too small for AMASS manifold. |
| **Normalization** | `LayerNorm` | **Spectral Normalization** | **Consider switching.** SpectralNorm controls Lipschitz constant. |
| **Buffer** | None (Current Rollout only) | **Policy Replay Buffer** | **Add.** Prevents "catastrophic forgetting". |
| **Temporal Context** | Single frame (29-dim) | 2-3 frame window (58-87 dim) | **Future work.** Enables judging acceleration/jerk. |

### Feature Engineering Insight

> **Observation History (Temporal Context):** SOTA implementations (DeepMind, NVIDIA) rarely use a single frame. They pass a "window" of the last 2–3 frames to the discriminator.
>
> **Why?** A single frame can show a pose, but it can't distinguish between a "smooth swing" and a "teleporting jitter." Temporal context allows the discriminator to judge **acceleration and jerk**, which are the hallmarks of natural motion.

### Success Indicator from Reviewer

> "Once these are fixed, your `disc_acc` should immediately jump from 0.50 to roughly **0.60–0.80** within the first 20 iterations. If it stays at 0.50 after these fixes, you likely have a data-loading issue with your AMASS reference buffer."

---

## Implementation Status (v0.4.1)

| Fix | Priority | Status | File |
|-----|----------|--------|----- |
| Reward Formula | P0 | ✅ **IMPLEMENTED** (v0.4.0) | `discriminator.py` |
| Training Order | P0 | ✅ **IMPLEMENTED** (v0.4.0) | `trainer_jit.py` |
| Network Size `[512, 256]` | P1 | ✅ **IMPLEMENTED** (v0.4.0) | `ppo_amass_training.yaml` |
| **R1 Regularizer** | P1 | ✅ **IMPLEMENTED** (v0.4.1) | `discriminator.py`, `trainer_jit.py` |
| **Distribution Metrics** | P1 | ✅ **IMPLEMENTED** (v0.4.1) | `trainer_jit.py` |
| Spectral Normalization | P2 | 📋 TODO (v0.5.0) | - |
| Policy Replay Buffer | P2 | 📋 TODO (v0.5.0) | - |
| Temporal Context (2-3 frames) | P3 | 📋 TODO (v0.6.0) | - |

---

## Background: AMP Overview

Adversarial Motion Priors (Peng et al., 2021) combines reinforcement learning with adversarial training to produce natural motion:

```
┌─────────────────────────────────────────────────────────────┐
│                    AMP Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  Policy  │────▶│  Environment │────▶│  Rollout     │   │
│  │  π(a|s)  │     │              │     │  Samples     │   │
│  └──────────┘     └──────────────┘     └──────┬───────┘   │
│       ▲                                       │            │
│       │                                       ▼            │
│       │                              ┌──────────────┐      │
│       │                              │ Discriminator│      │
│  ┌────┴─────┐                        │   D(s)       │      │
│  │   PPO    │◀───── AMP Reward ◀─────┤              │      │
│  │  Update  │       r_amp = f(D(s))  │ Real vs Fake │      │
│  └──────────┘                        └──────┬───────┘      │
│                                             ▲              │
│                              ┌──────────────┴─────────┐    │
│                              │   Reference Motion     │    │
│                              │   (AMASS Dataset)      │    │
│                              └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle:** The discriminator learns to distinguish reference motion (real) from policy rollouts (fake). The policy receives reward for fooling the discriminator.

---

## Current Implementation

### Architecture

**File:** `playground_amp/amp/discriminator.py`

```python
class AMPDiscriminator(nn.Module):
    """Current architecture."""
    hidden_dims: Sequence[int]  # Config: [256, 128]

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = x.astype(jnp.float32)

        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim,
                        kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)))(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        # Output: single logit (no sigmoid)
        logits = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=0.01))(x)
        return logits.squeeze(-1)
```

**Current Configuration (v0.4.0 Updated):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dims | `[512, 256]` | 2 layers (increased from [256, 128]) |
| Activation | ELU | Good for stability |
| Normalization | LayerNorm | After each hidden layer |
| Output | Raw logit | No sigmoid (LSGAN) |
| Init | Orthogonal | scale=√2 hidden, 0.01 output |

### Loss Function

**LSGAN Loss + R1 Regularizer (v0.4.1):**

v0.4.1 replaced WGAN-GP with R1 regularizer for consistency with LSGAN:

```python
def discriminator_loss(params, model, real_obs, fake_obs, rng_key,
                       r1_gamma=5.0, input_noise_std=0.0):
    # Add input noise (optional)
    if input_noise_std > 0.0:
        real_obs = real_obs + noise
        fake_obs = fake_obs + noise

    # Forward pass
    real_scores = model.apply(params, real_obs)  # Target: 1
    fake_scores = model.apply(params, fake_obs)  # Target: 0

    # LSGAN loss: (D(real) - 1)² + D(fake)²
    real_loss = jnp.mean((real_scores - 1.0) ** 2)
    fake_loss = jnp.mean(fake_scores ** 2)
    lsgan_loss = 0.5 * (real_loss + fake_loss)

    # R1 Regularizer (v0.4.1): gradient penalty on REAL samples only
    # More appropriate for LSGAN than WGAN-GP (which uses interpolated)
    def real_disc_sum(x):
        return jnp.sum(model.apply(params, x, training=True))
    grad_real = jax.grad(real_disc_sum)(real_obs)
    r1_penalty = jnp.mean(jnp.sum(grad_real ** 2, axis=-1))

    total_loss = lsgan_loss + (r1_gamma / 2.0) * r1_penalty
    return total_loss, metrics
```

**Why R1 over WGAN-GP (from second external review):**
- LSGAN uses bounded targets [0, 1]
- WGAN-GP was designed for Wasserstein critics (unbounded outputs)
- R1 only penalizes gradient on real samples (simpler, faster, more stable)
- Mixing LSGAN + WGAN-GP is theoretically inconsistent

**Current Hyperparameters (v0.4.1):**
| Parameter | Value | Notes |
|-----------|-------|-------|
| `disc_lr` | `1e-4` | Learning rate |
| `update_steps` | `3` | Updates per policy iteration |
| `r1_gamma` | `5.0` | R1 regularizer weight |
| `disc_input_noise_std` | `0.02` | Input blur |
| `batch_size` | `256` | Samples per update |

### Reward Computation

**File:** `playground_amp/amp/discriminator.py` (lines 196-230)

```python
def compute_amp_reward(params, model, obs):
    """Current reward formula."""
    scores = model.apply(params, obs, training=False)

    # Current formula: max(0, 1 - 0.25 * (D(s) - 1)²)
    amp_reward = jnp.maximum(0.0, 1.0 - 0.25 * (scores - 1.0) ** 2)

    return amp_reward
```

**Reward Mapping (v0.4.0 Fixed):**
| D(s) Score | Old (Buggy) | New (Clipped) | Interpretation |
|------------|-------------|---------------|----------------|
| 1.0 | 1.00 | 1.00 | Perfect (looks real) |
| 0.5 | 0.94 | 0.50 | Partial match |
| **0.0** | **0.75** ❌ | **0.00** ✅ | Fake gets zero reward |
| -1.0 | 0.00 | 0.00 | Minimum |

### Training Pipeline (v0.4.0 Fixed)

**File:** `playground_amp/training/trainer_jit.py`

```python
def train_iteration(state, env_state, ref_buffer_data):
    # Step 1: Collect rollout
    new_env_state, transitions, bootstrap_value = collect_rollout_scan(...)

    # Step 2: Extract AMP features
    amp_features = extract_amp_features_batched(transitions.obs, config)
    norm_amp_features = normalize_features(amp_features, state.feature_mean, state.feature_var)
    norm_ref_data = normalize_features(ref_buffer_data, state.feature_mean, state.feature_var)

    # Step 3: Compute AMP rewards FIRST (using OLD disc_params) ✅ FIXED
    amp_rewards = compute_amp_reward(state.disc_params, disc_model, norm_amp_features)
    #                                 ^^^^^^^^^^^^^^^^^^
    #                                 Uses OLD params (correct!)

    # Step 4: THEN train discriminator ✅ FIXED
    new_disc_params, new_disc_opt_state, disc_loss, disc_accuracy = train_discriminator_scan(
        disc_params=state.disc_params,
        ...
    )

    # Step 5: PPO update with shaped rewards
    shaped_rewards = transitions.reward + config.amp_reward_weight * amp_rewards
    ...
```

### Feature Extraction

**File:** `playground_amp/amp/amp_features.py`

**29-dim AMP Feature Vector:**
| Index | Feature | Dims | Notes |
|-------|---------|------|-------|
| 0-8 | Joint positions | 9 | All actuated joints |
| 9-17 | Joint velocities | 9 | All actuated joints |
| 18-20 | Root velocity direction | 3 | **Normalized to unit vector** |
| 21-23 | Root angular velocity | 3 | World frame |
| 24 | Root height | 1 | Via gravity[z] |
| 25-28 | Foot contacts | 4 | Currently zeros |

**Velocity Normalization (Correct):**
```python
# Speed threshold for stationary detection
min_speed_threshold = 0.1  # m/s

linvel_norm = jnp.linalg.norm(root_linvel, axis=-1, keepdims=True)
is_moving = linvel_norm > min_speed_threshold

root_linvel_dir = jnp.where(
    is_moving,
    root_linvel / (linvel_norm + 1e-8),  # Unit direction
    jnp.zeros_like(root_linvel)           # Zero if stationary
)
```

---

## Identified Issues

### Critical: Reward Formula Bug

**Severity:** 🔴 Critical
**Location:** `discriminator.py:228`

**Problem:**
The current reward formula gives excessively high rewards even when the discriminator correctly identifies motion as fake:

```python
# Current (WRONG)
amp_reward = jnp.maximum(0.0, 1.0 - 0.25 * (scores - 1.0) ** 2)
```

**Impact Analysis:**
- When D(s) = 0.0 (discriminator is 100% confident motion is fake)
- Reward = 1.0 - 0.25 × (0.0 - 1.0)² = 1.0 - 0.25 × 1.0 = **0.75**
- Policy receives **75% of maximum reward** for completely unnatural motion
- **Result:** Policy has almost no gradient signal to improve style

**Comparison with Original AMP Paper:**

| Formula | D=1.0 | D=0.5 | D=0.0 | D=-1.0 |
|---------|-------|-------|-------|--------|
| Current (buggy) | 1.00 | 0.94 | **0.75** | 0.00 |
| Original AMP | 1.00 | 0.69 | **0.00** | 0.00 |
| Clipped linear | 1.00 | 0.50 | **0.00** | 0.00 |

### Critical: Training Order Bug

**Severity:** 🔴 Critical
**Location:** `trainer_jit.py:656-684`

**Problem:**
The discriminator is trained BEFORE computing AMP rewards, but the NEW discriminator parameters are used for reward computation:

```python
# Current order (WRONG)
new_disc_params = train_discriminator(state.disc_params, ...)  # Step 3
amp_rewards = compute_amp_reward(new_disc_params, ...)         # Step 4 - uses NEW params!
```

**Impact Analysis:**
1. Discriminator updates to better detect current policy
2. D(policy) scores drop immediately
3. Same rollout gets lower AMP reward than it deserves
4. Reward signal becomes noisy and inconsistent

**Correct Order:**
```python
# Correct order
amp_rewards = compute_amp_reward(state.disc_params, ...)       # Use OLD params first
new_disc_params = train_discriminator(state.disc_params, ...)  # Then update
```

### Moderate: Network Capacity

**Severity:** ⚠️ Moderate
**Location:** `ppo_amass_training.yaml:93`

**Problem:**
Current network `[256, 128]` may be too small to learn the complex manifold of natural motion.

**Industry Comparison:**
| Implementation | Hidden Dims | Total Params |
|----------------|-------------|--------------|
| **Current** | `[256, 128]` | ~40K |
| DeepMind AMP | `[1024, 512]` | ~600K |
| NVIDIA IsaacGym | `[512, 256, 128]` | ~200K |
| Typical Robotics | `[512, 256]` | ~150K |

**Impact:**
- Discriminator may collapse to trivial solution
- Cannot represent complex decision boundaries
- Limited ability to distinguish subtle motion differences

### Minor: No Replay Buffer

**Severity:** ⚠️ Minor
**Location:** `trainer_jit.py` (architectural)

**Problem:**
Discriminator only trains on samples from the current rollout. No historical policy samples are retained.

**Impact:**
- Catastrophic forgetting of previous failure modes
- Discriminator may "forget" how to detect old bad behaviors
- Less stable training dynamics

**Industry Practice:**
Many implementations maintain a replay buffer of policy samples (typically last 10-50 rollouts) for discriminator training.

---

## Proposed Improvements

### Fix 1: Correct Reward Formula

**Priority:** P0 (Critical)

```python
# Option A: Clipped linear (simple, recommended)
def compute_amp_reward(params, model, obs):
    scores = model.apply(params, obs, training=False)
    amp_reward = jnp.clip(scores, 0.0, 1.0)
    return amp_reward

# Option B: Sigmoid mapping (smoother gradients)
def compute_amp_reward(params, model, obs):
    scores = model.apply(params, obs, training=False)
    amp_reward = jax.nn.sigmoid(scores)
    return amp_reward
```

**Expected Impact:**
| D(s) | Before | After (Clipped) |
|------|--------|-----------------|
| 1.0 | 1.00 | 1.00 |
| 0.5 | 0.94 | 0.50 |
| 0.0 | 0.75 | 0.00 |

### Fix 2: Correct Training Order

**Priority:** P0 (Critical)

```python
def train_iteration(state, env_state, ref_buffer_data):
    # ... collect rollout and extract features ...

    # Step 3: Compute AMP rewards FIRST (using OLD disc_params)
    amp_rewards = compute_amp_reward(
        state.disc_params,  # ← OLD parameters
        disc_model,
        norm_amp_features
    )

    # Step 4: THEN train discriminator
    new_disc_params, new_disc_opt_state, disc_loss, disc_accuracy = train_discriminator_scan(
        disc_params=state.disc_params,
        ...
    )

    # ... rest of PPO update ...
```

### Fix 3: Increase Network Capacity

**Priority:** P1 (Recommended)

```yaml
# ppo_amass_training.yaml
amp:
  discriminator_hidden: [512, 256, 128]  # 3 layers, more capacity
```

### Fix 4: Add Policy Replay Buffer (Optional)

**Priority:** P2 (Nice to have)

```python
class PolicyReplayBuffer:
    """Stores historical policy samples for discriminator training."""

    def __init__(self, max_size: int, feature_dim: int):
        self.buffer = jnp.zeros((max_size, feature_dim))
        self.ptr = 0
        self.size = 0

    def add(self, samples: jnp.ndarray):
        # Add new samples, overwriting oldest
        ...

    def sample(self, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        # Sample from buffer
        ...
```

---

## Implementation Plan

### Phase 1: Critical Bug Fixes (Immediate)

| Task | File | Lines | Effort |
|------|------|-------|--------|
| Fix reward formula | `discriminator.py` | 228 | 5 min |
| Fix training order | `trainer_jit.py` | 656-684 | 10 min |
| Update version | `ppo_amass_training.yaml` | 23-24 | 1 min |

### Phase 2: Improvements (After Validation)

| Task | File | Effort |
|------|------|--------|
| Increase network size | `ppo_amass_training.yaml` | 1 min |
| Add replay buffer | `trainer_jit.py` (new class) | 2 hours |

---

## Validation Criteria

### After Fix 1 & 2 (Critical Bugs)

Run training for 100 iterations and verify:

| Metric | Before (Broken) | Expected After |
|--------|-----------------|----------------|
| `disc_acc` @ iter 10 | 0.50 (stuck) | > 0.55 |
| `disc_acc` @ iter 50 | 0.50 (stuck) | 0.60 - 0.75 |
| `disc_acc` stability | Never changes | Oscillates healthily |
| `amp_reward` when D=0 | 0.75 | ~0.0 |
| `amp_reward` when D=1 | 1.0 | 1.0 |

### Long-term Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| `disc_acc` steady state | 0.55 - 0.80 | Not saturated |
| Episode length | > 200 steps | Robot survives |
| Forward velocity | > 0.5 m/s | Tracks command |
| Gait visual quality | Smooth, human-like | Subjective |

---

## References

1. **Peng et al. (2021)** - "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" - Original AMP paper
2. **NVIDIA IsaacGym** - Reference implementation with AMP
3. **DeepMind MuJoCo Locomotion** - Industry-standard architecture patterns
4. **WGAN-GP (Gulrajani et al., 2017)** - Gradient penalty formulation

---

## Appendix A: Implemented Fixes

### A.1 Reward Formula Fix ✅ (v0.4.0)

**File:** `playground_amp/amp/discriminator.py`

```python
def compute_amp_reward(params, model, obs):
    """Compute AMP reward for policy training (clipped linear formulation).

    v0.4.0 Fix: Changed from quadratic to clipped linear mapping.

    Old (buggy) formula: max(0, 1 - 0.25 * (D(s) - 1)²)
    - D(s) = 0.0 (fake) → reward = 0.75 (too high!)

    New (correct) formula: clip(D(s), 0, 1)
    - D(s) = 1.0 (looks like reference) → reward = 1.0
    - D(s) = 0.0 (looks fake) → reward = 0.0
    """
    scores = model.apply(params, obs, training=False)
    amp_reward = jnp.clip(scores, 0.0, 1.0)  # Clipped linear
    return amp_reward
```

### A.2 Training Order Fix ✅ (v0.4.0)

**File:** `playground_amp/training/trainer_jit.py`

The training order was corrected to:
1. **Step 3:** Compute AMP rewards FIRST (using OLD `state.disc_params`)
2. **Step 4:** THEN train discriminator (update params for next iteration)

This ensures the policy is rewarded based on the discriminator that was active when samples were collected.

### A.3 Network Size Increase ✅ (v0.4.0)

**File:** `playground_amp/configs/ppo_amass_training.yaml`

```yaml
amp:
  discriminator_hidden: [512, 256]  # Increased from [256, 128]
```

### A.4 R1 Regularizer ✅ (v0.4.1)

**Files:** `playground_amp/amp/discriminator.py`, `playground_amp/training/trainer_jit.py`

Replaced WGAN-GP with R1 regularizer for theoretical consistency with LSGAN:

```python
# R1 Regularization: gradient penalty on REAL samples only
def real_disc_sum(x):
    return jnp.sum(model.apply(params, x, training=True))

grad_real = jax.grad(real_disc_sum)(real_obs)
r1_penalty = jnp.mean(jnp.sum(grad_real ** 2, axis=-1))

total_loss = lsgan_loss + (r1_gamma / 2.0) * r1_penalty
```

**Why this matters:**
- WGAN-GP was designed for Wasserstein critics (unbounded outputs)
- LSGAN uses bounded targets [0, 1]
- R1 penalizes gradient on REAL samples only (not interpolated)
- More theoretically consistent and often more stable

### A.5 Distribution Metrics ✅ (v0.4.1)

**File:** `playground_amp/training/trainer_jit.py`

Added detailed discriminator distribution metrics for debugging:

```python
class IterationMetrics(NamedTuple):
    # ... existing metrics ...
    # v0.4.1: Distribution metrics for debugging
    disc_real_mean: jnp.ndarray  # Mean D(real) score - should → 1
    disc_fake_mean: jnp.ndarray  # Mean D(fake) score - should → 0
    disc_real_std: jnp.ndarray   # Std of D(real) scores
    disc_fake_std: jnp.ndarray   # Std of D(fake) scores
```

These metrics help diagnose discriminator behavior:
- `disc_real_mean` should approach 1.0 (real samples correctly identified)
- `disc_fake_mean` should approach 0.0 (fake samples correctly identified)
- If both are similar, discriminator is not learning separation

---

## Appendix B: Future Improvements (TODO)

### B.1 Spectral Normalization (P2 - v0.5.0)

Replace LayerNorm with Spectral Normalization for strict Lipschitz control:

```python
# Current:
x = nn.LayerNorm()(x)

# Proposed:
x = SpectralNorm(nn.Dense(hidden_dim))(x)
```

### B.2 Policy Replay Buffer (P2 - v0.5.0)

Store historical policy samples to prevent catastrophic forgetting:

```python
class PolicyReplayBuffer:
    def __init__(self, max_size: int, feature_dim: int):
        self.buffer = jnp.zeros((max_size, feature_dim))
        self.ptr = 0
```

### B.3 Temporal Context (P3 - v0.6.0)

Pass 2-3 frame window instead of single frame:

```python
# Current: 29-dim single frame
# Proposed: 87-dim (3 frames × 29 features)
amp_features = jnp.concatenate([obs_t_minus_2, obs_t_minus_1, obs_t], axis=-1)
```

This enables the discriminator to judge acceleration and jerk, which are hallmarks of natural motion.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial design review |
| 2.0 | 2025-12-23 | Added external review feedback, marked v0.4.0 fixes as implemented |
| 2.1 | 2025-12-23 | Added v0.4.1 changes: R1 regularizer, distribution metrics |

---

**Document End**
