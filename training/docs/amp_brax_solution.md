# AMP + Brax v2 Integration: Solution Architecture

**Date**: 2025-12-16
**Status**: Technical Design Complete
**Problem**: AMP cannot be cleanly injected into `brax.training.core.train()`
**Solution**: Reimplement PPO using Brax building blocks (Option A)

---

## Executive Summary

**Root Cause**: Brax v2's `ppo.train()` is a monolithic, fully-JIT compiled training loop with no hooks for:
- Accessing intermediate `(s_t, s_{t+1})` transitions during rollout
- Separate discriminator updates
- Reward shaping using discriminator logits

**Solution**: Own the training loop by using Brax's PPO building blocks directly.

---

## Problem Analysis

### Why Current Approach Fails

| Symptom | Root Cause |
|---------|------------|
| Cannot inject AMP during rollout | `ppo.train()` is a closed-form compiled program |
| Wrapper tracing errors | JAX forbids hidden Python-side state |
| Cannot collect intermediate obs | Rollout, GAE, and update are fused in single JIT |
| Discriminator can't train | No callback mechanism in Brax PPO |

### The Impedance Mismatch

**Brax's Design Philosophy** (Pure Functional JAX):
```
env.step       → pure function (no side effects)
policy.apply   → pure function
ppo.train      → closed-form compiled program
```

**AMP Requirements** (Stateful + Interleaved):
```
1. Access (s_t, s_{t+1}) during rollout
2. Separate discriminator updates (own optimizer)
3. Reward shaping using discriminator logits
4. Replay buffer for reference motion
```

**Conclusion**: AMP needs an *open* training loop, but `ppo.train()` is *closed*.

---

## Recommended Solution: Option A (Custom PPO Loop)

### Approach

Reimplement PPO using Brax's building blocks, but write our own outer loop.

### Files to Reuse from Brax

```
brax/training/ppo.py          # Reference implementation
brax/training/acting.py       # Rollout utilities
brax/training/losses.py       # PPO loss function (compute_gae, ppo_loss)
brax/training/types.py        # Transition dataclasses
```

Components to extract:
- ✅ `compute_gae` (Generalized Advantage Estimation)
- ✅ `ppo_loss` (Clipped surrogate objective)
- ✅ `Transition` dataclass
- ❌ `train()` function (replace with custom loop)

---

## Implementation Plan

### Step 1: Define AMP Transition Dataclass

```python
@dataclass
class AMPTransition:
    """Transition with AMP features for discriminator training."""
    obs: jnp.ndarray           # Current observation
    next_obs: jnp.ndarray      # Next observation
    action: jnp.ndarray        # Action taken
    reward: jnp.ndarray        # Environment reward
    done: jnp.ndarray          # Episode termination
    amp_feat: jnp.ndarray      # Motion features for discriminator
    log_prob: jnp.ndarray      # Log probability of action
    value: jnp.ndarray         # Value function estimate
```

### Step 2: Custom Rollout with AMP Features

```python
def rollout_step(carry, _):
    """Single rollout step that collects AMP features."""
    state, rng = carry
    rng, action_rng, step_rng = jax.random.split(rng, 3)

    # Policy forward pass
    action, log_prob = policy.apply(params, state.obs, action_rng)
    value = value_fn.apply(value_params, state.obs)

    # Environment step
    next_state = env.step(state, action)

    # Extract AMP features (joint angles, velocities, etc.)
    amp_feat = extract_amp_features(state.obs, next_state.obs)

    transition = AMPTransition(
        obs=state.obs,
        next_obs=next_state.obs,
        action=action,
        reward=next_state.reward,
        done=next_state.done,
        amp_feat=amp_feat,
        log_prob=log_prob,
        value=value,
    )

    return (next_state, rng), transition

# Collect full rollout
(final_state, _), transitions = jax.lax.scan(
    rollout_step,
    (init_state, rng),
    None,
    length=num_steps
)
```

### Step 3: Discriminator Training (Separate from PPO)

```python
def disc_loss_fn(disc_params, agent_feat, expert_feat, rng):
    """Discriminator loss with gradient penalty."""
    # Forward pass
    agent_logits = disc.apply(disc_params, agent_feat)
    expert_logits = disc.apply(disc_params, expert_feat)

    # Binary cross-entropy
    bce_loss = (
        optax.sigmoid_binary_cross_entropy(agent_logits, jnp.zeros_like(agent_logits)).mean() +
        optax.sigmoid_binary_cross_entropy(expert_logits, jnp.ones_like(expert_logits)).mean()
    )

    # Gradient penalty (WGAN-GP)
    alpha = jax.random.uniform(rng, (agent_feat.shape[0], 1))
    interpolated = alpha * expert_feat + (1 - alpha) * agent_feat
    grad_fn = jax.grad(lambda x: disc.apply(disc_params, x).sum())
    grads = jax.vmap(grad_fn)(interpolated)
    grad_norm = jnp.sqrt((grads ** 2).sum(axis=-1) + 1e-8)
    gp_loss = ((grad_norm - 1) ** 2).mean()

    return bce_loss + 5.0 * gp_loss

# Update discriminator
grads = jax.grad(disc_loss_fn)(disc_params, agent_feat, expert_feat, rng)
disc_updates, disc_opt_state = disc_optimizer.update(grads, disc_opt_state)
disc_params = optax.apply_updates(disc_params, disc_updates)
```

### Step 4: AMP Reward Shaping (Pure JAX)

```python
def compute_amp_reward(disc_params, amp_feat):
    """Compute AMP reward from discriminator output."""
    logits = disc.apply(disc_params, amp_feat)
    probs = jax.nn.sigmoid(logits)
    amp_reward = jnp.log(probs + 1e-8)  # Log probability of being "real"
    return amp_reward

# Shape rewards before PPO update
shaped_reward = transitions.reward + amp_weight * compute_amp_reward(disc_params, transitions.amp_feat)
```

### Step 5: PPO Update (Using Brax Loss)

```python
def ppo_update(policy_params, value_params, transitions, shaped_rewards):
    """PPO update using Brax's loss function."""
    # Compute advantages with shaped rewards
    advantages = compute_gae(
        rewards=shaped_rewards,
        values=transitions.value,
        dones=transitions.done,
        gamma=0.99,
        gae_lambda=0.95
    )

    # PPO loss
    def loss_fn(params):
        return ppo_loss(
            policy_params=params,
            observations=transitions.obs,
            actions=transitions.action,
            old_log_probs=transitions.log_prob,
            advantages=advantages,
            clip_epsilon=0.2
        )

    # Gradient update
    grads = jax.grad(loss_fn)(policy_params)
    policy_updates, policy_opt_state = policy_optimizer.update(grads, policy_opt_state)
    policy_params = optax.apply_updates(policy_params, policy_updates)

    return policy_params

```

### Step 6: Complete Training Loop

```python
def train_amp_ppo(
    env,
    policy,
    value_fn,
    discriminator,
    ref_buffer,
    num_iterations=3000,
    num_steps=10,
    num_envs=4096,
    amp_weight=1.0,
):
    """Custom PPO+AMP training loop."""

    # Initialize
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    for iteration in range(num_iterations):
        # 1. Collect rollout with AMP features
        (state, rng), transitions = jax.lax.scan(
            rollout_step, (state, rng), None, length=num_steps
        )

        # 2. Train discriminator (separate from PPO)
        agent_feat = transitions.amp_feat.reshape(-1, amp_feat_dim)
        expert_feat = ref_buffer.sample_single_frames(agent_feat.shape[0])

        for _ in range(disc_updates_per_iter):
            disc_params = update_discriminator(disc_params, agent_feat, expert_feat)

        # 3. Compute AMP-shaped rewards
        amp_rewards = compute_amp_reward(disc_params, transitions.amp_feat)
        shaped_rewards = transitions.reward + amp_weight * amp_rewards

        # 4. PPO update with shaped rewards
        policy_params, value_params = ppo_update(
            policy_params, value_params, transitions, shaped_rewards
        )

        # 5. Logging
        log_metrics(iteration, transitions, amp_rewards, disc_params)

    return policy_params, value_params, disc_params
```

---

## Why This Solves All Issues

| Problem | Solution |
|---------|----------|
| JIT black box | We control the outer loop |
| Cannot inject AMP logic | AMP runs between rollout & PPO update |
| No intermediate obs | We collect them in `AMPTransition` |
| Wrapper tracing errors | No Python-side wrappers, pure JAX tensors |
| Discriminator can't train | Separate update step with own optimizer |

---

## AMP Features for Humanoid Walking

Based on DeepMind AMP paper and 2024 best practices:

```python
def extract_amp_features(obs, next_obs):
    """Extract motion features for discriminator."""
    # From current observation (44-dim):
    # - Joint positions (indices 0-10): 11 values
    # - Joint velocities (indices 11-21): 11 values
    # - Base height (index 38 or obs[-6]): 1 value
    # - Base orientation (indices 22-24): 3 values (roll, pitch, yaw)

    joint_pos = obs[..., 0:11]
    joint_vel = obs[..., 11:22]
    base_height = obs[..., 38:39]
    base_orient = obs[..., 22:25]

    # Concatenate features
    amp_feat = jnp.concatenate([
        joint_pos,      # 11
        joint_vel,      # 11
        base_height,    # 1
        base_orient,    # 3
    ], axis=-1)  # Total: 26 features

    return amp_feat
```

---

## Alternative Options

### Option B: Fork Brax PPO (Acceptable, Brittle)

- Copy `brax/training/agents/ppo/train.py`
- Modify rollout function to collect AMP features
- Modify reward computation to add AMP reward
- **Downsides**: Large diff, breaks on Brax updates

### Option C: Wrappers/Side Channels (NOT Recommended)

- Attempt to inject state via environment wrapper
- **Will never be stable**: JAX forbids hidden state, tracing breaks

---

## Implementation Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Create `amp_ppo_training.py` scaffold | 2h |
| 2 | Port Brax building blocks (GAE, PPO loss) | 2h |
| 3 | Implement `AMPTransition` and rollout | 2h |
| 4 | Implement discriminator training step | 1h |
| 5 | Wire up complete training loop | 2h |
| 6 | Integration testing (100 iters smoke) | 1h |
| 7 | Full training run (3000 iters) | 8h |
| **Total** | | **~18h** |

---

## Files to Create

```
training/
├── amp/
│   ├── discriminator.py      # ✅ EXISTS (Phase 1 complete)
│   ├── ref_buffer.py         # ✅ EXISTS (Phase 1 complete)
│   └── amp_features.py       # NEW: Feature extraction
├── training/
│   ├── amp_ppo_training.py   # NEW: Custom training loop
│   ├── ppo_building_blocks.py # NEW: Ported Brax components
│   └── transitions.py        # NEW: AMPTransition dataclass
└── train.py              # NEW: Entry point
```

---

## References

- **Brax PPO Source**: `brax/training/agents/ppo/train.py`
- **Brax Losses**: `brax/training/agents/ppo/losses.py`
- **AMP Paper**: "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" (2021)
- **ETH Deep Whole-Body Control**: 2024 best practices for humanoid AMP

---

## Summary

**The answer is YES** - AMP can be implemented cleanly in Brax, just not on top of `ppo.train()`.

By moving one layer down to Brax's building blocks:
- Rollout, GAE, and PPO loss remain battle-tested
- AMP integrates naturally between rollout and update
- Full JIT compilation preserved
- No Python-side state or tracing issues

**Next Step**: Create `training/core/amp_ppo_training.py` using this architecture.
