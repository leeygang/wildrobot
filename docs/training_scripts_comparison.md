# Training Scripts Comparison

## Overview

The `playground_amp` directory contains three training scripts with different purposes and maturity levels:

## 1. `train_jax.py` - Minimal JAX/Optax Stub ⚠️

**Purpose:** Minimal proof-of-concept demonstrating JAX/Optax integration

**Status:** Development stub - NOT for actual training

**What it does:**
- Creates dummy parameters (10x10 weight matrix)
- Demonstrates optax SGD optimizer usage
- Runs a single parameter update step
- Prints updated parameter mean

**Use case:** Testing that JAX/Optax libraries are installed and working

**Example output:**
```
Updated param w mean: -1.0000005204346962e-05
```

**Code snippet:**
```python
def run_smoke():
    params = make_dummy_params()  # {"w": zeros(10,10), "b": zeros(10)}
    grads = {"w": ones(10,10) * 0.01, "b": ones(10) * 0.01}
    new_params, _ = simple_update_step(params, grads)
    print("Updated param w mean:", float(jnp.mean(new_params["w"])))
```

**When to use:** Never for actual training - only for library verification

---

## 2. `train_brax.py` - Brax Integration Wrapper 🔄

**Purpose:** Wire WildRobotEnv into Brax's training ecosystem

**Status:** Experimental integration - requires Brax v2 compatibility

**What it does:**
- Creates Brax-compatible environment wrapper
- Attempts to use `brax.training.ppo.train()` if available
- Falls back to minimal smoke-run if Brax PPO unavailable
- Demonstrates how to integrate custom environments with Brax

**Key features:**
- Uses `brax_wrapper.py` to adapt WildRobotEnv
- Configured via YAML config files
- Integrates with Brax's training infrastructure

**Dependencies:**
- Brax v2 library
- Brax PPO trainer
- Brax-compatible environment interface

**Limitations:**
- Requires Brax environment interface compatibility
- PPO integration is project-specific
- May need adaptation for different Brax versions

**Usage:**
```bash
uv run python -m playground_amp.train_brax \
    --config playground_amp/configs/wildrobot_phase3_training.yaml \
    --verify
```

**When to use:**
- If you want to leverage Brax's mature PPO implementation
- For distributed training on TPU/GPU with Brax infrastructure
- When you need Brax's advanced training features (checkpointing, logging, etc.)

---

## 3. `train.py` - Self-Contained PPO Trainer ✅ (RECOMMENDED)

**Purpose:** Full-featured, self-contained training script for Phase 3 development

**Status:** Production-ready for Phase 3 smoke training and development

**What it does:**
- **Complete PPO training loop** with proper clipping and minibatching
- **Flax/Optax integration** when available (with fallback to manual updates)
- **AMP (Adversarial Motion Priors) support** via `--enable-amp` flag
- **Configurable via YAML** with CLI overrides
- **Quick-verify mode** for smoke testing without full training
- **Checkpoint saving** every 5 updates
- **Reference motion buffer** for AMP discriminator training

**Key features:**

### Policy & Value Networks
- MLP policy with Gaussian distribution (mean + learned log_std)
- MLP value network for advantage estimation
- Default architecture: [obs_dim, 64, 64, act_dim/1]

### Training Configuration
```python
@dataclass
class TrainerConfig:
    num_envs: int = 8           # Vectorized environments
    rollout_steps: int = 64     # Steps per rollout
    total_updates: int = 20     # Number of training updates
    lr: float = 1e-3            # Learning rate
    gamma: float = 0.99         # Discount factor
    seed: int = 0               # Random seed
```

### PPO Implementation
- Proper advantage estimation with GAE
- PPO clipping (configurable clip ratio)
- Minibatch training with multiple epochs
- Value function loss with MSE
- Policy gradient with log probability ratios

### AMP Integration
- Optional discriminator network for style rewards
- Reference motion buffer for collecting expert demonstrations
- Configurable AMP reward weighting
- Automatic fallback if AMP unavailable

### Configuration System
- Primary config: `playground_amp/configs/wildrobot_phase3_training.yaml`
- Environment config: `playground_amp/configs/wildrobot_env.yaml`
- CLI overrides: `--set trainer.num_envs=512`
- Quick-verify mode: `--verify` (applies `quick_verify` overrides from config)

**Usage examples:**

Quick verification (minimal test):
```bash
PYTHONPATH=/Users/ygli/projects/wildrobot uv run --with numpy --with jax --with mujoco \
    python playground_amp/train.py --verify \
    --config playground_amp/configs/wildrobot_phase3_training.yaml
```

Smoke training (100 updates, 16 envs):
```bash
PYTHONPATH=/Users/ygli/projects/wildrobot uv run --with numpy --with jax --with mujoco \
    python playground_amp/train.py \
    --config playground_amp/configs/wildrobot_phase3_training.yaml \
    --set trainer.total_updates=100 \
    --set trainer.num_envs=16
```

Full training with AMP:
```bash
PYTHONPATH=/Users/ygli/projects/wildrobot uv run --with numpy --with jax --with mujoco \
    python playground_amp/train.py \
    --config playground_amp/configs/wildrobot_phase3_training.yaml \
    --enable-amp
```

**Pros:**
- ✅ Self-contained - no external training library dependencies
- ✅ Direct WildRobotEnv integration
- ✅ Flax/Optax support when available
- ✅ Proper PPO implementation (clipping, minibatches, epochs)
- ✅ AMP hooks ready for Phase 3 Task 5
- ✅ Quick-verify mode for smoke testing
- ✅ Config system with CLI overrides
- ✅ Checkpoint saving

**Cons:**
- ⚠️ Less mature than Brax's battle-tested PPO
- ⚠️ No distributed training support
- ⚠️ Basic logging (no WandB/TensorBoard hooks yet)

**When to use:**
- ✅ **Phase 3 development and smoke training** (current use case)
- ✅ When you need direct environment control
- ✅ For rapid iteration during debugging
- ✅ When you want to avoid Brax compatibility issues
- ✅ For AMP integration testing

---

## Recommendation for Phase 3

### For Current Task 9.3 (Smoke Training):

**Use `train.py`** with `--verify` flag for quick validation:

```bash
PYTHONPATH=/Users/ygli/projects/wildrobot uv run --with numpy --with jax --with mujoco \
    python playground_amp/train.py --verify \
    --config playground_amp/configs/wildrobot_phase3_training.yaml
```

This runs a minimal forward pass without optimization to verify:
- Environment resets correctly
- Observations are valid
- Rewards are computed
- No crashes or terminations

For full smoke training (100 updates):

```bash
PYTHONPATH=/Users/ygli/projects/wildrobot uv run --with numpy --with jax --with mujoco \
    python playground_amp/train.py \
    --config playground_amp/configs/wildrobot_phase3_training.yaml \
    --set trainer.total_updates=100 \
    --set trainer.num_envs=16
```

### Migration Path

1. **Phase 3 (Current):** Use `train.py` for development and smoke testing
2. **Phase 4 (Optional):** Migrate to `train_brax.py` for production-scale training if needed
3. **Future:** Consider integrating with other training frameworks (CleanRL, RLlib, etc.)

---

## Summary Table

| Feature | `train_jax.py` | `train_brax.py` | `train.py` |
|---------|----------------|-----------------|------------|
| **Purpose** | Library test stub | Brax integration | Self-contained trainer |
| **Status** | Dev stub | Experimental | Production-ready |
| **PPO Implementation** | ❌ None | ✅ Brax PPO | ✅ Custom PPO |
| **WildRobotEnv Direct** | ❌ No env | 🔄 Via wrapper | ✅ Direct |
| **AMP Support** | ❌ No | ❌ No | ✅ Yes |
| **Flax/Optax** | ✅ Demo only | 🔄 Via Brax | ✅ Optional |
| **Minibatch Training** | ❌ No | ✅ Via Brax | ✅ Yes |
| **Quick-Verify Mode** | ❌ No | 🔄 Basic | ✅ Full |
| **Config System** | ❌ No | ✅ YAML | ✅ YAML + CLI |
| **Checkpointing** | ❌ No | ✅ Via Brax | ✅ Yes |
| **Lines of Code** | 39 | 97 | 549 |
| **Recommended For** | Testing libs | Production scale | Phase 3 dev |

---

## Why Three Scripts?

1. **`train_jax.py`**: Proof-of-concept during JAX/Optax exploration (Task 7 prep)
2. **`train_brax.py`**: Exploration of Brax integration for production training
3. **`train.py`**: Main development trainer with full control for Phase 3 tasks

This is common in ML projects during development - multiple approaches are explored, and the best one emerges based on project needs. For Phase 3, `train.py` is the clear winner for development and smoke testing.
