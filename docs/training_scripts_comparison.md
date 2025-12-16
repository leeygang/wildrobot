# Training Scripts - Historical Comparison & Consolidation

**Status**: CONSOLIDATED (2025-12-16)
**Current**: Single unified `train.py` script
**Previous**: 4 separate training scripts (now removed)

---

## Current State

The `playground_amp` directory now has **ONE unified training script**:

### ✅ `train.py` - Unified Brax PPO + AMP Trainer

**Status**: Production-ready, actively maintained

**Features**:
- ✅ Brax PPO trainer (battle-tested, GPU-optimized)
- ✅ AMP discriminator support (Task 5 ready)
- ✅ Pure-JAX backend (full JIT/vmap compatibility)
- ✅ Config system (YAML + CLI overrides)
- ✅ Quick-verify mode for smoke testing
- ✅ WandB logging and checkpointing
- ✅ Progress callbacks with detailed metrics

**Usage**: See `playground_amp/TRAINING_README.md`

---

## Historical Context (Pre-Consolidation)

The `playground_amp` directory previously contained four training scripts with different purposes and maturity levels:

## 1. `train_jax.py` - Minimal JAX/Optax Stub ⚠️ [REMOVED]

**Purpose:** Minimal proof-of-concept demonstrating JAX/Optax integration

**Status:** Development stub - NOT for actual training [DELETED 2025-12-16]

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

## 2. `train_brax.py` - Brax Integration Wrapper 🔄 [REMOVED]

**Purpose:** Wire WildRobotEnv into Brax's training ecosystem

**Status:** Experimental integration - requires Brax v2 compatibility [DELETED 2025-12-16, superseded by train.py]

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

## 3. `train_brax_ppo.py` - Brax PPO Integration ✅ [REMOVED]

**Purpose:** Brax's battle-tested PPO trainer integrated with WildRobotEnv

**Status:** Production-ready, Task 10 validated [DELETED 2025-12-16, merged into train.py]

**Key achievements**:
- Validated Brax PPO integration (Task 10 complete)
- Smoke test passed (4.1s, 10 iterations, 4 envs)
- GPU-optimized, JIT-compiled training
- Progress callbacks and checkpointing

**Removal rationale**: Merged with train.py to create single unified script

---

## 4. `train.py` - Original Self-Contained PPO+AMP Trainer [REPLACED]

**Purpose:** Full-featured, self-contained training script for Phase 3 development

**Status:** Replaced with unified version [2025-12-16]

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

---

## Consolidation Decision (2025-12-16)

### Why Consolidate?

**Problems with 4 separate scripts**:
- ❌ Maintenance burden (4 codebases to keep in sync)
- ❌ Confusion (which script to use when?)
- ❌ Duplication (both train.py and train_brax_ppo.py implemented PPO)
- ❌ Against DRY principle

**Solution**: Merge best features into single `train.py`
- ✅ Brax PPO trainer (from train_brax_ppo.py) - battle-tested
- ✅ AMP support (from original train.py) - Task 5 ready
- ✅ Config system (best of both)
- ✅ Single source of truth

### What Was Merged

**From `train_brax_ppo.py`**:
- Brax PPO trainer integration
- Progress callbacks
- Checkpoint saving
- WandB logging

**From original `train.py`**:
- AMP discriminator hooks
- Reference motion buffer
- Config override system
- Quick-verify mode

**Result**: Unified `train.py` with all features

---

## New Recommendation (Post-Consolidation)

### For All Training Tasks:

**Use `train.py`** - the unified training script:

```bash
# Quick verification
python playground_amp/train.py --verify

# CPU training (development)
python playground_amp/train.py --num-iterations 100 --num-envs 16

# GPU training (production)
python playground_amp/train.py --num-iterations 3000 --num-envs 2048

# Enable AMP for natural motion
python playground_amp/train.py --enable-amp --amp-weight 1.0
```

**See**: `playground_amp/TRAINING_README.md` for complete documentation

---

## Summary Table (Historical vs Current)

### Before Consolidation (4 scripts)

| Feature | `train_jax.py` | `train_brax.py` | `train_brax_ppo.py` | `train.py` (old) |
|---------|----------------|-----------------|---------------------|------------------|
| **Purpose** | Library test | Brax exploration | Brax PPO (Task 10) | Self-contained |
| **Status** | Dev stub | Experimental | Validated | AMP-ready |
| **PPO** | ❌ None | 🔄 Basic | ✅ Brax PPO | ✅ Custom PPO |
| **AMP Support** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Lines of Code** | 39 | 97 | 259 | 549 |

### After Consolidation (1 script)

| Feature | `train.py` (unified) |
|---------|----------------------|
| **Purpose** | Unified training script |
| **Status** | ✅ Production-ready |
| **PPO Implementation** | ✅ Brax PPO (battle-tested) |
| **AMP Support** | ✅ Yes (Task 5 ready) |
| **Pure-JAX Backend** | ✅ Yes (full JIT/vmap) |
| **Config System** | ✅ YAML + CLI overrides |
| **Quick-Verify Mode** | ✅ Yes |
| **WandB Logging** | ✅ Yes |
| **Checkpointing** | ✅ Yes |
| **Lines of Code** | ~450 |
| **Recommended For** | All training tasks |

---

## Why Consolidate Now?

**Historical**: During development, multiple approaches were explored:
1. **`train_jax.py`**: JAX/Optax exploration (Task 7 prep)
2. **`train_brax.py`**: Early Brax integration experiments
3. **`train_brax_ppo.py`**: Validated Brax PPO integration (Task 10)
4. **`train.py`**: Self-contained PPO + AMP development

**Decision Point**: After Task 10 validated Brax PPO, the path forward became clear:
- Brax PPO works ✅
- AMP integration is needed (Task 5) ✅
- Multiple scripts create confusion ❌

**Action**: Merge best features into single unified `train.py`

**Result**:
- Single source of truth
- Best of both worlds (Brax + AMP)
- Easier maintenance
- Clear upgrade path

---

## Benefits of Consolidation

### For Users
- ✅ No confusion about which script to use
- ✅ Single command interface for all training modes
- ✅ Consistent configuration system
- ✅ All features in one place

### For Developers
- ✅ Single codebase to maintain
- ✅ Easier to add new features
- ✅ Centralized bug fixes
- ✅ Clear testing path

### For Project
- ✅ Aligns with Phase 3 plan: "MuJoCo MJX Physics + Brax v2 Training"
- ✅ Task 10 validated: Brax integration works
- ✅ Task 5 ready: AMP hooks in place
- ✅ Clean foundation for remaining tasks

---

## Files Changed

**Deleted** (2025-12-16):
- `playground_amp/train_jax.py` (test stub)
- `playground_amp/train_brax.py` (early exploration)
- `playground_amp/train_brax_ppo.py` (validated, merged)

**Updated**:
- `playground_amp/train.py` (unified version)

**Created**:
- `playground_amp/TRAINING_README.md` (comprehensive guide)

**Documentation**:
- This file updated to reflect consolidation
