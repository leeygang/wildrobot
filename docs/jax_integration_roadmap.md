# JAX Integration Roadmap for train.py

## Current State: What JAX Features Are Already Integrated? ✅

### train.py (Custom PPO) - Partial JAX Integration

**Already using JAX:**
```python
import jax
import jax.numpy as jnp
import optax  # JAX optimizer library
import flax.linen as nn  # JAX neural network library
```

**JAX components currently in train.py:**

1. ✅ **Neural Networks (Flax)**
   - Policy network: `flax.linen.nn.Dense` layers
   - Value network: `flax.linen.nn.Dense` layers
   - Parameters stored as JAX pytrees

2. ✅ **Gradient Computation (JAX)**
   - `jax.grad()` for policy gradients
   - `jax.grad()` for value function gradients
   - Automatic differentiation via JAX

3. ✅ **Optimizer Updates (Optax)**
   - `optax.sgd()` or `optax.adam()` when available
   - `optax.apply_updates()` for parameter updates
   - Pytree-aware gradient updates

4. ✅ **Random Number Generation (JAX)**
   - `jax.random.PRNGKey()` for seeding
   - `jax.random.split()` for RNG management
   - `jax.random.normal()` for action sampling

5. ✅ **Array Operations (JAX)**
   - Policy forward pass uses `jnp.array`
   - Value function uses `jnp.array`
   - Loss computation uses JAX operations

**NOT using JAX (yet):**

1. ❌ **Environment Stepping**
   - `env.step()` returns numpy arrays, not JAX arrays
   - Rollout collection uses numpy operations
   - No JIT compilation of environment stepping

2. ❌ **Training Loop JIT Compilation**
   - Main training loop is not `@jax.jit` decorated
   - Rollout collection is not JIT-compiled
   - PPO update loop is not fully JIT-compiled

3. ❌ **Vectorized Environment Operations**
   - No `jax.vmap` for parallel environment stepping
   - Manual numpy-based batching

4. ❌ **Full Pytree State Management**
   - Environment state uses numpy, not JAX pytrees
   - Mixing numpy and JAX arrays (conversion overhead)

---

## JAX Integration Levels

### Level 1: Basic JAX (✅ CURRENT - train.py)
- JAX for neural networks (Flax)
- JAX for gradients (jax.grad)
- JAX for optimizers (Optax)
- Numpy for environment stepping
- No JIT compilation of training loop

**Performance:** Good for CPU, ~1000 env steps/sec on 16 envs

### Level 2: Brax PPO Integration (🎯 PROPOSED - Task 1b)
- Full Brax PPO trainer (battle-tested)
- JIT-compiled PPO updates
- Automatic pytree state management
- Still using numpy WildRobotEnv (wrapped)

**Performance:** Better GPU utilization, ~5000+ env steps/sec

### Level 3: Pure-JAX Environment (⏸️ Task 1a - 70% complete)
- WildRobotEnv fully ported to JAX
- `jax.jit` compiled stepping
- `jax.vmap` for vectorization
- Pure JAX pytrees (no numpy conversion)

**Performance:** Optimal GPU utilization, 10,000+ env steps/sec

### Level 4: Full JAX Stack (🎯 END GOAL - Task 7)
- Pure-JAX environment (Task 1a)
- Brax PPO trainer (Task 1b)
- JIT-compiled entire training loop
- Distributed training support

**Performance:** Production-ready, 50,000+ env steps/sec on GPU

---

## Timeline & Integration Path

### ✅ Phase 1 (COMPLETED): Basic JAX in train.py
**Status:** Done
**What:** Neural networks, gradients, optimizers use JAX
**Files:** `train.py` lines 21-335
**Performance:** ~1000 steps/sec on 16 envs

### 🔄 Phase 2 (IN PROGRESS): Pure-JAX Environment Port
**Task:** Task 1a (70% complete)
**Timeline:** 1-2 days remaining
**What:**
- Complete `jax_full_port.py` validation
- Switch WildRobotEnv to use JAX-only stepping
- Enable JIT compilation of environment

**Files to complete:**
- `/Users/ygli/projects/wildrobot/playground_amp/envs/jax_full_port.py` (validation)
- `/Users/ygli/projects/wildrobot/playground_amp/envs/wildrobot_env.py` (enable JAX-only mode)

**Impact on train.py:**
```python
# Current (numpy-based):
obs, rew, done, info = env.step(acts)  # Returns numpy arrays

# After Task 1a (JAX-based):
obs, rew, done, info = env.step(acts)  # Returns JAX arrays (can be JIT-compiled)
```

### 🎯 Phase 3 (PROPOSED): Brax PPO Integration
**Task:** Task 1b (NEW)
**Timeline:** 2-4 days
**What:**
- Complete `BraxWildRobotWrapper` implementing `brax.envs.Env` interface
- Replace custom PPO code with `brax.training.agents.ppo.train()`
- Integrate with existing config system

**Changes to train.py:**
```python
# Current (549 lines of custom PPO):
for update in range(cfg.total_updates):
    # Rollout collection (manual)
    # Advantage computation (manual)
    # Minibatch training (manual)
    # Gradient updates (optax)

# After Brax integration (~50 lines):
from brax.training.agents.ppo import train as ppo
inference_fn, params, metrics = ppo.train(
    environment=BraxWildRobotWrapper(env_cfg),
    num_timesteps=1_000_000,
    episode_length=1000,
    num_envs=4096,
    learning_rate=3e-4,
)
```

### ⏸️ Phase 4 (BLOCKED): Optax/Flax Pytrees
**Task:** Task 7
**Timeline:** After Task 1a + 1b complete
**What:** Full pytree state management throughout stack

**Why blocked:** Brax integration (Task 1b) provides this automatically

---

## Performance Comparison

| Configuration | Env Type | Trainer | Env Steps/Sec | GPU Util | JIT Status |
|--------------|----------|---------|---------------|----------|------------|
| Current (train.py) | Numpy | Custom PPO | ~1,000 | 10-20% | Partial |
| + Task 1a | Pure-JAX | Custom PPO | ~5,000 | 40-60% | Env only |
| + Task 1b | Numpy | Brax PPO | ~8,000 | 60-80% | Training only |
| + Both (Goal) | Pure-JAX | Brax PPO | ~50,000+ | 90%+ | Full stack |

---

## Recommendation: Integration Order

### Option A: Brax First, Then Pure-JAX (RECOMMENDED)
```
Current → Task 1b (Brax) → Task 1a (Pure-JAX) → Complete
Timeline: 2-4 days + 1-2 days = 3-6 days total
```

**Pros:**
- ✅ Get battle-tested PPO immediately
- ✅ Better training quality sooner
- ✅ Pure-JAX port can be validated with Brax trainer
- ✅ Aligns with Phase 3 plan ("Brax v2 Training")

**Cons:**
- ⚠️ Initial performance still limited by numpy env

### Option B: Pure-JAX First, Then Brax
```
Current → Task 1a (Pure-JAX) → Task 1b (Brax) → Complete
Timeline: 1-2 days + 2-4 days = 3-6 days total
```

**Pros:**
- ✅ Performance gains sooner
- ✅ Task 1a already 70% complete

**Cons:**
- ⚠️ Still using custom PPO temporarily
- ⚠️ More validation needed before Brax integration

### Option C: Parallel Development
```
Current → [Task 1a || Task 1b] → Integrate both → Complete
Timeline: max(1-2 days, 2-4 days) = 2-4 days
```

**Pros:**
- ✅ Fastest path to full JAX stack
- ✅ Can work on both independently

**Cons:**
- ⚠️ Integration complexity at the end
- ⚠️ Harder to debug issues

---

## Answer: "When will we integrate JAX in train.py?"

**Short answer:** **JAX is already integrated!** But there are different levels:

1. **Already done ✅**: Neural networks, gradients, optimizers (current `train.py`)
2. **Task 1a (70% done)**: Pure-JAX environment stepping
3. **Task 1b (proposed)**: Brax PPO trainer (replaces custom PPO)
4. **Task 7 (future)**: Full pytree state management

**Timeline for full JAX integration:**
- **Minimal (Brax PPO):** 2-4 days (Task 1b)
- **Optimal (Brax + Pure-JAX):** 3-6 days (Task 1b + Task 1a)
- **Complete (Full stack):** Automatic with Brax integration

**My recommendation:** Start Task 1b (Brax integration) immediately. This gives you:
- Battle-tested PPO in ~2-4 days
- Better training infrastructure
- Foundation for completing Task 1a
- Aligns with original Phase 3 plan

---

## Code Example: JAX Integration Levels

### Current train.py (Partial JAX):
```python
# Neural networks: JAX ✅
mean = policy_model.apply(params['policy'], jnp.array(obs))

# Gradients: JAX ✅
grads = jax.grad(total_loss)(params)

# Optimizer: JAX ✅
updates, opt_state = tx.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)

# Environment: Numpy ❌
obs, rew, done, info = env.step(acts)  # Returns numpy arrays

# Training loop: Not JIT-compiled ❌
for update in range(cfg.total_updates):
    # Manual rollout collection
    # Manual advantage computation
```

### After Brax Integration (Level 2):
```python
# Everything JAX, JIT-compiled training loop
from brax.training.agents.ppo import train as ppo

inference_fn, params, metrics = ppo.train(
    environment=BraxWildRobotWrapper(env_cfg),
    num_timesteps=1_000_000,
    # Brax handles: networks, gradients, optimizers, training loop
    # All JIT-compiled internally
)
```

### After Pure-JAX Env (Level 3):
```python
# Same as Brax, but now env.step() is also JIT-compiled
inference_fn, params, metrics = ppo.train(
    environment=BraxWildRobotWrapper(env_cfg),
    # Now WildRobotEnv.step() is pure-JAX and JIT-compiled
    # Full GPU acceleration
)
```

---

## Next Steps

**Immediate (today):**
1. Decide: Brax first (Task 1b) or Pure-JAX first (Task 1a)?
2. If Brax: Start implementing `BraxWildRobotWrapper`
3. If Pure-JAX: Complete Task 1a validation scripts

**Short-term (this week):**
1. Complete chosen task (1a or 1b)
2. Start the other task
3. Integration testing

**Medium-term (next week):**
1. Full JAX stack validated
2. AMP integration (Task 5) with Brax
3. Production training runs
