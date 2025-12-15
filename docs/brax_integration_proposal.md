# Brax PPO Integration Proposal

## Problem Statement

We currently have **three training scripts** with redundant PPO implementations:
- `train.py`: Custom PPO (549 lines, works but less optimized)
- `train_brax.py`: Incomplete Brax integration (97 lines, experimental)
- `train_jax.py`: Minimal stub (39 lines, not for training)

**Original Phase 3 plan (line 5-6):**
> Framework: MuJoCo MJX Physics + Brax v2 Training (PPO + AMP)
> Decision: Use Brax v2 with MJX backend for battle-tested training infrastructure

**Current reality:** We're using a custom PPO implementation instead of Brax's battle-tested trainer.

## Why We Should Use Brax PPO

### Benefits:
1. **Battle-tested**: Used in production by DeepMind and community
2. **Optimized**: Highly optimized for JAX/GPU/TPU
3. **Feature-rich**:
   - Advanced PPO with entropy bonus, value clipping
   - Multi-epoch minibatch training
   - Proper GAE implementation
   - Distributed training support
4. **Maintained**: Active development and bug fixes
5. **Well-documented**: Extensive docs and examples
6. **Less code to maintain**: ~500 lines of custom PPO → ~50 lines of integration

### Current Custom PPO Limitations:
- ⚠️ Not optimized for GPU (no JIT compilation of training loop)
- ⚠️ Basic PPO implementation (missing advanced features)
- ⚠️ No distributed training support
- ⚠️ We maintain all bugs and optimizations
- ⚠️ Requires significant testing to match Brax quality

## Proposed Solution: Unified `train.py` with Brax PPO

### Architecture:

```python
# train.py - Unified trainer with Brax PPO integration

from brax import envs
from brax.training.agents.ppo import train as ppo_train
from playground_amp.envs.wildrobot_env import WildRobotEnv, EnvConfig

class BraxWildRobotWrapper(envs.Env):
    """Full Brax-compatible wrapper for WildRobotEnv."""

    def __init__(self, config: EnvConfig):
        self._env = WildRobotEnv(config)
        # Implement Brax Env interface:
        # - reset(rng) -> State
        # - step(state, action) -> State
        # - observation_size, action_size, backend

    @property
    def observation_size(self) -> int:
        return self._env.OBS_DIM

    @property
    def action_size(self) -> int:
        return self._env.ACT_DIM

    def reset(self, rng: jax.Array) -> envs.State:
        """Reset to Brax State pytree."""
        obs = self._env.reset()
        return envs.State(
            pipeline_state=None,  # Not needed for MJX
            obs=jnp.array(obs),
            reward=jnp.zeros(self._env.num_envs),
            done=jnp.zeros(self._env.num_envs),
            metrics={},
        )

    def step(self, state: envs.State, action: jax.Array) -> envs.State:
        """Step using Brax State pytree."""
        obs, rew, done, info = self._env.step(action)
        return state.replace(
            obs=jnp.array(obs),
            reward=jnp.array(rew),
            done=jnp.array(done),
            metrics=info,
        )

def main():
    # Load config
    env_cfg = EnvConfig.from_file('configs/wildrobot_env.yaml')

    # Create Brax-compatible env
    env = BraxWildRobotWrapper(env_cfg)

    # Use Brax PPO with minimal config
    train_fn = functools.partial(
        ppo_train.train,
        num_timesteps=1_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=2048,
        seed=0,
    )

    # Train!
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=lambda step, metrics: print(f"Step {step}: {metrics}"),
    )

    # Save policy
    save_params(params, 'checkpoints/policy.pkl')
```

### Migration Path:

**Phase 1: Complete Brax Wrapper (1 day)**
- [ ] Implement full `brax.envs.Env` interface in `BraxWildRobotWrapper`
- [ ] Handle State pytree correctly (obs, reward, done, metrics)
- [ ] Ensure vectorized reset/step work with Brax expectations
- [ ] Test with simple Brax PPO smoke run

**Phase 2: Integrate into train.py (1 day)**
- [ ] Replace custom PPO code with Brax PPO calls
- [ ] Maintain config system (YAML + CLI overrides)
- [ ] Add `--use-brax-ppo` flag for gradual migration
- [ ] Keep custom PPO as fallback during transition

**Phase 3: Add AMP Support (1-2 days)**
- [ ] Integrate AMP discriminator into Brax reward wrapper
- [ ] Hook reference motion buffer into Brax training loop
- [ ] Test AMP + PPO end-to-end

**Phase 4: Deprecate Custom PPO (after validation)**
- [ ] Remove custom PPO code from train.py
- [ ] Update documentation
- [ ] Archive train_brax.py and train_jax.py

## Compatibility Considerations

### Challenge 1: State Management
**Issue:** Brax expects immutable `State` pytrees, WildRobotEnv uses numpy arrays

**Solution:** Wrapper converts numpy ↔ JAX arrays and manages State pytree

### Challenge 2: Vectorization
**Issue:** Brax expects pure JAX functions for vmap/jit, WildRobotEnv has numpy operations

**Solution:**
- Short-term: Keep host-side numpy (acceptable performance with 16-256 envs)
- Long-term: Complete Task 1a (pure-JAX port) for full JIT compilation

### Challenge 3: Episode Management
**Issue:** Brax handles auto-reset differently than our manual reset

**Solution:** Implement Brax's `autoreset` wrapper or handle in WildRobotWrapper

### Challenge 4: AMP Integration
**Issue:** Brax doesn't have built-in AMP support

**Solution:** Extend Brax PPO with custom reward wrapper (well-documented pattern)

## Implementation Estimate

- **Time:** 2-4 days
- **Effort:** Medium (mostly integration work, not new features)
- **Risk:** Low (Brax is stable, wrapper pattern is proven)
- **Benefit:** High (production-ready training, less maintenance)

## Recommendation

**Option A (Recommended): Integrate Brax PPO now**
- ✅ Aligns with original Phase 3 plan
- ✅ Production-ready training infrastructure
- ✅ Less code to maintain
- ✅ Better performance and features
- ⏱️ 2-4 days investment upfront

**Option B: Continue with custom PPO**
- ✅ Works today for smoke testing
- ⚠️ Need to maintain/optimize our own implementation
- ⚠️ Missing advanced PPO features
- ⚠️ Deviates from original plan

**Option C: Hybrid approach**
- Add `--use-brax-ppo` flag to train.py
- Default to Brax PPO, fallback to custom if issues
- Gradual migration with safety net

## Next Steps

If approved, I recommend:

1. **Immediate:** Complete current Task 9.3 with existing `train.py` (quick-verify already passed)
2. **Next:** Add Brax PPO integration as a new task in Section 3.1.2
3. **Then:** Use Brax PPO for all subsequent training (Task 5 AMP integration, etc.)

This would add a new task:

**Task 1b: Integrate Brax PPO Trainer (NEW)**
- Status: ⏸️ NOT STARTED
- Work: Complete BraxWildRobotWrapper and integrate ppo.train()
- Timeline: 2-4 days
- Priority: HIGH (aligns with original architecture)
- Blocks: Task 7 (Optax/Flax conversion - becomes easier with Brax)

---

## Reference: Brax PPO Usage Example

```python
from brax.training.agents.ppo import train as ppo

# Minimal Brax PPO training
train_fn = ppo.train(
    environment=env,  # Any env implementing brax.envs.Env
    num_timesteps=1_000_000,
    episode_length=1000,
    num_envs=4096,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.99,
    unroll_length=20,
    batch_size=2048,
    num_minibatches=32,
    num_updates_per_batch=4,
    normalize_observations=True,
    seed=0,
    progress_fn=progress_callback,
)

inference_fn, params, metrics = train_fn(environment=env)
```

That's it! ~10 lines vs 549 lines of custom PPO.
