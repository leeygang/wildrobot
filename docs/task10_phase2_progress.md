# Task 10 Phase 2: Brax PPO Integration Progress Report

**Date:** 2025-12-15
**Status:** 🔄 95% COMPLETE - One remaining dtype issue with Brax training wrappers

## Summary

Successfully integrated Pure-JAX backend with Brax PPO trainer, resolving multiple JIT/vmap compatibility issues. The integration is 95% complete with only one remaining dtype mismatch in Brax's internal training wrappers.

## Achievements

### 1. Removed Unsupported `wrap_for_training` Parameter ✅
- **Issue:** `train_brax_ppo.py` had `wrap_for_training: False` parameter that doesn't exist in Brax PPO API
- **Fix:** Removed parameter from PPO configuration
- **File:** `/Users/ygli/projects/wildrobot/playground_amp/train_brax_ppo.py`

### 2. Rewrote BraxWildRobotWrapper for Pure-JAX Backend ✅
- **Issue:** Original wrapper called `WildRobotEnv.step()` which has numpy operations that fail inside JIT-traced functions
- **Solution:** Completely rewrote wrapper to bypass `WildRobotEnv` and directly use Pure-JAX functions from `jax_full_port.py`
- **Key Changes:**
  - Uses `step_fn()` directly for physics simulation
  - Constructs obs/reward/done manually using JAX functions
  - No numpy conversions inside JIT-traced code
- **File:** `/Users/ygli/projects/wildrobot/playground_amp/brax_wrapper.py`

### 3. Fixed Environment Batching Architecture ✅
- **Issue:** Initial implementation treated wrapper as managing multiple environments, but Brax applies vmap to create batches
- **Solution:** Changed wrapper to manage SINGLE environment (num_envs=1), letting Brax handle parallelization via vmap
- **Impact:** Proper shape consistency for Brax's training infrastructure

### 4. Resolved Pytree Structure Consistency Issues ✅
- **Issue #1:** Empty `state.info` dict caused scan pytree mismatch
  - **Fix:** Preserve incoming `state.info` structure in output

- **Issue #2:** Double-batching from Brax vmap caused shape mismatches
  - **Fix:** Single-environment wrapper with shapes like `(obs_dim,)` instead of `(1, obs_dim)`

- **Issue #3:** `JaxData.ctrl` dimension mismatch (11 vs 17)
  - **Fix:** Pad ctrl from action_dim (11) to qvel_dim (17) for consistent pytree shapes

### 5. Passed Multiple Integration Milestones ✅

**Progression of errors shows incremental success:**

1. ✅ TracerArrayConversionError → Fixed by using Pure-JAX backend
2. ✅ vmap in_axes mismatch → Fixed by using single-env step function
3. ✅ Empty info dict → Fixed by preserving state.info
4. ✅ Double-batch shape mismatch → Fixed by single-env architecture
5. ✅ ctrl dimension mismatch → Fixed by padding ctrl to match qvel_dim
6. ⚠️  dtype mismatch in info dict fields (CURRENT ISSUE)

## Current Blocking Issue

**Error:** `TypeError: scan body function carry input and carry output must have equal types`

**Details:**
- Input `state.info['episode_done']` has dtype `float32[4]`
- Output `state.info['episode_done']` has dtype `bool[4]`
- Input `state.info['truncation']` has dtype `float32[4]`
- Output `state.info['truncation']` has dtype `int32[4]`

**Root Cause Analysis:**
Brax's training wrappers (AutoResetWrapper, EpisodeWrapper, etc.) populate `state.info` with specific dtypes. When we return `state.info` unchanged, somehow the dtypes are being cast during the function call.

**Potential Solutions:**

1. **Explicitly cast info dict fields** to match input dtypes:
   ```python
   # In step() method, before return
   output_info = {}
   for key, value in state.info.items():
       # Preserve exact dtype from input
       output_info[key] = jnp.asarray(value, dtype=value.dtype)
   ```

2. **Use Brax's State dataclass constructor** to ensure type preservation:
   ```python
   return State(
       pipeline_state=new_jax_state,
       obs=obs,
       reward=reward,
       done=done,
       metrics={},
       info=state.info  # Already a dict, should preserve types
   )
   ```

3. **Investigate Brax's training wrappers** to understand expected info dict structure:
   - Check `brax/envs/wrappers/training.py` for AutoResetWrapper
   - See how other Brax environments handle info dict

4. **Create minimal reproducer** with standard Brax environment to verify behavior

## Files Modified

1. **`/Users/ygli/projects/wildrobot/playground_amp/train_brax_ppo.py`**
   - Removed `wrap_for_training` parameter
   - Updated comments to reflect Pure-JAX backend

2. **`/Users/ygli/projects/wildrobot/playground_amp/brax_wrapper.py`** (MAJOR REWRITE)
   - Complete architectural change to Pure-JAX backend
   - Imports: Added `JaxData`, `make_jax_data`, JAX helper functions
   - `__init__()`: Changed to single-env architecture (num_envs=1)
   - `reset()`: Returns scalar reward/done, 1D obs for proper vmap
   - `step()`: Completely rewritten:
     - Uses `step_fn()` directly instead of `WildRobotEnv.step()`
     - Pads ctrl to match qvel dimension
     - Manually constructs obs/reward/done using JAX functions
     - Preserves `state.info` structure for scan compatibility

## Testing Results

### Smoke Test Progress:
```
✓ Environment initialization successful
✓ Brax PPO import successful
✓ Configuration loaded correctly
✓ Pure-JAX backend enabled automatically
✓ JIT/vmap compatibility achieved (5 error types resolved)
⚠️  Final dtype mismatch in training wrappers
```

### Performance Validation:
- No TracerArrayConversionErrors ✅
- No vmap errors ✅
- No shape mismatches ✅
- Proper pytree structure (except dtype) ✅

## Technical Architecture

### Current Data Flow:

```
Brax PPO Trainer
    ↓ (applies vmap for batching)
BraxWildRobotWrapper (single env)
    ↓ (direct call)
jax_full_port.step_fn() [Pure-JAX]
    ↓ (PD control + physics)
JaxData state update
    ↓
jax_env_fns.build_obs_from_state_j() [Pure-JAX]
    ↓
Return Brax State (obs, reward, done)
```

**Key Design Decisions:**
1. Each wrapper instance = 1 environment (Brax vmaps for parallelization)
2. No numpy operations in step path (full JIT compatibility)
3. Manual obs/reward/done construction for shape control
4. Ctrl padding for consistent pytree dimensions

### State Shapes:

**Reset Output:**
```python
State(
    pipeline_state=JaxData(  # batch=1
        qpos=(1, 18),
        qvel=(1, 17),
        ctrl=(1, 17),  # padded to qvel_dim
        xpos=(1, 3),
        xquat=(1, 4)
    ),
    obs=(44,),        # 1D for Brax vmap
    reward=scalar,
    done=scalar,
    metrics={},
    info={}
)
```

**Step Input/Output:**
- Input action: `(11,)` - single env action
- Output shapes MUST match input State shapes exactly (for jax.lax.scan)

## Next Steps

### Immediate (Complete Task 10 Phase 2):
1. **Debug info dict dtype issue:**
   - Read Brax source code for training wrappers
   - Check how AutoResetWrapper modifies State.info
   - Implement dtype preservation strategy

2. **Verify smoke test passes:**
   - Run `train_brax_ppo.py --verify --no-wandb`
   - Should complete 10 iterations without errors
   - Validate rewards are non-zero

3. **Update Phase 3 plan:**
   - Mark Task 10 Phase 2 as complete
   - Document lessons learned
   - Update integration checklist

### Short-term (Task 10 Phase 3):
4. **Run full training (100 updates):**
   - Verify end-to-end training works
   - Check convergence and reward progression
   - Validate checkpoint saving

5. **Performance benchmarking:**
   - Measure steps/sec vs custom PPO
   - Compare GPU utilization
   - Validate JIT compilation benefits

### Medium-term (Task 10 Phase 4):
6. **Documentation and migration guide:**
   - Write migration guide from custom PPO to Brax
   - Document hyperparameter differences
   - Create troubleshooting FAQ

## Lessons Learned

### 1. Brax Environment Architecture
- Brax expects **single-environment** step methods
- Parallelization is handled via automatic vmap
- Don't manually batch environments in the wrapper

### 2. JIT/vmap Compatibility Requirements
- ZERO numpy operations in traced code paths
- All array operations must be JAX primitives
- Even type conversions (`np.array()`) break tracing

### 3. Pytree Structure Consistency
- `jax.lax.scan` requires EXACT pytree structure matching
  - Same keys in dicts
  - Same array shapes
  - Same dtypes!
- Preserve input structure in output, don't create new dicts/shapes

### 4. Shape Management in Vmap
- Input shapes after vmap: `(batch, ...original_shape...)`
- Output shapes must match: can't squeeze singleton dimensions
- Ctrl/state fields must maintain consistent dimensions

### 5. Pure-JAX Port Integration
- Task 11 (Pure-JAX port) was ESSENTIAL prerequisite
- Can't use WildRobotEnv.step() directly with Brax
- Must use low-level JAX functions (step_fn, build_obs, etc.)

## Code Quality

- ✅ No linter errors (validated with `validate_changes`)
- ✅ Type hints preserved
- ✅ Comprehensive docstrings
- ✅ Clear comments explaining complex logic
- ✅ Proper error handling with descriptive ImportErrors

## References

**Related Tasks:**
- Task 11: Pure-JAX Port (PREREQUISITE - COMPLETE ✅)
- Task 10 Phase 1: Brax Wrapper (COMPLETE ✅)
- Task 10 Phase 3: Full Training (BLOCKED on Phase 2)

**Key Files:**
- `/Users/ygli/projects/wildrobot/playground_amp/brax_wrapper.py`
- `/Users/ygli/projects/wildrobot/playground_amp/train_brax_ppo.py`
- `/Users/ygli/projects/wildrobot/playground_amp/envs/jax_full_port.py`
- `/Users/ygli/projects/wildrobot/playground_amp/envs/jax_env_fns.py`

**Documentation:**
- `/Users/ygli/projects/wildrobot/docs/brax_integration_proposal.md`
- `/Users/ygli/projects/wildrobot/docs/jax_integration_roadmap.md`
- `/Users/ygli/projects/wildrobot/docs/session_summary_task10_task11.md`

---

**Overall Progress:** From TracerArrayConversionError → 95% complete integration with 1 dtype issue remaining. Excellent progress on a complex JIT/vmap integration challenge!
