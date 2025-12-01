# WildRobot AMP Test Suite

**Purpose:** Catch JAX pytree structure errors and other contract violations **before** starting expensive training runs.

---

## 🎯 **Why These Tests Are Critical**

### **The Problem:**
JAX's `jax.lax.scan` requires that the state structure (pytree) remains **identical** between iterations:
- `reset()` must initialize **all** fields that `step()` will use
- Missing even a single field causes: `TypeError: scan body function carry input and carry output must have the same pytree structure`

### **The Cost:**
- Training takes **8-10 hours** per run
- Finding a structure error **after starting** wastes GPU time
- **Solution:** Run these tests (takes ~30 seconds) before every training run

---

## 🧪 **Test Categories**

### **1. JAX Pytree Contracts (CRITICAL)**
Tests that prevent JAX scan errors:
- ✅ `test_reset_step_metrics_structure_match` - Metrics dict keys must match
- ✅ `test_reset_step_info_structure_match` - Info dict keys must match
- ✅ `test_all_reward_components_initialized` - All rewards from config initialized
- ✅ `test_forward_velocity_bonus_in_metrics` - Specific test for Phase 1E fix

### **2. Environment Basics**
Tests for correct environment behavior:
- ✅ `test_reset_returns_valid_state` - Reset produces valid initial state
- ✅ `test_step_produces_valid_outputs` - Step produces valid outputs
- ✅ `test_forward_velocity_bonus_computation` - Bonus = max(0, velocity)
- ✅ `test_multiple_steps_maintain_structure` - Structure consistent over time

### **3. Reward Components**
Tests for specific reward logic:
- ✅ `test_existential_penalty_applied` - Existential penalty is -5.0
- ✅ `test_forward_velocity_bonus_positive_for_forward_motion` - Bonus logic correct

---

## 🚀 **Usage**

### **Quick Run (Before Training):**
```bash
cd /Users/ygli/projects/wildrobot/amp

# Run tests directly (no pytest required)
python tests/test_env_contracts.py
```

**Expected output:**
```
================================================================================
WILDROBOT AMP ENVIRONMENT CONTRACT TESTS
================================================================================

Creating environment...
Environment created: obs_size=62, action_size=11

Running: Metrics structure match... ✅ PASS: metrics structure match (41 keys)
Running: Info structure match... ✅ PASS: info structure match (6 keys)
Running: All reward components initialized... ✅ PASS: all 24 reward components initialized
Running: forward_velocity_bonus in metrics... ✅ PASS: forward_velocity_bonus present in both reset() and step()
Running: Reset returns valid state... ✅ PASS: reset() returns valid state (obs_size=62)
Running: Step produces valid outputs... ✅ PASS: step() produces valid outputs
Running: forward_velocity_bonus computation... ✅ PASS: forward_velocity_bonus computed correctly (vel=-0.001, bonus=0.000)
Running: Multiple steps maintain structure... ✅ PASS: structure maintained over 10 steps
Running: Existential penalty applied... ✅ PASS: existential penalty is -5.0
Running: forward_velocity_bonus logic... ✅ PASS: bonus logic correct (vel=-0.001, bonus=0.000)

================================================================================
TEST SUMMARY: 10 passed, 0 failed
================================================================================

✅ ALL TESTS PASSED - SAFE TO START TRAINING!
```

### **With pytest (Optional):**
```bash
# Install pytest if needed
pip install pytest

# Run with verbose output
python -m pytest tests/test_env_contracts.py -v

# Run specific test
python -m pytest tests/test_env_contracts.py::TestJAXPytreeContracts::test_reset_step_metrics_structure_match -v
```

### **On Remote Server (Before Training):**
```bash
# SSH to remote
ssh leeygang@linux-pc.local
cd ~/projects/wildrobot/amp

# Run tests
python tests/test_env_contracts.py

# If tests pass, start training
python train.py --config phase1_contact.yaml
```

---

## 🔧 **Adding New Tests**

When adding new reward components or modifying the environment:

### **1. Add to test config:**
```python
config.reward_weights.my_new_reward = 1.0
```

### **2. Verify initialization:**
The test `test_all_reward_components_initialized` will automatically check it!

### **3. Add specific test (if needed):**
```python
def test_my_new_reward_computation(self, env):
    """Test my_new_reward is computed correctly."""
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    action = jp.zeros(env.action_size)
    state_next = env.step(state, action)

    my_reward = state_next.metrics["reward/my_new_reward"]
    # Add assertions...
```

---

## ⚠️ **Common Failures & Fixes**

### **Failure: "Keys in step() but missing in reset()"**
```
AssertionError: Keys in step() but missing in reset(): {'reward/my_new_component'}
This will cause JAX scan to fail!
```

**Fix:** Add to `reset()` metrics initialization:
```python
metrics = {
    # ... existing metrics ...
    "reward/my_new_component": jp.zeros(()),  # ADD THIS
}
```

### **Failure: "forward_velocity_bonus not initialized"**
```
AssertionError: forward_velocity_bonus not initialized in reset()!
```

**Fix:** Already fixed in current code, but if you see this:
```python
"reward/forward_velocity_bonus": jp.zeros(()),  # Must be in reset()
```

### **Failure: "Pytree structure mismatch"**
```
AssertionError: Pytree structure mismatch at metrics:
reset(): <dict with 40 children>
step():  <dict with 41 children>
```

**Fix:** Ensure all keys in `step()` are also in `reset()` with same types (jp.zeros(()) for scalars)

---

## 📊 **Test Execution Time**

| Test Suite | Duration | When to Run |
|------------|----------|-------------|
| **Full suite** | ~30 seconds | Before every training |
| **Contract tests only** | ~10 seconds | After env changes |
| **Basic tests** | ~5 seconds | Quick sanity check |

**ROI:** 30 seconds of testing saves 8-10 hours of wasted training!

---

## 🎯 **Integration with Training Workflow**

### **Recommended Workflow:**

```bash
# 1. Make changes to environment or config
vim walk_env.py
vim phase1_contact.yaml

# 2. Run tests locally
python tests/test_env_contracts.py

# 3. If tests pass, copy to remote
scp walk_env.py phase1_contact.yaml leeygang@linux-pc.local:~/projects/wildrobot/amp/

# 4. SSH and run tests on remote
ssh leeygang@linux-pc.local
cd ~/projects/wildrobot/amp
python tests/test_env_contracts.py

# 5. If tests pass, start training
python train.py --config phase1_contact.yaml
```

### **Pre-commit Hook (Future):**
```bash
#!/bin/bash
# .git/hooks/pre-commit
cd amp
python tests/test_env_contracts.py
if [ $? -ne 0 ]; then
    echo "Tests failed! Fix before committing."
    exit 1
fi
```

---

## 📝 **Test Coverage**

Current coverage:
- ✅ JAX pytree structure (metrics, info)
- ✅ Reward component initialization
- ✅ Basic environment functionality
- ✅ forward_velocity_bonus logic
- ✅ Existential penalty
- ⏳ Contact rewards (future)
- ⏳ Phase 2/3 features (future)

---

## 🚨 **MANDATORY: Run Before Training**

**DO NOT START TRAINING WITHOUT RUNNING THESE TESTS!**

If tests fail:
1. ❌ **DO NOT** start training
2. 🔧 Fix the issue
3. ✅ Re-run tests until they pass
4. 🚀 Then (and only then) start training

**Remember:** 30 seconds of testing >> 8-10 hours of wasted GPU time!

---

## 📚 **Related Files**

- `walk_env.py` - Environment implementation
- `phase1_contact.yaml` - Training configuration
- `train.py` - Training script
- `TRAINING_PLAN.md` - Overall training strategy
