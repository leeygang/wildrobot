# Phase 1 Rebase Strategy
**Date**: 2025-12-06  
**Trigger**: Step 1 experiment (disable gate) FAILED to solve degradation  
**New Approach**: Rebase to Phase 0 proven config, add capabilities incrementally

---

## 📊 Step 1 Experiment Conclusion

### Training Results: `phase1_contact_step1_no_gate_flat_20251206-101354`

**Configuration**:
- Tracking gate: **DISABLED** (tested Hypothesis 1)
- All other settings: Same as Phase 1 (weak penalty, strong sliding penalty)
- Duration: 2.75M steps (137% of configured 2M)

**Performance**:
| Metric | Initial | Peak (0.1M) | Final (2.75M) | Change |
|--------|---------|-------------|---------------|--------|
| **Reward** | 17.36 | **56.07** | 26.98 | -51.9% |
| **Velocity** | 0.099 | ~0.40 | 0.155 | -61% from peak |
| **Gate Active** | 0.0% | 0.0% | 0.0% | N/A (disabled) |
| **Top Penalty** | foot_sliding | foot_sliding | **foot_sliding: -10.67** | Dominant |

**Key Observations**:
1. ✅ **Tracking gate was NOT the root cause** - disabling it didn't fix degradation
2. 🔥 **Foot sliding penalty dominates**: -10.67 (largest penalty by far)
3. 🔥 **Forward velocity bonus insufficient**: +33.09 can't overcome sliding penalty
4. ⚠️ **Early peak then collapse**: Peak at 0.1M steps, then catastrophic forgetting
5. ⚠️ **Velocity threshold penalty active**: -0.24 (robot still moving too slow)

### Root Cause Analysis (Updated)

**Hypothesis 1 (Tracking Gate)**: ❌ **REJECTED**
- Gate disabled, but degradation still occurred
- Pattern identical to original Phase 1 failure

**Hypothesis 2 (Weak Velocity Threshold Penalty)**: ✅ **LIKELY**
- Phase 0: penalty = 4.9 (constant, no decay)
- Phase 1: penalty = 1.0 (80% weaker) + decays to 0.3x
- Evidence: velocity_threshold_penalty only -0.24 (too weak to enforce fast movement)

**Hypothesis 3 (Strong Foot Sliding Penalty)**: ✅ **CONFIRMED PRIMARY**
- Phase 0: foot_sliding = 0.5
- Phase 1: foot_sliding = 5.0 (10x stronger!)
- Evidence: foot_sliding = -10.67 (dominates all other penalties)
- Math: At 0.155 m/s → velocity bonus +15.5, sliding penalty -10.67
- **Net effect**: Robot chooses slow, careful movement to minimize sliding

**Hypothesis 4 (Weak Velocity Bonus)**: ✅ **CONTRIBUTING FACTOR**
- Phase 0: forward_velocity_bonus = 51.0
- Phase 1: forward_velocity_bonus = 100.0 (2x increase)
- But sliding penalty 10x increase cancels out the benefit
- Need stronger bonus OR reduce sliding penalty

### Critical Insight: Reward Balance Broken

**Phase 0 Success Formula**:
```
STRICT velocity enforcement (penalty 4.9, constant)
  + GENTLE contact penalties (sliding 0.5)
  = Robot learns fast walking is the only viable strategy
```

**Phase 1 Failure Pattern**:
```
WEAK velocity enforcement (penalty 1.0, decaying)
  + HARSH contact penalties (sliding 5.0)
  = Robot learns slow, careful movement minimizes total penalty
```

**Conclusion**: Phase 1 inverted Phase 0's winning formula. Must revert to Phase 0's proven approach, then add contact capabilities carefully.

---

## 🔄 New Strategy: Rebase to Phase 0

### Principle: Start from Working Baseline

**Problem with Previous Approach**:
- Changed 4+ variables simultaneously (penalty strength, decay schedule, sliding weight, bonus)
- Impossible to isolate which change caused failure
- Created contradictory incentives (weak enforcement + harsh penalties)

**New Approach**:
1. **START**: Phase 0 config (proven to achieve 0.55+ m/s sustained)
2. **ADD**: Contact rewards ONE AT A TIME
3. **VALIDATE**: Each step must maintain velocity >0.40 m/s
4. **ITERATE**: Only proceed if previous step succeeds

### Phase 0 Configuration (Proven Baseline)

```yaml
# Velocity enforcement - STRICT & CONSTANT
velocity_threshold: 0.12
velocity_threshold_penalty: 4.9          # Strong enforcement
velocity_threshold_penalty_decay_start: 0    # NO DECAY
velocity_threshold_penalty_decay_steps: 0    # Constant pressure
velocity_threshold_penalty_min_scale: 0.5    # N/A (no decay)

# Forward velocity bonus - MODERATE
forward_velocity_bonus: 51.0             # Sufficient with strict enforcement

# Contact penalties - MINIMAL
foot_contact: 5.0
foot_sliding: 0.5                        # GENTLE (allows learning)

# Tracking gate - TIGHT
tracking_gate_velocity: 0.05             # Must track well to activate
tracking_gate_scale: 0.45

# Velocity commands - CHALLENGING
min_velocity: 0.5
max_velocity: 1.0
target_velocity: 0.7

# Training duration - FULL
num_timesteps: 20000000                  # 20M steps (~10 hours)
```

**Results**: Velocity 0.55+ m/s sustained, reward stable, no degradation

---

## 📋 Incremental Capability Roadmap

### Ranking by Training Performance ROI

**Ranking Criteria**:
1. **Training stability** (low risk of degradation)
2. **Sim2real transfer** (real-world benefit)
3. **Implementation complexity** (engineering cost)

### Step 1: Pure Phase 0 Rebase (Baseline Validation)

**Config**: `phase1_contact_rebase_step1_baseline.yaml`

**Changes from Phase 0**: NONE (exact copy)

**Goal**: Confirm Phase 0 config still works in current codebase

**Success Criteria**:
- Velocity >0.40 m/s by 2M steps
- Velocity >0.50 m/s by 10M steps
- No catastrophic forgetting detected
- Reward stable or increasing

**Duration**: 2M steps quick verify (~45 min)

**ROI**: ⭐⭐⭐⭐⭐ (Critical - establishes working baseline)

---

### Step 2: Add Alternation Tracking (Metrics Only)

**Config**: `phase1_contact_rebase_step2_alternation_metrics.yaml`

**Changes from Step 1**:
```yaml
tracked_reward_components:
  # Add Phase 1 contact metrics for visibility
  + contact_alternation_ratio  # Track but don't reward yet

# No reward weight changes
```

**Goal**: Verify alternation tracking doesn't break training (pure observability)

**Success Criteria**:
- Same as Step 1 (velocity, stability)
- Alternation ratio visible in W&B
- No performance degradation

**Duration**: 2M steps quick verify (~45 min)

**ROI**: ⭐⭐⭐⭐⭐ (Zero risk, pure data collection)

---

### Step 3: Gentle Contact Rewards (Minimal Weight)

**Config**: `phase1_contact_rebase_step3_gentle_contact.yaml`

**Changes from Step 2**:
```yaml
reward_weights:
  # Keep Phase 0 sliding penalty (GENTLE)
  foot_sliding: 0.5              # NO CHANGE
  
  # Add NEW contact shaping (minimal to start)
  foot_contact: 5.0              # Keep same as Phase 0
  contact_alternation: 2.0       # NEW: Gentle bonus for L→R→L gait
```

**Goal**: Add contact shaping without disrupting velocity learning

**Rationale**:
- Phase 0 already has foot_contact: 5.0 and foot_sliding: 0.5
- Only new component is alternation bonus
- Weight 2.0 provides ~1-3 points reward (small compared to velocity 50+)

**Success Criteria**:
- Velocity >0.40 m/s maintained
- Alternation ratio >30% (shows gait structure emerging)
- No velocity degradation from Step 1

**Duration**: 5M steps (~2.5 hours)

**ROI**: ⭐⭐⭐⭐ (Low risk, high sim2real value)

---

### Step 4: Strengthen Sliding Penalty (Moderate)

**Config**: `phase1_contact_rebase_step4_moderate_sliding.yaml`

**Changes from Step 3**:
```yaml
reward_weights:
  foot_sliding: 2.0              # CHANGED: 0.5 → 2.0 (4x increase, but still half Phase 1)
  contact_alternation: 3.0       # CHANGED: 2.0 → 3.0 (compensate for added difficulty)
```

**Goal**: Increase sliding penalty while maintaining velocity

**Rationale**:
- 2.0x is middle ground between Phase 0 (0.5) and Phase 1 (5.0)
- Compensate with stronger alternation bonus to maintain net reward
- Test if robot can maintain velocity with moderate sliding penalty

**Success Criteria**:
- Velocity >0.40 m/s maintained
- Sliding velocity decreases from Step 3
- Alternation ratio >50% (stronger gait structure)

**Duration**: 5M steps (~2.5 hours)

**ROI**: ⭐⭐⭐ (Medium risk, essential for sim2real)

---

### Step 5: Full Phase 1 Sliding Penalty (High Risk)

**Config**: `phase1_contact_rebase_step5_strong_sliding.yaml`

**Changes from Step 4**:
```yaml
reward_weights:
  foot_sliding: 5.0              # CHANGED: 2.0 → 5.0 (Phase 1 target)
  contact_alternation: 5.0       # CHANGED: 3.0 → 5.0 (match sliding weight)
  forward_velocity_bonus: 100.0  # CHANGED: 51.0 → 100.0 (double to overcome difficulty)
```

**Goal**: Achieve Phase 1's strict sliding penalty while maintaining velocity

**Rationale**:
- This is the Phase 1 target configuration
- But approached incrementally with proven stable base
- Double velocity bonus to overcome added difficulty

**Success Criteria**:
- Velocity >0.40 m/s maintained
- Sliding velocity <2.5 m/s (significant improvement)
- Alternation ratio >70% (clean gait)

**Duration**: 10M steps (~5 hours)

**ROI**: ⭐⭐ (High risk, but necessary for Phase 1 goal)

**Contingency**: If velocity degrades:
- Revert to Step 4 (moderate sliding: 2.0)
- Accept that 5.0 is too harsh for this robot
- Proceed to Phase 2 imitation learning

---

### Step 6: Easier Velocity Commands (Optional Polish)

**Config**: `phase1_contact_rebase_step6_easier_commands.yaml`

**Changes from Step 5**:
```yaml
env:
  min_velocity: 0.3              # CHANGED: 0.5 → 0.3
  max_velocity: 0.6              # CHANGED: 1.0 → 0.6
  target_velocity: 0.5           # CHANGED: 0.7 → 0.5
```

**Goal**: Match velocity commands to robot's natural capability

**Rationale**:
- If Step 5 struggles to maintain 0.5+ m/s with harsh penalties
- Reduce target to achievable range
- Better to succeed at 0.45 m/s than fail at 0.7 m/s

**Success Criteria**:
- Velocity >0.40 m/s (lower bar acceptable)
- Sliding velocity <2.5 m/s
- Alternation ratio >80% (clean gait at achievable speed)

**Duration**: 10M steps (~5 hours)

**ROI**: ⭐ (Low priority - only if Step 5 fails)

---

## 📊 Expected Outcomes by Step

| Step | Velocity | Alternation | Sliding | Training Time | Risk |
|------|----------|-------------|---------|---------------|------|
| 1. Baseline | 0.55+ | N/A | ~6 m/s | 2M / 45min | ⭐ Very Low |
| 2. Metrics | 0.55+ | 20-40% | ~6 m/s | 2M / 45min | ⭐ Very Low |
| 3. Gentle Contact | 0.50+ | 30-50% | ~5 m/s | 5M / 2.5hr | ⭐⭐ Low |
| 4. Moderate Sliding | 0.45+ | 50-70% | ~3 m/s | 5M / 2.5hr | ⭐⭐⭐ Medium |
| 5. Strong Sliding | 0.40+ | 70-85% | <2.5 m/s | 10M / 5hr | ⭐⭐⭐⭐ High |
| 6. Easy Commands | 0.40+ | 80-90% | <2.5 m/s | 10M / 5hr | ⭐⭐ Low |

**Total Training Time**: ~12-15 hours (Steps 1-5)

**Success Path**: If all steps succeed, we have Phase 1 working!

**Failure Path**: Revert to last successful step, proceed to Phase 2 imitation learning

---

## 🎯 Decision Points

### After Step 3 (Gentle Contact):
- ✅ **SUCCESS**: Proceed to Step 4
- ❌ **FAILURE**: Contact alternation incompatible with velocity, skip to Phase 2

### After Step 4 (Moderate Sliding):
- ✅ **SUCCESS**: Proceed to Step 5
- ❌ **FAILURE**: Accept foot_sliding: 2.0 as maximum, proceed to Phase 2

### After Step 5 (Strong Sliding):
- ✅ **SUCCESS**: Phase 1 COMPLETE! Proceed to Phase 2
- ❌ **FAILURE**: Try Step 6 (easier commands)
- ❌ **STILL FAILS**: Revert to Step 4, proceed to Phase 2

### After Step 6 (Easy Commands):
- ✅ **SUCCESS**: Phase 1 COMPLETE (at reduced velocity)
- ❌ **FAILURE**: Contact rewards + strict sliding fundamentally incompatible with RL
  - Conclusion: Need imitation learning (Phase 2) to provide trajectory priors
  - RL alone insufficient to balance velocity + contact quality

---

## �� Key Lessons

### From Failed Experiments

1. **Never change multiple variables**: Phase 1 changed 4+ things simultaneously
2. **Trust the working baseline**: Phase 0 succeeded for a reason
3. **Incentives must align**: Weak enforcement + harsh penalties = slow movement
4. **Test hypotheses systematically**: Step 1 proved gate wasn't the issue

### From Phase 0 Success

1. **Strict enforcement works**: Constant penalty 4.9 (no decay) forced fast walking
2. **Gentle penalties allow learning**: Sliding 0.5 let robot explore gaits
3. **Simple is better**: Phase 0 had fewer terms, clearer gradients
4. **Duration matters**: 20M steps gave time to stabilize

### For Future Phases

1. **Start from proven config**: Always baseline against Phase 0
2. **Add one capability at a time**: Validate before proceeding
3. **Monitor for degradation**: Use monitor_training.py with --watch
4. **Have rollback plan**: Know which step to revert to if failure occurs

---

## 📁 Files to Create

1. `phase1_contact_rebase_step1_baseline.yaml` - Pure Phase 0 copy
2. `phase1_contact_rebase_step2_alternation_metrics.yaml` - Add tracking only
3. `phase1_contact_rebase_step3_gentle_contact.yaml` - Minimal contact rewards
4. `phase1_contact_rebase_step4_moderate_sliding.yaml` - Sliding 2.0
5. `phase1_contact_rebase_step5_strong_sliding.yaml` - Sliding 5.0 (Phase 1 target)
6. `phase1_contact_rebase_step6_easier_commands.yaml` - Reduced velocity targets (optional)

---

## 📊 Step 2 Experiment Results

### Training Completed: `phase1_contact_rebase_step2_alternation_metrics_flat_20251206-120247`

**Configuration**:
- Base: Phase 0 proven config (strict velocity enforcement: 4.9, gentle sliding: 0.5)
- Change: Added `contact_alternation_ratio` to tracked metrics (NO reward weight)
- Warm-start: Loaded from `saved_checkpoints/phase0_final_policy.pkl`
- Duration: ~5.1M steps (255% of configured 2M - continued from Phase 0 checkpoint)

**Performance Trajectory**:
| Metric | Initial (0 steps) | Peak (131K) | Trough (1.7M) | Final (5.1M) | Status |
|--------|-------------------|-------------|---------------|--------------|--------|
| **Reward** | 13.96 | **28.13** | 9.26 | 19.87 | 📉 67% drop, 56% recovery |
| **Velocity** | 0.110 m/s | **0.486** m/s | 0.156 m/s | 0.304 m/s | 📉 68% drop from peak |
| **Success Rate** | 55.3% | 55.3% | 39.8% | 46.4% | 📉 Degraded from baseline |
| **Alternation Ratio** | 0.580 | 0.592 | 0.556 | 0.593 | ⚠️ Stuck ~60% (target: >85%) |
| **Left Sliding** | 4.97 m/s | 4.25 m/s | 4.11 m/s | 3.28 m/s | ⚠️ Still very high |
| **Right Sliding** | 4.24 m/s | 4.11 m/s | 2.79 m/s | 3.41 m/s | ⚠️ Still very high |
| **Gate Active** | 48% | 25% | 41% | 36% | ⚠️ High gating rate |

### Key Observations

**1. Catastrophic Forgetting Confirmed**
- ✅ Warm-started from Phase 0 checkpoint at ~2.5M effective steps
- 🔥 **Major performance drop**: Reward collapsed 67% from peak (28.13 → 9.26)
- 🔥 **Velocity degradation**: From 0.486 m/s peak → 0.304 m/s final (37% below target)
- ⚠️ Pattern: Early peak at 131K steps, then continuous degradation through 1.7M steps
- ⚠️ Partial recovery by 5M steps, but never returned to peak performance

**2. Phase 1 Criteria: ALL FAILED**
```
✗ Alternation ratio > 0.85:     0.593 (FAIL - only 69% of target)
✗ Avg sliding vel < 0.15 m/s:   3.34 m/s (FAIL - 22x over target!)
✗ Forward velocity ≥ 0.55 m/s:  0.304 m/s (FAIL - 45% below target)
✗ Success rate > 55%:           46.4% (FAIL - 16% below target)
```

**3. Root Cause Analysis**

**Why did warm-start from Phase 0 fail?**

The warm-start created a **distribution shift problem**:

1. **Phase 0 Policy Behavior**:
   - Learned to walk fast (~0.55 m/s) with high sliding (~6 m/s)
   - Optimized for velocity tracking with minimal contact constraints
   - Policy parameters encode "fast but sloppy" walking strategy

2. **Phase 1 Environment Change**:
   - Added alternation tracking (pure observability - no reward change)
   - But policy still trained on Phase 0 rewards
   - **Hypothesis**: Even observability changes can affect gradient flow

3. **Catastrophic Forgetting Mechanism**:
   - Policy warm-started with "fast but sloppy" strategy
   - Early training (0-131K) briefly improves reward (13.96 → 28.13)
   - But policy parameters drift away from Phase 0 optimum
   - New gradients contradict Phase 0 learned behaviors
   - Performance collapses as policy forgets Phase 0 skills

**4. Tracking Gate Ineffective**
- Gate active rate remained high (25-48%)
- Failed to enforce strict velocity tracking
- Robot exploited gate by moving slowly (0.304 m/s < 0.5 target)

**5. Contact Metrics Never Improved**
- Alternation ratio stuck at ~60% throughout training
- Sliding velocity remained extremely high (3-5 m/s)
- No reward weight for these metrics = no incentive to optimize

### Critical Insights

**Insight 1: Warm-Start DOES Work (Initially)** ✓
- Phase 0 checkpoint started at 13.96 reward, 0.11 m/s
- After 131K steps of continued training: **28.13 reward, 0.49 m/s** (101% improvement!)
- **Conclusion**: Phase 0 policy CAN adapt and improve with continued training

**Insight 2: Observability-Only Causes Drift**
- Adding metrics without reward weights provides no learning signal
- Policy initially improved (benefit of continued training)
- Then drifted without new objectives to stabilize around
- **Conclusion**: Must add reward terms to provide learning signal and prevent drift

**Insight 3: Incremental Approach is Viable**
- Evidence: Policy successfully improved initially (13.96 → 28.13 reward)
- The collapse happened AFTER peak, not immediately
- **Root cause**: No reward term for new objectives → policy wandered → forgot Phase 0 skills
- **Conclusion**: Warm-start + incremental reward addition SHOULD work

**Insight 4: Need Active Reward Terms, Not Just Metrics**
- Pure observability (Step 2) caused forgetting after 131K steps
- Need small but non-zero reward weight to:
  - Provide learning signal for new objective
  - Anchor policy and prevent drift
  - Guide exploration toward desired behaviors
- **Conclusion**: Add `contact_alternation: 2.0` (small but active) in Step 3

### Step 2 Verdict: ⚠️ PARTIAL SUCCESS (Proves Warm-Start Viability)

**Original Hypothesis**: "Adding metrics-only won't break training"
- ✓ **CONFIRMED**: Warm-start works! Policy improved 101% (13.96 → 28.13 reward)
- ⚠️ **BUT**: Without reward terms, policy drifted and forgot (collapsed to 9.26)
- ❌ **FAILED**: All Phase 1 criteria still failed at end

**Lessons Learned**:
1. ✓ Warm-starting from Phase 0 DOES work (initial 101% improvement)
2. ✓ Phase 0 policy CAN adapt to continued training
3. ❌ Pure observability (no reward) causes drift after ~131K steps
4. ✓ Need active reward terms to anchor policy and prevent forgetting

**Key Discovery**: The collapse happened AFTER the peak, not immediately. This proves Phase 0 policy is adaptable - it just needs a learning signal to guide it.

---

## 🔄 Updated Strategy: Incremental Reward Addition (Keep Warm-Start!)

### Why Step 2 Showed Promise

The warm-start initially succeeded:
1. ✓ Phase 0 policy adapted successfully (13.96 → 28.13 reward)
2. ✓ Velocity improved dramatically (0.11 → 0.49 m/s)
3. ✓ Policy showed it can learn new behaviors

**What went wrong**: No reward term for new objectives → policy wandered → forgot Phase 0 skills

**Conclusion**: Warm-start is viable! Just need to add reward terms, not remove warm-start.

### Corrected Approach: Warm-Start + Incremental Reward Weights

**Principle**: Start from Phase 0, add small reward terms to guide adaptation

**Key Changes from Step 2**:
1. **Keep warm-start**: `load_checkpoint: "saved_checkpoints/phase0_final_policy.pkl"` ✓
2. **Add contact reward**: `contact_alternation: 2.0` (NEW - small but active)
3. **Keep Phase 0 dominant**: velocity terms (75.0) >> contact (2.0)
4. **Gentle sliding increase**: `foot_sliding: 1.0` (2x Phase 0, but gentle)

### Proposed Step 3: Gentle Contact Reward (With Warm-Start!)

**Config**: `phase1_contact_rebase_step3_gentle_contact_warmstart.yaml`

**Reward Weight Philosophy**:
```yaml
checkpointing:
  load_checkpoint: "saved_checkpoints/phase0_final_policy.pkl"  # KEEP warm-start!

reward_weights:
  # Velocity enforcement - KEEP PHASE 0 STRICT ENFORCEMENT
  velocity_threshold_penalty: 4.9              # NO CHANGE (Phase 0 proven formula)
  forward_velocity_bonus: 51.0                 # NO CHANGE (Phase 0 baseline)
  
  # Contact shaping - ADD GENTLE REWARD (new learning signal)
  foot_contact: 5.0                            # NO CHANGE (Phase 0 baseline)
  contact_alternation: 2.0                     # NEW: Small reward to guide gait structure
  
  # Sliding penalty - KEEP PHASE 0 GENTLE
  foot_sliding: 0.5                            # NO CHANGE (Phase 0 allows exploration)
  
  # Tracking rewards - KEEP SAME
  tracking_exp_xy: 25.0
  tracking_lin_xy: 12.0
  tracking_exp_yaw: 5.0
  tracking_lin_yaw: 1.5
  # ... (rest unchanged from Phase 0)
  
  # Gate - KEEP PHASE 0 TIGHT
  tracking_gate_velocity: 0.05                 # NO CHANGE
  tracking_gate_scale: 0.45                    # NO CHANGE
```

**Rationale**:
- **Keep Phase 0 config**: Proven to work (velocity enforcement, gentle sliding)
- **Add ONLY contact_alternation: 2.0**: Small (4% of velocity bonus) but provides learning signal
- **Minimal change**: Only 1 new term, everything else unchanged
- **Learning signal**: 2.0 weight prevents drift while not dominating velocity objective

**Success Criteria**:
- ✅ Velocity >0.45 m/s sustained (near Phase 0's 0.55 m/s)
- ✅ Alternation ratio >0.65 by 2M steps (improvement from 0.60 baseline)
- ✅ No catastrophic forgetting (reward stable, no collapse like Step 2)
- ✅ Reward maintains peak or grows (unlike Step 2's collapse)

**Duration**: 2M steps (~1 hour quick verify)

**Expected Outcome**:
- Velocity: ~0.50 m/s (Phase 0 level maintained)
- Alternation: 0.65-0.70 (gradual improvement from 0.60)
- Sliding: ~4-5 m/s (unchanged, acceptable for Step 3)
- **No forgetting**: Stable reward, no collapse

**If Successful**: Proceed to Step 4 (strengthen contact rewards + add sliding penalty)

---

## 📋 Updated Roadmap

### Completed Steps

- ✅ **Step 1**: No-gate experiment → Rejected gate hypothesis
- ✅ **Step 2**: Metrics-only + warm-start → Catastrophic forgetting, warm-start failed
- ❌ **Original Steps 3-6**: Abandoned (warm-start approach not viable)

### Remaining Path

- 🔄 **Step 3 (CORRECTED)**: Warm-start + gentle contact reward
  - Config: `phase1_contact_rebase_step3_gentle_contact_warmstart.yaml`
  - Change: Add ONLY `contact_alternation: 2.0`, keep everything else Phase 0
  - Duration: 2M steps (~1 hour)
  - Risk: ⭐⭐ Low (minimal change, proven warm-start)

- 🔮 **Step 4**: If Step 3 succeeds (no forgetting), strengthen contact rewards
  - Increase `contact_alternation: 2.0 → 5.0`
  - Increase `foot_sliding: 0.5 → 1.5` (gentle increase)
  - Duration: 5M steps
  - Target: Alternation >0.75, velocity maintained

- 🔮 **Step 5**: Continue strengthening (if Step 4 succeeds)
  - Increase `foot_sliding: 1.5 → 2.5`
  - Increase `contact_alternation: 5.0 → 8.0`
  - Target: Alternation >0.80, sliding improving

- 🔮 **Step 6 (FALLBACK)**: If forgetting persists, try Phase 2 Imitation
  - Use motion reference from best checkpoint
  - Add imitation rewards (pose tracking, COM tracking)
  - Leverage trajectory priors to guide contact patterns

### Decision Tree

```
Step 3 (Warm-start + Gentle Contact Reward)
├─ SUCCESS (no forgetting, alt >0.65)
│  └─ Step 4: Strengthen contact_alternation → 5.0, gentle sliding → 1.5
│     ├─ SUCCESS (no forgetting, alt >0.75)
│     │  └─ Step 5: Further strengthen → sliding 2.5, alternation 8.0
│     │     └─ SUCCESS → Phase 1 approaching complete
│     └─ FAIL (forgetting returns)
│        └─ Use Step 3 checkpoint, proceed to Phase 2
└─ FAIL (catastrophic forgetting persists even with reward term)
   └─ CONCLUSION: Need different approach
      └─ Try lower learning rate OR proceed to Phase 2 (Imitation)
```

**Why This Should Work**:
- Step 2 proved warm-start works (101% initial improvement)
- Adding small reward term (2.0) provides learning signal
- Keeps Phase 0 velocity enforcement (4.9) strong
- Minimal change reduces risk

---

**Next Action**: Create `phase1_contact_rebase_step3_gentle_contact_warmstart.yaml` with:
- Warm-start from Phase 0
- Add ONLY `contact_alternation: 2.0`
- Keep all other Phase 0 settings unchanged
- Run 2M steps quick verify

