# WildRobot AMP: 3-Phase Training Plan

**Goal:** Train a humanoid robot to walk with natural, human-like gait using a curriculum learning approach.

**Strategy:** Start simple (basic locomotion), then refine (contact patterns), then polish (human-like motion).

---

## 🎯 **Phase 1E: Foundation - Basic Forward Locomotion**

### **Goal:**
Learn to walk forward reliably at 0.5-0.8 m/s **without** degrading to local minima (backward movement, standing still).

### **Philosophy:**
**"First learn to walk, then learn to walk well"**

Strip down to absolute minimum rewards:
- ✅ Velocity tracking (encourage forward motion)
- ✅ **Forward velocity bonus** (NEW - prevents exploitation)
- ✅ Stability (stay upright, minimal bobbing)
- ✅ Basic smoothness (avoid joint limits, jerky movements)
- ❌ NO contact penalties (too restrictive for initial learning)

### **Key Innovation: Forward Velocity Bonus**
```python
forward_velocity_bonus = max(0, velocity) × 10.0
```

**Why this fixes the local minimum issue:**
- Backward motion: 0 bonus ❌
- Standing still: 0 bonus ❌
- Forward at 0.7 m/s: +7.0 bonus ✅

Tracking rewards alone give 42% reward even when moving backward (`exp(-error²)` is too forgiving). The bonus eliminates this loophole.

### **Config Highlights:**
```yaml
reward_weights:
  # Tracking (same as proven baseline/playground)
  tracking_exp_xy: 50.0
  tracking_lin_xy: 15.0

  # NEW: Explicit forward incentive
  forward_velocity_bonus: 10.0

  # Contact penalties: ALL DISABLED
  foot_contact: 0.0
  foot_sliding: 0.0
  foot_air_time: 0.0

  # Smoothness: Minimal (allow dynamic movement)
  joint_velocity: 0.0         # Disabled
  joint_acceleration: 1e-8    # Nearly negligible

  # Existential: Force action
  existential: 1.0  # Applied as -5.0 in code
```

### **Success Criteria:**

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| **Forward velocity** | 0.5-0.8 m/s | > 0.3 m/s |
| **Training stability** | Stable or improving | No degradation |
| **Success rate** | 65-75% | > 55% |
| **Velocity trend** | Consistently positive | Never backward |
| **forward_velocity_bonus** | 5.0-8.0 | > 3.0 |

### **Expected Gait Quality:**
- ⚠️ **NOT human-like** (no contact guidance)
- ⚠️ May shuffle, slide, or have awkward foot placement
- ⚠️ Dynamic but potentially jerky
- ✅ Functional forward locomotion
- ✅ Stable and reliable
- **Human-like score: 3/10** (functional but rough)

### **Timeline:**
- **Training:** 20M steps (~8-10 hours on 12GB GPU)
- **First checkpoint:** 10% (2M steps, ~1 hour)
  - Check: velocity > 0.3 m/s, no backward movement
- **Mid checkpoint:** 50% (10M steps, ~4 hours)
  - Check: velocity > 0.5 m/s, stable reward
- **Final checkpoint:** 100% (20M steps)
  - Expected: 0.6-0.8 m/s, 70%+ success rate

### **If Phase 1E Fails:**
- Tracking rewards themselves need redesign
- Consider stricter formula: `exp(-k×error²)` with k > 1
- May need exploration bonuses or shaped curriculum

### **If Phase 1E Succeeds:**
- ✅ Foundation established
- ✅ Ready for Phase 2 contact refinement
- ✅ Checkpoint can be used as initialization for Phase 2

---

## 🔧 **Phase 2: Refinement - Contact Pattern Learning**

### **Goal:**
Improve gait quality by adding contact penalties **gradually** to encourage natural foot placement without losing forward motion.

### **Philosophy:**
**"Don't break what works - refine it carefully"**

Use **curriculum learning** to introduce contact constraints:
1. Start with Phase 1E checkpoint (proven to walk)
2. Add contact penalties ONE AT A TIME
3. Start with VERY LOW weights (10x lower than failed attempts)
4. Increase gradually IF training remains stable

### **Curriculum Strategy:**

#### **Phase 2A: Minimal Contact Rewards (0-5M steps)**
Start training from Phase 1E checkpoint with tiny contact penalties:

```yaml
reward_weights:
  # Keep Phase 1E rewards unchanged
  tracking_exp_xy: 50.0
  tracking_lin_xy: 15.0
  forward_velocity_bonus: 10.0  # Keep!

  # Add MINIMAL contact penalties
  foot_contact: 1.0     # Was 10.0 in Phase 1C (10x lower)
  foot_sliding: 0.2     # Was 2.0 in Phase 1C (10x lower)
  foot_air_time: 0.5    # Was 5.0 in Phase 1C (10x lower)
```

**Monitor:** If velocity stays > 0.5 m/s, proceed to Phase 2B

#### **Phase 2B: Moderate Contact Rewards (5-10M steps)**
Increase weights gradually:

```yaml
  foot_contact: 3.0     # 3x increase from Phase 2A
  foot_sliding: 0.5     # 2.5x increase
  foot_air_time: 1.5    # 3x increase
```

**Monitor:** If velocity stays > 0.5 m/s, proceed to Phase 2C

#### **Phase 2C: Full Contact Rewards (10-15M steps)**
Reach target weights:

```yaml
  foot_contact: 5.0     # Near original 10.0
  foot_sliding: 1.0     # Half of original 2.0
  foot_air_time: 2.5    # Half of original 5.0
```

**Note:** May not reach full original weights if they cause degradation

### **Adaptive Strategy:**
At each substep:
- ✅ **If stable (velocity > 0.5 m/s):** Continue to next step
- ⚠️ **If degrading (velocity < 0.4 m/s):** Reduce weights by 50%
- ❌ **If failing (velocity < 0.2 m/s):** Revert to previous weights

### **Success Criteria:**

| Metric | Target | Improvement over Phase 1E |
|--------|--------|---------------------------|
| **Forward velocity** | 0.6-0.8 m/s | Maintained or +10% |
| **Foot sliding velocity** | < 0.5 m/s | -50% reduction |
| **Proper foot contacts** | > 60% of steps | +30% improvement |
| **Foot air time** | > 0.15s average | Natural swing phase |
| **Success rate** | 70-80% | +5-10% |

### **Expected Gait Quality:**
- ✅ Reduced foot sliding (more natural foot placement)
- ✅ Better contact timing (alternating legs)
- ✅ Smoother weight transfer
- ⚠️ Still not fully human-like (no motion imitation)
- **Human-like score: 7/10** (good functional gait)

### **Timeline:**
- **Phase 2A:** 5M steps (~2 hours)
- **Phase 2B:** 5M steps (~2 hours)
- **Phase 2C:** 5M steps (~2 hours)
- **Total:** 15M steps (~6 hours)

### **Risk Mitigation:**
- Start from Phase 1E checkpoint (don't train from scratch)
- Monitor velocity CLOSELY at each substep
- Be conservative with weight increases
- Accept partial success (don't need to reach full original weights)

---

## 🎨 **Phase 3: Polish - Human-Like Motion Imitation**

### **Goal:**
Achieve natural, human-like walking by imitating motion capture data while maintaining robust locomotion.

### **Philosophy:**
**"Make it beautiful without breaking functionality"**

Combine physics-based rewards (Phase 2) with motion imitation:
- ✅ Keep velocity tracking + contact rewards from Phase 2
- ✅ Add motion imitation term (compare to mocap reference)
- ✅ Balance task performance vs. style matching

### **Approach: Adversarial Motion Priors (AMP)**

Based on: ["AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"](https://arxiv.org/abs/2104.02180)

**Key idea:** Train a discriminator to distinguish real human motion from robot motion, use it to shape rewards.

### **Architecture:**

```
┌─────────────────────────────────────────┐
│         Phase 2 Checkpoint              │
│  (Functional walking with contacts)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        Motion Discriminator             │
│  Input: Robot state trajectory          │
│  Output: "How human-like is this?"      │
│  Trained on: Human mocap data           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Combined Reward                 │
│  = Task rewards (Phase 2)               │
│    + Style reward (discriminator)       │
│    + Regularization                     │
└─────────────────────────────────────────┘
```

### **Reward Formulation:**

```python
total_reward = (
    # Task rewards (from Phase 2)
    w_task × (tracking + contact + stability)

    # Style reward (NEW - from discriminator)
    + w_style × discriminator_score

    # Regularization (prevent overfitting to mocap)
    + w_reg × diversity_bonus
)
```

**Weight schedule:**
- Start: `w_task=1.0, w_style=0.1` (prioritize task)
- Middle: `w_task=1.0, w_style=0.5` (balance)
- End: `w_task=1.0, w_style=1.0` (equal weight)

### **Data Requirements:**

**Motion capture dataset:**
- Source: CMU MoCap, Human3.6M, or custom recordings
- Format: Joint positions/velocities at 50-100 Hz
- Actions: Walking at various speeds (0.5-1.0 m/s)
- Duration: 5-10 minutes of clean walking data

**Preprocessing:**
1. Retarget mocap skeleton → WildRobot skeleton
2. Filter and smooth trajectories
3. Normalize velocities to match command range
4. Augment with random variations

### **Training Strategy:**

#### **Phase 3A: Discriminator Pretraining (Offline)**
Train discriminator to classify real vs. fake motion:
- Real: Human mocap data
- Fake: Phase 2 robot trajectories
- Loss: Binary cross-entropy
- Architecture: LSTM or Transformer (temporal patterns)
- Duration: ~1M discriminator training steps

#### **Phase 3B: Joint Training (Online)**
Fine-tune policy + discriminator together:
- Policy learns to fool discriminator
- Discriminator adapts to new policy samples
- Use GAIL or AIRL framework
- Duration: 20M steps (~8-10 hours)

### **Success Criteria:**

| Metric | Target | Improvement over Phase 2 |
|--------|--------|--------------------------|
| **Forward velocity** | 0.7-0.9 m/s | Maintained |
| **Joint trajectory similarity** | > 0.8 correlation | Human-like motion |
| **Discriminator score** | > 0.7 | Convincing imitation |
| **Success rate** | 75-85% | +5-10% |
| **Energy efficiency** | Within 20% of human | Natural efficiency |

### **Expected Gait Quality:**
- ✅ Natural joint trajectories
- ✅ Human-like timing and rhythm
- ✅ Smooth weight transfer
- ✅ Realistic arm swing (if included)
- ✅ Visually indistinguishable from human walking
- **Human-like score: 9/10** (near-perfect imitation)

### **Timeline:**
- **Discriminator pretraining:** Offline (~1 hour)
- **Joint training:** 20M steps (~8-10 hours)
- **Total:** ~10-12 hours

### **Risk Mitigation:**
- Start from Phase 2 checkpoint (don't train from scratch)
- Gradually increase style weight (curriculum)
- Monitor task performance (don't sacrifice velocity for style)
- Use reward clipping to prevent discriminator exploitation

### **Alternative: Skip Phase 3**
If Phase 2 achieves acceptable gait quality:
- Deploy Phase 2 model directly
- Save Phase 3 for future refinement
- Phase 2 (7/10 quality) may be "good enough" for many applications

---

## 📊 **Overall Training Progression**

| Phase | Duration | Focus | Gait Quality | Velocity | Success Rate |
|-------|----------|-------|--------------|----------|--------------|
| **1E** | ~10 hours | Forward locomotion | 3/10 (rough) | 0.6-0.8 m/s | 65-75% |
| **2** | ~6 hours | Contact refinement | 7/10 (good) | 0.6-0.8 m/s | 70-80% |
| **3** | ~10 hours | Human-like motion | 9/10 (excellent) | 0.7-0.9 m/s | 75-85% |
| **Total** | **~26 hours** | Full pipeline | **9/10** | **0.7-0.9 m/s** | **75-85%** |

---

## 🎯 **Decision Points**

### **After Phase 1E:**
- ✅ **Success (velocity > 0.5 m/s):** Proceed to Phase 2
- ❌ **Failure (velocity degrading):** Debug tracking rewards, try stricter formula

### **After Phase 2:**
- ✅ **Good quality (7/10):** Decide if Phase 3 needed
- ⚠️ **Partial success (5/10):** Adjust contact weights, possibly skip Phase 3
- ❌ **Failure:** Revert to Phase 1E, redesign contact curriculum

### **After Phase 3:**
- ✅ **Success:** Deploy to robot!
- ⚠️ **Style improved but task degraded:** Balance weights differently
- ❌ **Failure:** Deploy Phase 2 model instead

---

## 🚀 **Deployment Readiness**

### **Minimum Viable Product (MVP):**
- **Phase 1E:** Basic locomotion demo
- **Use case:** Proof of concept, research demos
- **Quality:** Functional but not polished

### **Production Ready:**
- **Phase 2:** Good quality locomotion
- **Use case:** Warehouse navigation, simple tasks
- **Quality:** Professional, reliable

### **Premium:**
- **Phase 3:** Human-like locomotion
- **Use case:** Public demos, human-robot interaction
- **Quality:** Indistinguishable from human walking

---

## 📝 **Current Status**

- ✅ **Phase 1E config ready** (`phase1_contact.yaml`)
- ✅ **Phase 1E code ready** (`walk_env.py` with `forward_velocity_bonus`)
- ✅ **Monitoring tools ready** (`monitor_training.py`)
- ⏳ **Phase 1E training:** Ready to launch
- ⏳ **Phase 2 config:** To be created after Phase 1E success
- ⏳ **Phase 3 implementation:** Future work

**Next step: Launch Phase 1E training and monitor for 10% progress checkpoint!**

---

## 🔗 **Related Documents**

- `baseline.yaml` - Proven baseline configuration
- `playground_analysis.md` - Analysis of playground implementation
- `three_way_comparison.md` - Comparison of configs
- `compare_configs.md` - Baseline vs Phase 1D comparison
- `monitor_training.py` - Training monitoring tool
- `verify_metrics.py` - Metrics verification tool

**Good luck with training! 🚀**
