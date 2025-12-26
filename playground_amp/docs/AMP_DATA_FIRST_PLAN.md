# AMP Data-First Implementation Plan

## Version: 1.1.0
## Date: 2025-12-25

---

## Executive Summary

This document defines a **data-first approach** to AMP training with:
- **Hard quantitative gates** at each phase
- **Clear stop conditions** to avoid sunk-cost fallacy
- **Fallback path** to baseline PPO if gates fail
- **Prescriptive parameters** for segment mining and feature dropping

The goal is NOT to make AMASS retargeting perfect. It is to produce **sufficiently large, sufficiently clean physics reference** that does not rely on harness artifacts and has usable contact structure.

---

## First Principles: What Must Be True for AMP to Work

AMP from scratch will work **when ALL THREE are true**:

| # | Condition | Measurement |
|---|-----------|-------------|
| A | Reference has **support overlap** with what policy can reach early | Micro-train smoke test stable |
| B | Discriminator cannot win by **exploiting artifacts** | Harness metrics within caps |
| C | Reference **coverage** is enough to prevent overfitting | ≥60s Tier 0+ data |

These are measurable. The plan focuses on making them true.

---

## Minimum Viable Dataset Targets

| Metric | Minimum | Target | Optimal |
|--------|---------|--------|---------|
| **Tier 0+ Duration** | 30s | 60s | 60-180s |
| **Segment Length** | 0.4s (20 frames) | 0.5s | 1.0s |
| **Source Clip Diversity** | 3 clips | 5 clips | 8+ clips |

Diversity prevents trivial overfitting to a single motion pattern.

---

## Tier 0+ Gates (Prescriptive)

Evaluate **per segment window** of length 0.4–1.0s. These are segment-based, not clip-based.

### 1. Load Support
| Metric | Gate | Rationale |
|--------|------|-----------|
| mean(ΣFn)/mg | ≥ **0.85** | Feet carry majority of weight |
| p10(ΣFn)/mg | ≥ **0.60** | Prevents "mostly loaded except brief collapse" |

### 2. Harness Reliance
| Metric | Gate | Rationale |
|--------|------|-----------|
| p90(\|F_stab\|)/mg | ≤ **0.15** | Harness is assist, not support |
| p95(\|F_stab\|)/mg | ≤ **0.25** | Allow transients |
| Saturated steps at cap | ≤ **5%** | Harness not constantly maxed |

### 3. Unloaded Frames
| Metric | Gate | Rationale |
|--------|------|-----------|
| Frames with ΣFn/mg < 0.60 | ≤ **10%** | Rare floating frames |

### 4. Attitude Sanity
| Metric | Gate | Rationale |
|--------|------|-----------|
| p95(\|pitch\|) | ≤ **25°** | Not falling forward/back |
| p95(\|roll\|) | ≤ **25°** | Not falling sideways |

### 5. Slip Sanity (NEW - when loaded)
| Metric | Gate | Rationale |
|--------|------|-----------|
| Loaded foot definition | Fn_foot > 0.10 mg | Clear contact |
| Stance slip rate | ≤ **15%** | \|v_foot_tangent\| > 0.10 m/s when loaded |

---

## Reference Generation Loop (Looping + Blending)

### Why Looping Works
Most Tier 0+ windows occur in "stable" regions. Repetition increases chances of mining those windows **without adding new mocap**.

### Generation Protocol

For each source motion clip of length T (~3 seconds):

```
┌─────────────────────────────────────────────────────────────┐
│  Generate N cycles (20-60 seconds total per source clip)   │
│                                                             │
│  [Clip] → [Blend 0.2-0.4s] → [Clip] → [Blend] → [Clip]...  │
│                                                             │
│  Blend = linear interpolate to neutral standing pose        │
│  Prevents discontinuities at loop boundaries                │
└─────────────────────────────────────────────────────────────┘
```

### PD Tracking Protocol (No qpos Overwrite)

At each control step (50 Hz):
```python
# Set joint position targets = reference joint angles
mj_data.ctrl[:n] = ref_dof_pos[frame_idx]

# Optionally set velocity targets = 0 (or use damping only)
# mj_data.ctrl[n:2*n] = 0  # If using velocity control

# Step MuJoCo with 10 substeps (500 Hz physics)
for _ in range(10):
    mujoco.mj_step(mj_model, mj_data)
```

**Critical**: Do NOT overwrite qpos every frame. Let physics evolve.

### Recording at 50 Hz

| Data | Purpose |
|------|---------|
| qpos (root + joints) | State features |
| Foot contacts and per-foot Fn | Contact features |
| Per-foot tangential speed | Slip gating |
| Stabilizer force/torque | Harness metrics |

---

## Segment Miner: Windowing and Scoring

### Windowing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size W | **50 frames (1.0s)** | Long enough for gait cycle |
| Stride S | **10 frames (0.2s)** | Overlap to catch all good windows |
| Min segment | **20 frames (0.4s)** | Minimum usable length |

### Deduplication Strategy

When windows overlap:
- Take center 20-30 frames of each passing window, OR
- Dedupe by hashing features

### Quality Scoring (for Sampling Weights)

Even within Tier 0+, weight better windows higher:

```python
score = 0
if mean_load_support >= 0.95:
    score += 1
if slip_rate <= 0.05:
    score += 1
if p95_pitch <= 15 and p95_roll <= 15:
    score += 1
if harness_p95 close to limit:
    score -= 1

# Use score as sampling weight for reference minibatches
```

### Output Format

```python
{
    "features": np.array,           # Concatenated frames from accepted windows
    "motion_id": List[str],         # Source clip identifiers
    "segment_id": List[int],        # Segment index within clip
    "segment_scores": List[float],  # Quality scores for weighted sampling
    "per_segment_metrics": List[Dict],
    "feature_version": str,         # For reproducibility
}
```

---

## Stage 0 Discriminator Features (Feature Dropping)

### The Shortcut Problem

Most shortcut-prone channels in current feature vector:
1. **Contacts (4)** - Discrete, easy to exploit
2. **Root height (1)** - Policy height != reference height early
3. **Velocity magnitude** - If reference/policy differ systematically

### Stage 0 Feature Set (Recommended for AMP from Scratch)

**KEEP:**
| Feature | Dim | Rationale |
|---------|-----|-----------|
| Joint positions | 8 | Core pose information |
| Joint velocities | 8 | Dynamics |
| Root angular velocity | 3 | Orientation dynamics |
| Root linear velocity (modified) | 3 | See below |

**DROP TEMPORARILY:**
| Feature | Dim | Rationale |
|---------|-----|-----------|
| Foot contacts | 4 | Too discrete, shortcut-prone |
| Root height | 1 | Policy/reference mismatch early |

**VELOCITY MODIFICATION OPTIONS:**
```python
# Option A: Direction only (normalize)
root_lin_vel_normalized = root_lin_vel / (np.linalg.norm(root_lin_vel) + 1e-6)

# Option B: Clipped magnitude
root_lin_vel_clipped = np.clip(root_lin_vel, -1.5, 1.5)  # 0-1.5 m/s band
```

This prevents discriminator from learning "policy is slower" as trivial classifier.

### Feature Reintroduction Schedule

```
Phase 0 (Steps 0-50K):
  Features: joint_pos, joint_vel, root_ang_vel, root_lin_vel_clipped
  Dropped: contacts, root_height

Phase 1 (Steps 50K-100K):
  Add back: root_height
  Trigger: Policy walks stably, falls rare

Phase 2 (Steps 100K+):
  Add back: contacts (or continuous proxies first)
  Trigger: Stable walking maintained after height added
```

**Important**: Dropping features is NOT cheating. It is standard adversarial training hygiene to prevent early shortcut learning.

---

## Training Schedule Adjustments

### Reference Sampling Strategy

```python
# Sample with probability proportional to segment score
probs = softmax(segment_scores / temperature)

# Ensure each minibatch includes multiple motion IDs
# to avoid single-clip overfit
assert len(set(batch_motion_ids)) >= min(3, num_unique_clips)
```

### AMP Weight Schedule (Ramp-Up)

```python
# Start low, ramp only after discriminator stops saturating
amp_weight_schedule = {
    "initial": 0.05,      # 5% of final AMP weight
    "ramp_start": 5000,   # Start ramping at step 5K
    "ramp_end": 50000,    # Full weight by step 50K
    "final": 1.0,
}

# Ramp condition checks:
# - Discriminator AUC stays between 0.60 and 0.85
# - Logits are not saturated (check max logit magnitude)
```

### Discriminator Regularization

| Technique | Implementation | Status |
|-----------|----------------|--------|
| Input noise | Add small noise to disc inputs | ⏳ TODO |
| Replay buffer | Store old (state, label) pairs | ⏳ TODO |
| Gradient penalty | R1 or GP regularization | ✅ Implemented |
| Early stopping | If disc_acc extreme and stuck | Monitor |

**The biggest stabilizer**: Feature dropping + Tier 0+ mining > regularization tricks.

---

## Actuator Saturation Check (Critical Pre-Requisite)

Your actuator force range is **±4 N**. If actuators frequently saturate during tracking, you will NOT get Tier 0+ seconds regardless of mining.

### Diagnostic

```bash
# Check saturation during reference generation
uv run python scripts/generate_tier0plus_segments.py --verbose 2>&1 | grep "saturated"
```

### If Saturation > 10%

Options (within current tools):
1. **Time warp slower** - Reduce motion speed to 0.8-0.9x
2. **Scale forward velocity** - Reduce step length in reference
3. **Increase forcerange** - If physically plausible for WildRobot

```yaml
# In robot XML
<actuator>
  <general name="..." forcerange="-6 6"/>  # Try ±6 N if ±4 N saturates
</actuator>
```

---

## The Three Blockers and Their Solutions

### Blocker A: Small Dataset (~15 seconds Tier 0+ currently)

**Root Cause**: Only extracting the longest segment, not all valid segments.

**Solution**: Aggressive segment mining with multi-start rollouts.

| Action | Implementation | Status |
|--------|----------------|--------|
| Multi-start rollouts | `--multi-start-interval 50` in `generate_tier0plus_segments.py` | ✅ Implemented |
| Time warp augmentation | 0.9–1.1x speed during reference generation | ⏳ TODO |
| Left-right mirroring | If morphology symmetric | ⏳ TODO |
| Heading variations | Small δ during generation | ⏳ TODO |
| Mine ALL segments | Extract every valid window, not just longest | ✅ Implemented |

**Exit Criterion**: 
```
✓ ≥60 seconds Tier 0+ data (optimal: 60-180s)
⚠ 30-60 seconds: borderline, can attempt with caution
✗ <30 seconds: STOP, switch to baseline PPO
```

### Blocker B: Harness Dependence Issues

**Root Cause**: Harness carries weight or forces contact patterns that don't match learnable policy behavior.

**Solution**: Cap and log, plus controller redesign.

| Action | Implementation | Current Value | Target |
|--------|----------------|---------------|--------|
| p95 harness cap | Tier 0+ gate | **≤10% mg** | ≤10% mg |
| p90 harness cap | Tier 0+ gate | ≤15% mg | ≤15% mg |
| Mean harness | Logging | -- | ≤5% mg |
| Orientation-gated harness | Only when upright (<25°) | ✅ Implemented | -- |
| Low-frequency harness | Filter impulses | ⏳ TODO | -- |

**Physical Redesign (Recommended)**:
```yaml
# Replace height spring with orientation stabilizer
# Current: F_z = Kp * (z_target - z) - Kd * z_dot  # Most artifact-prone
# Better: τ_orient = Kp * (roll_target, pitch_target) - Kd * ω  # Helps balance
```

**Exit Criterion**:
```
In Tier 0+ segments:
✓ mean(ΣFn)/mg ≥ 0.85 (feet carry the weight)
✓ unloaded_frames ≤ 10%
✓ p95(|F_stab|)/mg ≤ 0.10
```

### Blocker C: Contact Ambiguity (Double Stance Dominated)

**Root Cause**: Retargeted reference has degenerate contact structure (mostly double stance).

**Solution**: Initially soften contact features, improve swing emergence.

| Action | Implementation | Status |
|--------|----------------|--------|
| **Remove contact features from discriminator initially** | Config option `use_contact_features: false` | ⏳ TODO |
| Use continuous contact proxies | foot_height, foot_z_velocity | ⏳ TODO |
| Swing clearance term in tracker | Minimum clearance during swing windows | ⏳ TODO |
| Phase heuristic for swing detection | From kinematic foot height/velocity | ⏳ TODO |
| Data selection by alternation score | Prefer segments with single_stance ≥15% | ⏳ TODO |

**Exit Criterion**:
```
Tier 0+ dataset has meaningful subset with:
✓ alternation_score ≥ 15% (some frames with exactly one foot down)
✓ Consistent stepping pattern in at least some segments
```

---

## Phased Implementation Roadmap

### Phase 1: Generate Tier 0+ Segments at Scale

**Goal**: Produce ≥60 seconds of Tier 0+ data.

**Actions**:
1. Run segment mining with multi-start:
   ```bash
   uv run python scripts/generate_tier0plus_segments.py \
       --multi-start-interval 50 \
       --max-window-frames 200 \
       --harness-cap 0.15 \
       --verbose
   ```

2. Analyze results:
   ```bash
   uv run python scripts/analyze_tier0plus_dataset.py --verbose
   ```

**Gate**: Total Tier 0+ seconds ≥ 60

**If FAIL**: 
- Try relaxing thresholds slightly (e.g., `--max-harness-p95 0.12`)
- If still <30s, proceed to Phase 4 (Stop Condition)

### Phase 2: Validate Reference Manifold

**Goal**: Ensure discriminator can learn without exploiting artifacts.

**Actions**:
1. Remove contact features (or soften):
   ```yaml
   # In training config
   discriminator:
     use_contact_features: false  # Initially disable
     # OR use continuous proxies:
     use_contact_proxies: true
     contact_proxy_type: "foot_height"  # Not binary contact
   ```

2. Run micro-train smoke test:
   ```bash
   uv run python scripts/micro_train_smoke_test.py \
       --dataset playground_amp/data/tier0plus/tier0plus_merged.pkl \
       --steps 5000 \
       --log-interval 100
   ```

**Gate**:
- Discriminator AUC between 0.6-0.9 (not saturated, not random)
- AMP reward variance > 0.05 (policy can explore reward landscape)
- No NaN/Inf in features

**If FAIL**:
- Check for remaining harness artifacts
- Verify feature normalization
- If discriminator saturates immediately → reference too narrow or artifacts present

### Phase 3: Add Contacts Back Gradually

**Goal**: Once stable walking emerges, reintroduce contact structure.

**Actions**:
1. Train initial policy without contact features (Phase 2 config)
2. Once policy shows walking behavior (~50K-100K steps):
   - Add contact features with low weight
   - Or use soft contact proxies

**Gate**:
- Policy maintains walking after contact features added
- No regression in reward metrics

### Phase 4: STOP CONDITION (Critical Decision Point)

**If after implementing**:
- Segment mining with multi-start
- Harness caps and percentile gates
- Contact feature removal/softening

**You still cannot produce**:
- ≥60 seconds Tier 0+ data, AND
- Micro-train remains unstable

**Then**:
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   AMP FROM SCRATCH IS NOT THE EFFICIENT PATH                │
│                                                             │
│   Switch to: Baseline PPO Walking (Path B)                  │
│                                                             │
│   This is NOT a workaround. It is the correct engineering   │
│   choice given current robot actuation and motion sources.  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Path B: Baseline PPO Walking First**
1. Train with task rewards only:
   - Velocity tracking reward
   - Stability reward (orientation, base angular velocity)
   - Energy minimization
   - Foot clearance during swing
   
2. Once stable walking achieved:
   - Collect rollouts from policy
   - Add AMP with Tier 0+ references at **low weight** (w_amp = 0.1)
   - Gradually ramp up AMP weight

---

## Quick Reference: Scripts and Commands

### Segment Generation
```bash
# Standard Tier 0+ generation
uv run python scripts/generate_tier0plus_segments.py --verbose

# With multi-start for more coverage
uv run python scripts/generate_tier0plus_segments.py \
    --multi-start-interval 50 \
    --max-window-frames 200 \
    --verbose

# Relaxed thresholds (if borderline)
uv run python scripts/generate_tier0plus_segments.py \
    --max-harness-p95 0.12 \
    --min-load-support 0.85 \
    --verbose
```

### Dataset Analysis
```bash
# Full analysis with go/no-go decision
uv run python scripts/analyze_tier0plus_dataset.py --verbose

# Save analysis to JSON
uv run python scripts/analyze_tier0plus_dataset.py \
    --json-output playground_amp/data/tier0plus/analysis.json
```

### Harness Testing
```bash
# Test all clips
uv run python scripts/test_harness_contribution.py --verbose

# Test single clip
uv run python scripts/test_harness_contribution.py \
    --motion assets/motions/walking_medium01.pkl
```

### Micro-Train Smoke Test
```bash
uv run python scripts/micro_train_smoke_test.py \
    --dataset playground_amp/data/tier0plus/tier0plus_merged.pkl \
    --steps 5000
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                    START: Generate Tier 0+                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Total T0+ >= 60s?    │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
          YES ▼                   NO  ▼
    ┌─────────────────┐     ┌─────────────────┐
    │ Phase 2:        │     │ Total >= 30s?   │
    │ Micro-train     │     └────────┬────────┘
    │ smoke test      │              │
    └────────┬────────┘     ┌────────┴────────┐
             │              │                 │
             ▼          YES ▼             NO  ▼
    ┌─────────────────┐  ┌──────────┐   ┌──────────────┐
    │ AUC 0.6-0.9?    │  │Try with  │   │ STOP         │
    │ Reward var>0.05?│  │caution   │   │ → Path B     │
    └────────┬────────┘  └────┬─────┘   │   Baseline   │
             │                │         │   PPO        │
    ┌────────┴────────┐       │         └──────────────┘
    │                 │       │
YES ▼             NO  ▼       ▼
┌─────────────┐  ┌─────────────────────┐
│ Phase 3:    │  │ Debug artifacts,    │
│ Add contacts│  │ check harness,      │
│ gradually   │  │ try threshold relax │
└─────────────┘  └─────────────────────┘
```

---

## Metrics Dashboard (Target Values)

| Metric | Minimum | Target | Measured |
|--------|---------|--------|----------|
| **Tier 0+ Duration** | 30s | ≥60s | ___ s |
| **Tier 0+ Segments** | 5 | ≥10 | ___ |
| **mean(ΣFn)/mg** | 0.85 | ≥0.90 | ___% |
| **p95(harness)/mg** | ≤0.10 | ≤0.08 | ___% |
| **Unloaded frame rate** | ≤10% | ≤5% | ___% |
| **Alternation rate** | ≥10% | ≥20% | ___% |
| **Discriminator AUC** | 0.6 | 0.7-0.8 | ___ |
| **AMP reward variance** | 0.05 | 0.1 | ___ |

---

## Appendix: Contact Feature Softening Options

### Option 1: Remove Contact Features
```python
# In AMP feature config
contact_dim: 0  # No contact features
```

### Option 2: Continuous Proxies
```python
# Replace binary contact with:
foot_height_left   # Continuous, easy to learn
foot_height_right
foot_z_velocity_left  # Non-zero during swing
foot_z_velocity_right
foot_speed_near_ground  # Smooth transition
```

### Option 3: Delayed Contact Introduction
```python
# Training schedule
phase_1:  # Steps 0-50K
  contact_weight: 0.0
  
phase_2:  # Steps 50K-100K
  contact_weight: 0.3
  
phase_3:  # Steps 100K+
  contact_weight: 1.0
```

---

## Concrete "Do This Next" Checklist

### Step 1: Expand Data with Looping + Neutral Blending

For each of your 12 clips, generate 30 seconds of physics rollout:
- [ ] Implement clip looping with neutral blend transitions (0.2-0.4s)
- [ ] Record contact forces and slip signals at 50 Hz
- [ ] Check actuator saturation rate (must be <10%)

```bash
# Run with extended rollouts (looping support TBD)
uv run python scripts/generate_tier0plus_segments.py \
    --multi-start-interval 50 \
    --max-window-frames 200 \
    --verbose
```

### Step 2: Run Segment Miner with Tier 0+ Gates

- [ ] Generate Tier 0+ dataset with all gates applied
- [ ] Check total Tier 0+ seconds (target: ≥60s)

**Report these metrics:**
```
□ Total Tier 0+ seconds: ___s
□ Distribution of ΣFn/mg: mean=___, p10=___, p95=___
□ Harness p90=___%, p95=___%
□ Unloaded frame fraction: ___%
□ Slip rate distribution: mean=___%, p95=___%
□ Number of source clips contributing: ___
```

### Step 3: Train Discriminator with Reduced Feature Set

- [ ] Configure discriminator to drop contacts and root height
- [ ] Run micro-train smoke test

```bash
uv run python scripts/micro_train_smoke_test.py \
    --dataset playground_amp/data/tier0plus/tier0plus_merged.pkl \
    --steps 5000 \
    --drop-contacts \
    --drop-height
```

**Check:**
```
□ AUC between 0.60-0.85: ___
□ Logits not saturated: ___
□ AMP reward variance > 0.05: ___
```

### Step 4: Reintroduce Height, Then Contacts

- [ ] Add root height back once walking is stable
- [ ] Add contacts back (or continuous proxies) after height integration works

---

## Expected Outcome

If underlying issue is "clips are mostly bad but contain good moments":

| Scenario | Current | After Recipe |
|----------|---------|--------------|
| Tier 0+ seconds | ~14.7s | **60-200s** |
| Source clips contributing | 3-5 | 8-12 |

This is achievable through looping and mining **without new motion sources**.

If you still cannot reach ~60 seconds after this recipe, that is a **strong signal** the tracking controller and robot actuation envelope are too far from motion intent → pivot to baseline PPO.

---

## Diagnostic Log Template

When running reference generation, capture and report:

```
========================================
REFERENCE GENERATION DIAGNOSTIC
========================================
Clip: ___
Duration: ___ seconds
Actuator saturation rate: ___%  (MUST BE <10%)

Load Support:
  mean(ΣFn)/mg: ___%
  p10(ΣFn)/mg: ___%
  p95(ΣFn)/mg: ___%

Harness Reliance:
  mean(|F_stab|)/mg: ___%
  p90(|F_stab|)/mg: ___%
  p95(|F_stab|)/mg: ___%
  saturated_steps: ___%

Attitude:
  p95(|pitch|): ___°
  p95(|roll|): ___°

Slip (when loaded):
  stance_slip_rate: ___%

VERDICT: [PASS/FAIL] Tier 0+
Failure reasons (if any): ___
========================================
```

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2025-12-25 | Added prescriptive parameters: Tier 0+ gates (p10, slip), looping+blending protocol, windowing params (W=50, S=10), Stage 0 feature dropping (contacts, height), AMP weight ramp schedule, actuator saturation check, concrete checklist |
| 1.0.0 | 2025-12-25 | Initial data-first plan with hard gates and stop conditions |
