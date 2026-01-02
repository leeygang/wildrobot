# AMP Training Changelog
# CHANGELOG

This changelog tracks capability changes, configuration updates, and training results for the WildRobot AMP training pipeline.

**Ordering:** Newest releases first (reverse chronological order).

---

## [v0.11.4] - 2026-01-01: Walking Conservative Warm Start (v0.11.4)

### Plan
1. Validate assets + config:
   `uv run python scripts/validate_training_setup.py`
2. Walking warm-start (conservative range):
   `uv run python playground_amp/train.py --config playground_amp/configs/ppo_walking_conservative.yaml --no-amp --verify --resume playground_amp/checkpoints/ppo_standing_v00113_final.pkl`
3. Full run:
   `uv run python playground_amp/train.py --config playground_amp/configs/ppo_walking_conservative.yaml --no-amp --resume playground_amp/checkpoints/ppo_standing_v00113_final.pkl`

### Base Checkpoint
- `playground_amp/checkpoints/ppo_standing_v00113_final.pkl` (v0.11.3 standing best @ iter 310)

### Results (Walking PPO Conservative, v0.11.4)
- Run: `playground_amp/wandb/run-20260101_215853-5e15gkam`
- Checkpoints: `playground_amp/checkpoints/ppo_walking_conservative_v00114_20260101_215854-5e15gkam/`
- Best checkpoint (reward): `checkpoint_430_56360960.pkl` (reward=440.31, vel=0.31 @ cmd=0.35, ep_len=466.4)
- Topline (final @610): reward=431.74, ep_len=474.6, success=88.9%, vel=0.32 @ cmd=0.35, vel_err=0.079, max_torque=0.869
- Notes: Good conservative walking found; next run should continue v0.11.4 (no config changes) from best checkpoint to push success rate toward 95%+

### Next
- Continue v0.11.4 from best:
  `uv run python playground_amp/train.py --config playground_amp/configs/ppo_walking_conservative.yaml --no-amp --resume playground_amp/checkpoints/ppo_walking_conservative_v00114_20260101_215854-5e15gkam/checkpoint_430_56360960.pkl`

---

## [v0.11.3] - 2025-12-31: Foot Switches + Standing Retrain Plan

### Plan
1. `uv run python playground_amp/train.py --config playground_amp/configs/ppo_standing.yaml --no-amp --verify`
2. `uv run python playground_amp/training/visualize_policy.py --headless --num-episodes 1 --config playground_amp/configs/ppo_standing.yaml --checkpoint <path>`
3. Resume walking with new standing checkpoint:
   `uv run python playground_amp/train.py --config playground_amp/configs/ppo_walking_conservative.yaml --no-amp --verify --resume <new_standing_checkpoint>`

### Config Updates
- `min_height`: 0.20 → 0.40 (terminate squat posture)
- `reward_weights.base_height`: 1.0 → 3.0 (enforce upright height)
- `reward_weights.orientation`: -1.0 → -2.0 (stricter tilt penalty)
- Added standing push disturbances (random lateral force, short duration)
- Switched foot contact signals to boolean foot switches (toe/heel), obs dim now 39
- Removed `*_touch` sites/sensors from `assets/wildrobot.xml`; switch signal derived from geom forces
- Added height target shaping + stance width penalty to discourage squat posture

### Results (Standing PPO, v0.11.3)
- Run: `playground_amp/wandb/run-20260101_182859-2wsk4xt7`
- Resumed from: `checkpoint_260_34078720.pkl` (reward=518.32, ep_len=498)
- Best checkpoint: `playground_amp/checkpoints/ppo_standing_v00113_20260101_182901-2wsk4xt7/checkpoint_310_40632320.pkl`
- Final checkpoint: `playground_amp/checkpoints/ppo_standing_v00113_20260101_182901-2wsk4xt7/checkpoint_360_47185920.pkl`
- Summary (best @310): reward=574.31, ep_len=500, height=0.454m, vel=-0.02m/s (cmd=0.00), vel_err=0.076, torque~84.9%
- Terminations: 0% across all categories at best window; stable 500-step episodes
- Notes: Ready to warm-start walking with conservative velocity range

---

## [v0.11.2] - 2025-12-31: Upright Standing Retrain Plan

### Plan
1. `uv run python playground_amp/train.py --config playground_amp/configs/ppo_standing.yaml --no-amp --verify`
2. `uv run python playground_amp/training/visualize_policy.py --headless --num-episodes 1 --config playground_amp/configs/ppo_standing.yaml --checkpoint <path>`
3. Resume walking with new standing checkpoint:
   `uv run python playground_amp/train.py --config playground_amp/configs/ppo_walking_conservative.yaml --no-amp --verify --resume <new_standing_checkpoint>`

### Config Updates
- `min_height`: 0.20 → 0.40 (terminate squat posture)
- `reward_weights.base_height`: 1.0 → 3.0 (enforce upright height)
- `reward_weights.orientation`: -1.0 → -2.0 (stricter tilt penalty)
- Added standing push disturbances (random lateral force, short duration)

---

## [v0.11.1] - 2025-12-31: CAL PPO Walking Smoke Test

### Test Plan
1. `uv run python playground_amp/train.py --config playground_amp/configs/ppo_walking_conservative.yaml --no-amp --verify --resume playground_amp/checkpoints/ppo_standing_v0110_170.pkl`
2. `uv run python playground_amp/training/visualize_policy.py --headless --num-episodes 1 --config playground_amp/configs/ppo_walking.yaml --checkpoint <path>`

### Results
- Run: `playground_amp/wandb/run-20251231_003248-8p2bv838`
- Best checkpoint: `playground_amp/checkpoints/ppo_walking_conservative_v00111_20251231_003249-8p2bv838/checkpoint_360_47185920.pkl`
- Summary: reward ~426.79, ep_len ~491, success ~95.4%, vel ~0.33 m/s
- Notes: stable but squat-biased posture inherited from standing checkpoint; hip swing limited

---

## [v0.11.0] - 2025-12-30: Control Abstraction Layer (CAL) - BREAKING CHANGE

### Overview

Major architectural change introducing a **Control Abstraction Layer (CAL)** to decouple high-level flows (training, testing, sim2real) from MuJoCo primitives.

⚠️ **BREAKING CHANGE**: This version requires retraining from scratch. Existing checkpoints are incompatible due to action semantics changes.

### Why CAL?

The previous implementation had critical issues:
1. **Direct MuJoCo Primitive Access**: Multiple layers directly manipulated `data.ctrl`, `data.qpos`, etc.
2. **Ambiguous Actuator Semantics**: Policy outputs [-1, 1] were passed directly to MuJoCo ctrl (which expects radians)
3. **Incorrect Symmetry Assumptions**: Left/right hip joints have mirrored ranges, but code didn't account for this
4. **Training Learning Simulator Quirks**: Policies learned MuJoCo-specific behaviors instead of transferable motor skills

### New Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HIGH-LEVEL LAYER                           │
│  Training Pipeline │ Test Suite │ Sim2Real │ Policy Inference   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTROL ABSTRACTION LAYER (CAL)                    │
│                                                                 │
│  • policy_action_to_ctrl() - normalized [-1,1] → radians        │
│  • physical_angles_to_ctrl() - radians → ctrl (no normalization)│
│  • get_normalized_joint_positions() - for policy observations   │
│  • get_physical_joint_positions() - for logging/debugging       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LOW-LEVEL LAYER (MuJoCo)                      │
│                  data.ctrl, data.qpos, data.qvel                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

#### Two-Method API Pattern
- `policy_action_to_ctrl()`: For policy outputs [-1, 1], always applies symmetry correction
- `physical_angles_to_ctrl()`: For GMR/mocap angles (radians), no symmetry correction

#### Symmetry Correction
- `mirror_sign` field handles joints with inverted ranges (e.g., left/right hip pitch)
- Automatic correction ensures symmetric actions produce symmetric motion

#### Sim2Real Ready
- `ControlCommand` dataclass with target_position and target_velocity
- `VelocityProfile` enum for trajectory generation (STEP, LINEAR, TRAPEZOIDAL, S_CURVE)
- Clean separation between simulation and deployment

#### Default Positions from MuJoCo
- Home pose loaded from MuJoCo keyframe at runtime (single source of truth)
- Recommendation: Use `keyframes.xml` for dedicated keyframe management

### New Files

| File | Description |
|------|-------------|
| `playground_amp/control/__init__.py` | Export CAL, specs, types |
| `playground_amp/control/cal.py` | ControlAbstractionLayer class |
| `playground_amp/control/specs.py` | JointSpec, ActuatorSpec, ControlCommand |
| `playground_amp/control/types.py` | ActuatorType, VelocityProfile enums |
| `playground_amp/tests/test_cal.py` | CAL unit tests |
| `assets/keyframes.xml` | Dedicated keyframe file (home, t_pose) |

### Files Updated

| File | Change |
|------|--------|
| `playground_amp/envs/wildrobot_env.py` | Integrate CAL |
| `playground_amp/configs/robot_config.py` | Add actuated_joints support |
| `assets/robot_config.yaml` | Add actuated_joints section |
| `assets/post_process.py` | Add generate_actuated_joints_config() |

### Config Schema Update

New `actuated_joints` section in `robot_config.yaml`:

```yaml
actuated_joints:
  - name: left_hip_pitch
    type: position
    class: htd45hServo
    range: [-0.087, 1.571]
    symmetry_pair: right_hip_pitch
    mirror_sign: 1.0
    max_velocity: 10.0
    # NOTE: default_pos loaded from MuJoCo keyframe at runtime
```

### Migration

1. **Retrain from scratch** - existing checkpoints are incompatible
2. Update `robot_config.yaml` with `actuated_joints` section
3. Create `assets/keyframes.xml` with home pose
4. Replace direct MuJoCo access with CAL methods

### Documentation

See `playground_amp/docs/CONTROL_ABSTRACTION_LAYER_PROPOSAL.md` for full design document.

### Results (Standing PPO, v0.11.0)
- Config: `playground_amp/configs/ppo_standing.yaml`
- Run: `playground_amp/wandb/run-20251230_181458-27832na7`
- Best checkpoint: `playground_amp/checkpoints/ppo_standing_v00110_20251230_181500-27832na7/checkpoint_170_22282240.pkl`
- Summary: ep_len ~484, height ~0.434 m, success ~89.8%, vel ~0.01 m/s
- Notes: success rate below 95% target; height_low terminations ~10% late in training


## [v0.10.6] - 2025-12-28: Hip/Knee Swing Reward (Stage 1)

### Config Updates
- Added swing-gated hip/knee rewards to encourage leg articulation during swing.
- Added small flight-phase penalty to discourage hopping.
- Kept gait periodicity reward (alternating support).
- Maintained v0.10.5 tuning for velocity/orientation/torque.

### Base Checkpoint (v0.10.5)
- `playground_amp/checkpoints/wildrobot_ppo_20251228_205536/checkpoint_520_68157440.pkl`

### Training Results (v0.10.5 → v0.10.6 prep)
- v0.10.5 best: reward ~566.42 (iter 520), vel ~0.63 m/s, ep_len ~452, torque ~97%.
- Gait analysis: alternating support ~79–80%, flight phase ~18%, knees used ~22–27% of range.
- Interpretation: ankle-dominant shuffle persists; add swing-gated knee/hip rewards + flight penalty.

### Files Updated
- `playground_amp/configs/ppo_walking.yaml`
- `playground_amp/envs/wildrobot_env.py`
- `playground_amp/configs/training_config.py`
- `playground_amp/configs/training_runtime_config.py`
- `playground_amp/training/metrics_registry.py`

---

## [v0.10.5] - 2025-12-28: Gait Shaping Tune (Stage 1)

### Config Updates
- Reduced velocity dominance to allow gait shaping.
- Increased orientation + torque/saturation penalties to curb forward-lean shuffle.
- Added gait periodicity reward (alternating support).
- Shortened default training run to 400 iterations.

### Notes
- Resume-ready PPO run (best at iter 520) still ankle-dominant; used as base for v0.10.6.

---

## [v0.10.4] - 2025-12-28: Velocity Incentive (Stage 1)

### Config Updates
- Strong velocity tracking incentive to escape standing-still local minimum.
- Reduced stability penalties to allow forward motion discovery.

### Results Summary
- Tracking met (vel_err ~0.085, ep_len ~460) but peak torque ~0.98 and forward-lean shuffle.

---

## [v0.10.3] - 2025-12-27: Forward Walking (Stage 1)

### Config Updates
- PPO walking config aligned to 0.5–1.0 m/s command range.
- Enabled velocity tracking metrics and exit criteria logging.

---

## [v0.10.2] - 2025-12-27: Basic Standing (Stage 1)

### Updates
- Added termination diagnostics and reset preservation for accurate logging.
- Standing training readiness (stability-focused PPO).

---

## [v0.10.1] - 2025-12-26: Config Schema Migration

### Updates
- Migrated to unified `TrainingConfig` with Freezable runtime config for JIT.
- Simplified config loading and CLI override flow.

---

## [v0.8.0] - 2024-12-26: Feature Set Refactoring

### Major Change: Single Source of Truth Architecture

Complete refactoring of AMP feature configuration and extraction to establish clear separation of concerns and a single source of truth for feature layout.

### Feature Dropping (Prevent Discriminator Shortcuts)

Added v0.8.0 feature flags to prevent discriminator from exploiting artifact-prone features:

| Flag | Effect | Rationale |
|------|--------|-----------|
| `drop_contacts` | Drop foot contacts (4 dims) | Discrete values, easy to exploit |
| `drop_height` | Drop root height (1 dim) | Policy/reference mismatch early in training |
| `normalize_velocity` | Normalize root_linvel to unit direction | Remove speed as discriminative signal |

```yaml
# ppo_amass_training.yaml
amp:
  feature:
    drop_contacts: false
    drop_height: false
    normalize_velocity: false
```

### Architecture Refactoring

**New Structure:**
```
playground_amp/
├── amp/
│   ├── policy_features.py     # Online extraction (JAX) + extract_amp_features_batched
│   └── ref_features.py        # Offline extraction (NumPy) + load_reference_features
├── configs/
│   ├── feature_config.py      # FeatureConfig + _FeatureLayout (SINGLE SOURCE OF TRUTH)
│   ├── training_config.py     # TrainingConfig + RobotConfig
│   └── training_runtime_config.py  # TrainingRuntimeConfig (JIT)
└── training/
    └── trainer_jit.py         # Training loop with normalize_features (training-specific)
```

#### Files Created
| File | Description |
|------|-------------|
| `configs/training_runtime_config.py` | `TrainingRuntimeConfig` class (renamed from `AMPPPOConfigJit`) |

#### Files Moved
| From | To |
|------|-----|
| `amp/feature_config.py` | `configs/feature_config.py` |

#### Files Deleted
| File | Reason |
|------|--------|
| `amp/amp_features.py` | Superseded by `policy_features.py` |
| `common/preproc.py` | Functionality moved to `policy_features.py` and `ref_features.py` |

### Key Changes

#### `_FeatureLayout` Class (Single Source of Truth)
```python
class _FeatureLayout:
    """Centralized feature layout definition."""

    COMPONENT_DEFS = [
        ("joint_pos", "num_joints", False),      # not droppable
        ("joint_vel", "num_joints", False),      # not droppable
        ("root_linvel", "ROOT_LINVEL_DIM", False),
        ("root_angvel", "ROOT_ANGVEL_DIM", False),
        ("root_height", "ROOT_HEIGHT_DIM", True),   # droppable
        ("foot_contacts", "FOOT_CONTACTS_DIM", True), # droppable
    ]
```

#### `TrainingConfig.to_runtime_config()` Method
```python
# Only way to create TrainingRuntimeConfig
runtime_config = training_cfg.to_runtime_config()
```

#### `load_reference_features()` Function
```python
# Consolidated reference data loading in ref_features.py
from playground_amp.amp.ref_features import load_reference_features
ref_features = load_reference_features(args.amp_data)
```

#### CLI Overrides Applied to TrainingConfig
```python
# CLI parameters overwrite training_cfg BEFORE to_runtime_config()
if args.iterations is not None:
    training_cfg.iterations = args.iterations
# ... other CLI overrides
config = training_cfg.to_runtime_config()
```

### Removed Code

| Removed | Reason |
|---------|--------|
| `mask_waist` config/code | No waist joint in robot |
| `velocity_filter_alpha` from runtime | Not used in training |
| `ankle_offset` from runtime | Not used in training |
| `RunningMeanStd` class | Only used in tests |
| `create_running_stats()` | Only used in tests |
| `update_running_stats()` | Only used in tests |
| `normalize_features()` in policy_features.py | Duplicate, kept in trainer_jit.py |

### Design Principles Established

1. **Feature extraction** (`policy_features.py`, `ref_features.py`) - extracts raw features
2. **Training normalization** (`trainer_jit.py`) - statistical normalization for training
3. **Single source of truth** - `_FeatureLayout.COMPONENT_DEFS` defines feature ordering
4. **Config-driven** - All drop flags read from `TrainingConfig`

### Config
```yaml
version: "0.8.0"
version_name: "Feature Set Refactoring"
```

### Migration

If upgrading from v0.7.0:
1. Update imports from `amp.feature_config` → `configs.feature_config`
2. Update imports from `amp.amp_features` → `amp.policy_features`
3. Replace `AMPPPOConfigJit` with `TrainingRuntimeConfig`
4. Use `training_cfg.to_runtime_config()` instead of direct instantiation
5. Remove any references to `mask_waist`, `velocity_filter_alpha`, `ankle_offset` in runtime code

### Status
✅ Complete - architecture refactored with single source of truth

---

## [v0.7.0] - 2024-12-25: Physics Reference Data

### Major Change: Physics-Generated Reference Data

Switched from GMR retargeted motions to physics-realized reference data.

**Why**: GMR motions are dynamically infeasible (robot falls immediately). Physics rollout ensures reference data is achievable by the robot.

### Physics Reference Generator

New script: `scripts/generate_physics_reference_dataset.py`

**Features:**
- Full harness mode (Height + XY + Orientation stabilization)
- Physics-derived contacts from MuJoCo `efc_force`
- AMP features computed from realized physics states
- Quality gates (accept/trim/reject based on load support)

**Results:**
- 12 motions processed → All accepted
- 3,407 frames, 68.14s total duration
- Output: `playground_amp/data/physics_ref/walking_physics_merged.pkl`

### GMR Script Updates (v0.7.0 + v0.9.3)

**v0.7.0 - Heading-Local Frame:**
```python
# All velocities now in heading-local frame (rotation invariance)
root_lin_vel_heading = world_to_heading_local(root_lin_vel, root_rot)
root_ang_vel_heading = world_to_heading_local(root_ang_vel, root_rot)
```

**v0.7.0 - FK-Based Contacts:**
```python
# Replaced hip-pitch heuristic (84% double stance) with FK
foot_contacts = estimate_foot_contacts_fk(
    dof_pos, root_pos, root_rot, mj_model,
    contact_height_threshold=0.02
)
```

### Config Changes

```yaml
# ppo_amass_training.yaml
version: "0.7.0"
version_name: "Physics Reference Data"

amp:
  # NEW: Physics-generated reference data
  dataset_path: playground_amp/data/physics_ref/walking_physics_merged.pkl
  # was: playground_amp/data/walking_motions_normalized_vel.pkl
```

### Expected Training Behavior

| Metric | GMR Baseline | Expected (Physics) |
|--------|--------------|-------------------|
| Robot falls | Immediately | Should walk |
| `disc_acc` | 0.50 (stuck) | 0.55-0.75 (healthy) |
| Reference feasibility | ~15% load support | 100% (harness) |

---

## [v0.6.6] - 2024-12-24: Feature Parity Fix (Root Linear Velocity)

### Problem: Policy and Reference Features Had Different Root Velocity Semantics

Root cause of `disc_acc = 0.50` (discriminator stuck at chance):

| Feature | Policy Extractor | Reference Generator | Impact |
|---------|------------------|---------------------|--------|
| **Root linear vel** | Normalized to unit direction | Raw velocity (m/s) | **CRITICAL** |

When features have different semantics, the discriminator cannot learn a stable decision boundary → collapses to 0.50 accuracy.

### Root Cause Analysis

**Root Linear Velocity Normalization Mismatch**
```python
# Policy (WRONG - normalized to unit direction):
root_linvel_dir = root_linvel / ||root_linvel||  # Removes magnitude!

# Reference (correct - raw velocity):
root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt  # Keeps magnitude
```

After z-score normalization with reference stats, policy velocities (unit vectors with small variance) become near-constant, removing a major discriminative signal.

### Changes

**Policy Feature Extractor (`amp/amp_features.py`):**
```python
# BEFORE (v0.6.5):
root_linvel_dir = root_linvel / (linvel_norm + 1e-8)  # ❌ Normalized

# AFTER (v0.6.6):
root_linvel_feat = root_linvel  # ✅ Raw velocity in m/s
```

**Waist Masking:** Removed (both policy and reference use 29 dims).

The waist_yaw being constant 0 in AMASS is actually valid signal - the policy will learn to keep waist stable like the reference data.

### Feature Vector Contract (v0.6.6)

Both policy and reference produce identical **29-dim** vectors:

| Index | Feature | Dims |
|-------|---------|------|
| 0-8 | Joint positions | 9 |
| 9-17 | Joint velocities | 9 |
| 18-20 | Root linear velocity (**raw m/s**) | 3 |
| 21-23 | Root angular velocity | 3 |
| 24 | Root height | 1 |
| 25-28 | Foot contacts | 4 |
| **Total** | | **29** |

### No Reference Data Rebuild Needed

This fix only changes the policy extractor. Reference data already uses raw velocities and 29 dims.

### Expected Post-Fix Behavior

| Metric | Before (v0.6.5) | Expected (v0.6.6) |
|--------|-----------------|-------------------|
| `disc_acc` | 0.50 (stuck) | 0.55-0.75 (oscillating) |
| `D(real)` | ≈0.5 | 0.7-0.9 |
| `D(fake)` | ≈0.5 | 0.3-0.5 |
| `amp_reward` | ≈0.05 | 0.1-0.4 |

---

## [v0.6.5] - 2024-12-24: Discriminator Middle-Ground + Diagnostic

### Problem: disc_acc stuck at extremes

| Version | `disc_acc` | Problem |
|---------|-----------|---------|
| v0.6.3 | 1.00 | D too strong (overpowered) |
| v0.6.4 | 0.50 | D collapsed (not learning) |

### Changes

**Middle-ground discriminator settings:**
| Parameter | v0.6.3 | v0.6.4 | v0.6.5 |
|-----------|--------|--------|--------|
| `disc_lr` | 1e-4 | 5e-5 | **8e-5** |
| `update_steps` | 3 | 1 | **2** |
| `r1_gamma` | 5.0 | 20.0 | **10.0** |

**Diagnostic settings (test if noise/filter erases distribution):**
| Parameter | Before | v0.6.5 | Rationale |
|-----------|--------|--------|-----------|
| `amp.weight` | 0.3 | **0.5** | Boost AMP signal (was underfeeding) |
| `disc_input_noise_std` | 0.03 | **0.0** | Test if noise erases distribution |
| `velocity_filter_alpha` | 0.5 | **0.0** | Test if filter erases distribution |

### Expected Outcomes

**If disc_acc jumps above 0.50:**
- Noise/filter was erasing the distribution difference
- Re-enable with lower values

**If disc_acc stays at 0.50:**
- Feature mismatch between policy and reference
- Run `diagnose_amp_features.py` to compare features
- Check: joint ordering, pelvis orientation (Z-up vs Y-up), degrees vs radians

### Target

`disc_acc = 0.55-0.75` (healthy adversarial game)

---

## [v0.6.4] - 2024-12-24: Discriminator Balance Fix

### Problem: Discriminator Collapse (disc_acc = 1.00)

Training logs showed classic discriminator collapse:
```
#60-#220: disc_acc=1.00 | amp≈0.02-0.03 | success=0.0%
```

**Root Cause Analysis:**
- `disc_acc = 1.00` means discriminator perfectly classifies real vs policy samples
- This causes `amp_reward → 0` (policy gets no gradient from AMP)
- Policy ignores style and only optimizes task reward → no AMASS transfer
- This is the "perfect expert classifier" failure mode

**Why it happened (previous v0.6.3 config):**
| Parameter | Old Value | Problem |
|-----------|-----------|---------|
| `disc_lr` | 1e-4 | Too high - D learns too fast |
| `update_steps` | 3 | Too many D updates per PPO iter |
| `r1_gamma` | 5.0 | Too weak - not enough gradient penalty |
| `disc_input_noise_std` | 0.02 | Too low - D can overfit to clean features |
| `replay_buffer_ratio` | 0.5 | D learns to perfectly classify old samples |

### Changes

**Config (ppo_amass_training.yaml):**
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `disc_lr` | 1e-4 | 5e-5 | Slow down D learning |
| `update_steps` | 3 | 1 | Fewer D updates per PPO iteration |
| `r1_gamma` | 5.0 | 20.0 | Stronger gradient penalty to smooth D |
| `disc_input_noise_std` | 0.02 | 0.05 | More noise to blur D decision boundary |
| `replay_buffer_ratio` | 0.5 | 0.2 | Less historical samples when D is strong |

### Expected Post-Fix Behavior

| Metric | Before | Expected After |
|--------|--------|----------------|
| `disc_acc` | 1.00 (stuck) | 0.55-0.75 (oscillating) |
| `amp_reward` | ≈0.02 | 0.1-0.4 (rising) |
| `task_r` | Rising fast | May dip initially, then rise |
| `success` | 0.0% | Should become non-zero |

**Key insight:** You want the discriminator to **struggle**, not win. A healthy GAN game keeps D accuracy around 0.55-0.75.

### Future Tuning

Once `disc_acc` stabilizes in 0.55-0.75 range:
- Increase `amp.weight` from 0.3 → 0.5 for stronger style influence
- Monitor for any regression back to 1.00

---

## [v0.6.2] - 2024-12-24: Golden Rule Configuration

### Problem
The v0.6.1 Golden Rule fixes had hardcoded values that should be configurable:
1. **Hardcoded joint indices**: `left_hip_pitch=1`, `left_knee_pitch=3`, etc. were hardcoded instead of derived from `robot_config.yaml`
2. **Magic numbers unexplained**: Contact estimation used unexplained values (0.5, 0.3, 1.0)
3. **Parameters as defaults**: `use_estimated_contacts`, `use_finite_diff_vel`, and contact params had defaults instead of being required from training config

### Changes

#### Configuration (Training Config → Feature Extraction)
| File | Change |
|------|--------|
| `configs/ppo_amass_training.yaml` | Added Golden Rule section with all configurable params |
| `configs/config.py` | Added Golden Rule params to `TrainingConfig` dataclass |

**New Config Options:**
```yaml
amp:
  # v0.6.2: Golden Rule Configuration (Mathematical Parity)
  use_estimated_contacts: true   # Use joint-based contact estimation (matches reference)
  use_finite_diff_vel: true      # Use finite difference velocities (matches reference)
  contact_threshold_angle: 0.1   # Hip pitch threshold for contact detection (rad, ~6°)
  contact_knee_scale: 0.5        # Knee angle at which confidence decreases (rad, ~28°)
  contact_min_confidence: 0.3    # Minimum confidence when hip indicates contact (0-1)
```

#### Feature Extraction (Config-Driven Joint Indices)
| File | Change |
|------|--------|
| `amp/amp_features.py` | Added joint indices to `AMPFeatureConfig` (derived from robot_config) |
| `amp/amp_features.py` | `estimate_foot_contacts_from_joints()` uses config indices |
| `amp/amp_features.py` | `extract_amp_features()` now REQUIRES Golden Rule params (no defaults) |
| `amp/amp_features.py` | Added comprehensive docstrings explaining magic numbers |

**AMPFeatureConfig Now Includes:**
```python
class AMPFeatureConfig(NamedTuple):
    # ... existing fields ...
    # v0.6.2: Joint indices for contact estimation (from robot_config)
    left_hip_pitch_idx: int   # Index of left_hip_pitch in joint_pos
    left_knee_pitch_idx: int  # Index of left_knee_pitch in joint_pos
    right_hip_pitch_idx: int  # Index of right_hip_pitch in joint_pos
    right_knee_pitch_idx: int # Index of right_knee_pitch in joint_pos
```

**Magic Numbers Explained:**
```python
def estimate_foot_contacts_from_joints(
    joint_pos, config,
    threshold_angle: float = 0.1,   # ~6° - leg behind body → contact
    knee_scale: float = 0.5,        # ~28° - confidence decreases as knee bends
    min_confidence: float = 0.3,    # 30% - minimum contact when hip indicates contact
):
    # Confidence formula: clip(1.0 - |knee| / knee_scale, min_confidence, 1.0)
    # - knee=0 → confidence=1.0 (fully extended)
    # - knee=0.5 → confidence=0.3 (min, bent knee)
```

#### Training Pipeline (Pass Config Through)
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Added Golden Rule params to `AMPPPOConfigJit` |
| `training/trainer_jit.py` | `extract_amp_features_batched()` accepts all params |
| `training/trainer_jit.py` | `make_train_iteration_fn()` passes config values |

#### Diagnostic Script (Config-Driven)
| File | Change |
|------|--------|
| `scripts/diagnose_amp_features.py` | Reads Golden Rule params from training config |
| `scripts/diagnose_amp_features.py` | Prints config values for debugging |

### Config
```yaml
version: "0.6.2"
version_name: "Golden Rule Configuration"
```

### Key Principle
**"No hardcoding, everything from config"** — All parameters that affect feature extraction now come from `ppo_amass_training.yaml` and `robot_config.yaml`. Joint indices are derived from actuator names, not hardcoded.

### Migration
If upgrading from v0.6.1, add these fields to your training config:
```yaml
amp:
  use_estimated_contacts: true
  use_finite_diff_vel: true
  contact_threshold_angle: 0.1
  contact_knee_scale: 0.5
  contact_min_confidence: 0.3
```

### Status
✅ Complete - all Golden Rule parameters are now configurable

---

## [v0.6.0] - 2024-12-23: Spectral Normalization + Policy Replay Buffer

### Problem
Two remaining discriminator stability issues:
1. **Unconstrained Lipschitz constant**: Discriminator gradients could explode, destabilizing training
2. **Catastrophic forgetting**: Discriminator overfits to recent policy samples, forgetting earlier distributions

### Changes

#### Spectral Normalization (Miyato et al., 2018)
| File | Change |
|------|--------|
| `amp/discriminator.py` | Added `SpectralNormDense` layer wrapping Flax `nn.SpectralNorm` |
| `amp/discriminator.py` | `AMPDiscriminator` now uses spectral-normalized layers by default |
| `amp/discriminator.py` | Added `use_spectral_norm` flag for A/B testing (default: `True`) |

**Why Spectral Normalization:**
- Industry standard for GANs (StyleGAN, BigGAN, etc.)
- Controls Lipschitz constant to 1 by normalizing weights by largest singular value
- Prevents gradient explosion without hyperparameter tuning
- More stable than gradient penalties alone

```python
class SpectralNormDense(nn.Module):
    """Dense layer with Spectral Normalization."""
    features: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        dense = nn.Dense(features=self.features, ...)
        spectral_dense = nn.SpectralNorm(dense, collection_name="batch_stats")
        return spectral_dense(x)
```

#### Policy Replay Buffer
| File | Change |
|------|--------|
| `amp/replay_buffer.py` | NEW: `PolicyReplayBuffer` (Python mutations) |
| `amp/replay_buffer.py` | NEW: `JITReplayBuffer` (functional JAX updates) |
| `training/trainer_jit.py` | Added `replay_buffer_size` and `replay_buffer_ratio` to config |
| `training/trainer_jit.py` | Added replay buffer state fields to `TrainingState` |
| `train.py` | Pass replay buffer config from YAML |
| `configs/ppo_amass_training.yaml` | Added replay buffer settings |

**Why Replay Buffer:**
- Stores historical policy features to prevent discriminator from overfitting to current policy
- Mixes `replay_buffer_ratio` (default 50%) historical samples with fresh samples
- JIT-compatible implementation using functional updates for `jax.lax.scan`

```python
# JIT-compatible replay buffer (functional updates)
class JITReplayBuffer:
    @staticmethod
    def add(state: dict, samples: jnp.ndarray) -> dict:
        # Functional update for use in jax.lax.scan
        ...

    @staticmethod
    def sample(state: dict, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        # Sample batch mixing fresh + historical
        ...
```

### Config
```yaml
version: "0.6.0"
version_name: "Spectral Normalization + Policy Replay Buffer"

amp:
  # v0.6.0: Policy Replay Buffer
  replay_buffer_size: 100000
  replay_buffer_ratio: 0.5  # 50% historical samples
```

### Expected Result
- More stable discriminator training (no gradient explosions)
- Smoother `disc_acc` curves (less catastrophic forgetting)
- Better generalization of discriminator across policy evolution

### Status
🔄 Ready to train - building on v0.5.0 foot contact fix

---

## [v0.5.0] - 2024-12-23: Foot Contact Fix + Temporal Context

### Problem
**Critical distribution mismatch bug discovered:** Policy rollouts sent zeros for foot contacts while reference data had 88% non-zero foot contacts. The discriminator learned to trivially distinguish them by checking if foot_contacts == 0, achieving 100% accuracy without learning actual motion quality.

| Source | Foot Contacts |
|--------|---------------|
| Reference Data | 88% non-zero (real contact values) |
| Policy Rollouts (v0.4.x) | 100% zeros (not extracted from sim!) |

**Result:** `disc_acc = 1.00` and `amp_reward ≈ 0` — the "foot contact cheat" bug.

### Root Cause
`foot_contacts` were never extracted from MJX simulation during policy rollouts. The `extract_amp_features()` function silently fell back to zeros when `foot_contacts=None`.

### Changes

#### Foot Contact Extraction (Core Fix)
| File | Change |
|------|--------|
| `envs/wildrobot_env.py` | Extract real foot contacts from MJX contact forces |
| `envs/wildrobot_env.py` | Use explicit config keys (`left_toe`, `left_heel`, etc.) |
| `envs/wildrobot_env.py` | Raise `RuntimeError` if foot geoms not found |
| `envs/wildrobot_env.py` | Made `contact_threshold` and `contact_scale` configurable |
| `training/trainer_jit.py` | Added `foot_contacts` to `Transition` namedtuple |
| `training/trainer_jit.py` | Pass `foot_contacts` through entire training pipeline |
| `amp/amp_features.py` | Require `foot_contacts` (no silent fallback to zeros!) |

#### Asset Updates
| File | Change |
|------|--------|
| `assets/post_process.py` | Renamed geoms: `left_toe/left_heel` (was `left_foot_btm_front/back`) |
| `assets/post_process.py` | Added explicit config keys to `robot_config.yaml` |
| `assets/robot_config.yaml` | Added `left_toe`, `left_heel`, `right_toe`, `right_heel` keys |
| `assets/wildrobot.xml` | Updated geom names via post_process.py |

#### Temporal Context Infrastructure (P3 - Prepared, Not Enabled)
| File | Change |
|------|--------|
| `amp/amp_features.py` | Added `TemporalFeatureConfig`, `create_temporal_buffer()` |
| `amp/amp_features.py` | Added `update_temporal_buffer()`, `add_temporal_context_to_reference()` |

#### Testing
| File | Description |
|------|-------------|
| `tests/test_foot_contacts.py` | 8 comprehensive tests for foot contact pipeline |

### Foot Contact Detection
```python
# 4-point model: [left_toe, left_heel, right_toe, right_heel]
# Soft thresholding: tanh(force / scale) for continuous 0-1 values
foot_contacts = jnp.tanh(contact_forces / self._contact_scale)
```

### Key Principle
**"Fail loudly, no silent fallbacks"** — If foot contacts are missing, raise an exception immediately rather than silently using zeros.

### Expected Result
- `disc_acc` should fluctuate around 0.5-0.7 (healthy discrimination)
- `amp_reward` should stay meaningful (0.3-0.8)
- `success_rate` should gradually increase as robot learns

### Status
🔄 Ready to train - foot contacts now correctly extracted from simulation

---

## [v0.4.1] - 2024-12-23: R1 Regularizer + Distribution Metrics

### Problem
WGAN-GP gradient penalty was theoretically inconsistent with LSGAN loss (which uses bounded targets [0, 1]).

### Changes

#### R1 Regularizer (Replaced WGAN-GP)
| File | Change |
|------|--------|
| `amp/discriminator.py` | Replaced WGAN-GP with R1 regularizer |
| `training/trainer_jit.py` | Updated discriminator training to use R1 |
| `configs/ppo_amass_training.yaml` | Renamed `gradient_penalty_weight` → `r1_gamma` |

**Why R1 over WGAN-GP:**
- WGAN-GP uses interpolated samples (designed for Wasserstein critics)
- R1 only penalizes gradient on REAL samples (simpler, faster)
- More theoretically consistent with LSGAN

```python
# R1 Regularization: gradient penalty on REAL samples only
def real_disc_sum(x):
    return jnp.sum(model.apply(params, x, training=True))

grad_real = jax.grad(real_disc_sum)(real_obs)
r1_penalty = jnp.mean(jnp.sum(grad_real ** 2, axis=-1))

total_loss = lsgan_loss + (r1_gamma / 2.0) * r1_penalty
```

#### Distribution Metrics
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Added `disc_real_mean`, `disc_fake_mean`, `disc_real_std`, `disc_fake_std` |

These metrics help diagnose discriminator behavior:
- `disc_real_mean` should approach 1.0
- `disc_fake_mean` should approach 0.0
- If both are similar, discriminator is not learning separation

### Config
```yaml
amp:
  r1_gamma: 5.0  # R1 regularizer weight (was gradient_penalty_weight)
```

### Status
✅ Implemented and validated

---

## [v0.4.0] - 2024-12-23: Critical Bug Fixes (Reward Formula + Training Order)

### Problem
Two critical bugs identified by external design review:

1. **Reward Formula Bug ("The Participation Trophy"):** Policy received 0.75 reward when discriminator was 100% sure motion was fake.
2. **Training Order Bug ("The Hindsight Bias"):** AMP rewards computed with updated discriminator, not the one active when samples were collected.

### Changes

#### Fix 1: Reward Formula (Clipped Linear)
| Before (Buggy) | After (Fixed) |
|----------------|---------------|
| `max(0, 1 - 0.25 * (D(s) - 1)²)` | `clip(D(s), 0, 1)` |
| D=0 (fake) → reward=0.75 ❌ | D=0 (fake) → reward=0.0 ✅ |

| File | Change |
|------|--------|
| `amp/discriminator.py` | Changed reward formula to clipped linear |

#### Fix 2: Training Order
| Before (Buggy) | After (Fixed) |
|----------------|---------------|
| Train disc → Compute reward (NEW params) | Compute reward (OLD params) → Train disc |

| File | Change |
|------|--------|
| `training/trainer_jit.py` | Moved AMP reward computation BEFORE discriminator training |

#### Fix 3: Network Size Increase
| Setting | Before | After |
|---------|--------|-------|
| `discriminator_hidden` | `[256, 128]` | `[512, 256]` |

### Expected Result
After these fixes, `disc_acc` should immediately jump from 0.50 to 0.60-0.80 within the first 20 iterations.

### Status
✅ Implemented and validated

---

## [v0.3.1] - 2024-12-23: Config-Driven Pipeline (No Hardcoding)

### Problem
Joint order mismatch between reference data and policy features. GMR's `convert_to_amp_format.py` had **hardcoded** joint order that didn't match MuJoCo qpos order, causing discriminator to compare misaligned features.

| Component | Expected | Bug |
|-----------|----------|-----|
| MuJoCo qpos | `waist_yaw` at index 0 | ✅ Correct |
| `robot_config.yaml` | `waist_yaw` at index 0 | ✅ Correct |
| GMR hardcoded | `left_hip_pitch` at index 0 | ❌ Wrong |

### Changes

#### GMR Pipeline (Config-Driven)
| File | Change |
|------|--------|
| `GMR/scripts/convert_to_amp_format.py` | Added `--robot-config` flag (required), removed all hardcoded joint names/indices |
| `GMR/scripts/batch_convert_to_amp.py` | Added `--robot-config` flag (required), uses `get_joint_names()` |

#### WildRobot Fixes
| File | Change |
|------|--------|
| `amp/amp_features.py` | Fixed NamedTuple field order (defaults must come last) |
| `envs/wildrobot_env.py` | Fixed `_init_qpos` access before initialization, added `_mjx_model` creation |
| `scripts/scp_to_remote.sh` | Exclude cache files (`__pycache__`, `*.pyc`), include `robot_config.yaml` |

#### Cleanup
- Deleted workaround scripts: `fix_reference_data.py`, `reorder_reference_data.py`
- Cleaned `playground_amp/data/` - only `walking_motions_normalized_vel.pkl` remains (641KB)
- Removed 14 intermediate/old data files

### Reference Data Regeneration
```bash
# Full pipeline with config-driven joint order
cd ~/projects/GMR
uv run python scripts/batch_retarget_walking.py
uv run python scripts/batch_convert_to_amp.py \
    --robot-config ~/projects/wildrobot/assets/robot_config.yaml

cd ~/projects/wildrobot
uv run python scripts/convert_ref_data_normalized_velocity.py
```

### Validation
```
✅ Joint order: ['waist_yaw', 'left_hip_pitch', ...] matches robot_config.yaml
✅ Features shape: (3407, 29)
✅ Velocity normalized: True
✅ 12 motions, 68.19s total duration
```

### Key Principle
**"Fail fast, no silent defaults"** - Joint order now comes from `robot_config.yaml`, never hardcoded. Missing config raises immediate error.

### Status
🔄 Ready to train - code synced to remote

---

## [v0.3.0] - 2024-12-23: Velocity Normalization

### Problem
Speed mismatch between reference motion data (~0.27 m/s) and commanded velocity (0.5-1.0 m/s). Discriminator penalized policy for walking at correct speed.

### Changes

#### Capability
- **Velocity Direction Normalization**: Root linear velocity normalized to unit direction vector
- **Safe Thresholding**: If speed < 0.1 m/s, use zero vector (handles stationary frames)
- **Feature Validation Script**: `scripts/validate_feature_consistency.py` ensures reference = policy features

#### Files Modified
| File | Change |
|------|--------|
| `amp/amp_features.py` | Safe velocity normalization with 0.1 m/s threshold |
| `scripts/convert_ref_data_normalized_velocity.py` | Reference data conversion script |
| `scripts/validate_feature_consistency.py` | End-to-end feature validation |
| `reference_data_generation.md` | Updated pipeline documentation |

#### Config
```yaml
amp:
  dataset_path: playground_amp/data/walking_motions_normalized_vel.pkl
  # Feature dim remains 29 (direction replaces raw velocity)
```

#### Feature Format (29-dim)
| Index | Feature | Notes |
|-------|---------|-------|
| 0-8 | Joint positions | |
| 9-17 | Joint velocities | |
| 18-20 | Root velocity **DIRECTION** | Normalized unit vector (NEW) |
| 21-23 | Root angular velocity | |
| 24 | Root height | |
| 25-28 | Foot contacts | |

### Validation
- ✅ 63.4% moving frames (norm = 1.0)
- ✅ 36.6% stationary frames (norm = 0.0)
- ✅ Reference and policy features identical

### Expected Result
- AMP reward should be smoother (no speed penalty)
- Discriminator learns gait style, not speed matching
- Policy can walk at commanded speed without AMP penalty

### Status
🔄 Ready to train - awaiting results

---

## [v0.2.0] - 2024-12-22: Discriminator Tuning (Middle-Ground)

### Problem
Initial discriminator settings caused disc_acc stuck at 0.97-0.99 (too strong).
Over-correction caused disc_acc collapse to 0.50 (not learning).

### Changes

#### Config Evolution
| Setting | Initial | Over-corrected | Final (Middle-ground) |
|---------|---------|----------------|----------------------|
| `disc_lr` | 1e-4 | 2e-5 | **5e-5** |
| `update_steps` | 3 | 1 | **2** |
| `gradient_penalty_weight` | 5.0 | 20.0 | **15.0** |
| `disc_input_noise_std` | 0.0 | 0.05 | **0.05** |

#### Final Config
```yaml
amp:
  weight: 0.3
  disc_lr: 5e-5
  batch_size: 256
  update_steps: 2
  gradient_penalty_weight: 15.0
  disc_input_noise_std: 0.05
```

### Result
- ✅ Healthy disc_acc oscillation: 0.72-0.95
- ✅ AMP reward stable
- ❌ Speed mismatch still present (addressed in v0.3.0)

### Lessons Learned
- `disc_acc = 0.50` means discriminator collapsed (not learning)
- `disc_acc = 0.97+` means discriminator too strong
- Target: `disc_acc = 0.6-0.8` with healthy oscillation
- Don't over-handicap the discriminator

---

## [v0.1.1] - 2024-12-21: JIT Compilation Fix

### Problem
30-second GPU idle after first training iteration.

### Cause
Second JIT compilation when switching from single iteration to batch mode.

### Fix
Pre-compile both `train_iteration_fn` and `train_batch_fn` before training starts.

#### Files Modified
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Pre-compile JIT functions with dummy call |
| `train.py` | Added PID display for easy termination |

#### Code
```python
# Pre-compile both functions before training
print("Pre-compiling JIT functions...")
_ = train_iteration_fn(state, env_state, ref_buffer_data)
jax.block_until_ready(_)

if train_batch_fn is not None:
    _ = train_batch_fn(state, env_state, ref_buffer_data)
    jax.block_until_ready(_)
```

### Result
- ✅ No more GPU idle after iteration #1
- ✅ Smooth training from start

---

## [v0.1.0] - 2024-12-20: Initial AMP Training Pipeline

### Capability
- PPO + AMP adversarial training
- LSGAN discriminator formulation
- Reference motion from AMASS via GMR retargeting
- W&B logging integration
- Checkpoint save/restore

### Config
```yaml
trainer:
  num_envs: 1024
  rollout_steps: 128
  iterations: 1000
  lr: 3e-4

reward_weights:
  tracking_lin_vel: 5.0
  base_height: 0.3
  action_rate: -0.01

amp:
  weight: 0.3
  discriminator_hidden: [256, 128]
```

### Result
- Training functional but disc_acc issues (see v0.2.0)
- Speed mismatch discovered (see v0.3.0)

---

## Training Results Log

| Date | Version | Iterations | disc_acc | Velocity | Notes |
|------|---------|------------|----------|----------|-------|
| 2024-12-20 | v0.1.0 | 500 | 0.97-0.99 | ~0.3 m/s | Disc too strong |
| 2024-12-22 | v0.2.0 | 300 | 0.72-0.95 | ~0.5 m/s | Healthy oscillation |
| 2024-12-23 | v0.3.0 | TBD | TBD | TBD | Velocity normalized |
| 2025-12-28 | v0.10.4 | 500 | 0.50 | 0.65 m/s | PPO-only; vel_err ~0.085; ep_len ~460; pitch fall ~10-17%; max torque ~0.98 |
| 2025-12-28 | v0.10.5 | 400 | 0.50 | 0.63 m/s | PPO-only resume from v0.10.4; reward ~566; ankle-dominant gait persisted |

---

## Quick Reference

### Discriminator Tuning Guide
| Symptom | disc_acc | Action |
|---------|----------|--------|
| Disc too strong | 0.95+ | ↓ disc_lr, ↓ update_steps, ↑ gradient_penalty |
| Disc collapsed | ~0.50 | ↑ disc_lr, ↑ update_steps, ↓ gradient_penalty |
| Healthy | 0.6-0.8 | Keep current settings |

### Feature Validation
```bash
# Validate reference = policy features
uv run python scripts/validate_feature_consistency.py

# Analyze reference data velocities
uv run python scripts/analyze_reference_velocities.py
```

### Training Commands
```bash
# Start training
cd ~/projects/wildrobot
uv run python playground_amp/train.py

# Kill training (PID printed at startup)
kill -9 <PID>
```

---

## Future Improvements (Backlog)

- [ ] Label smoothing for discriminator (defensive measure)
- [ ] Filter stationary frames from reference data if needed
- [ ] Curriculum learning for velocity commands
- [ ] Foot contact detection from simulation
- [ ] Multi-speed reference data augmentation
