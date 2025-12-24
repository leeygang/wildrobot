# AMP Training Changelog

This changelog tracks capability changes, configuration updates, and training results for the WildRobot AMP training pipeline.

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
