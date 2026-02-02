# Reference Motion Data Generation for AMP Training

This guide covers the complete pipeline to generate reference motion data for Adversarial Motion Prior (AMP) training with WildRobot.

## Pipeline Overview

```
AMASS (Human MoCap) → GMR Retargeting → AMP Features → Normalize Velocity → Training
     .npz files          .pkl files       29-dim          29-dim (dir)      ready!
```

| Stage | Tool | Input | Output |
|-------|------|-------|--------|
| 1. Retarget | GMR `smplx_to_robot_headless.py` | AMASS `.npz` | Robot motion `.pkl` |
| 2. Convert | GMR `convert_to_amp_format.py` | Robot `.pkl` | 29-dim features |
| 3. Merge | `batch_convert_to_amp.py` | Multiple `.pkl` | Merged dataset |
| 4. Normalize velocity | `convert_ref_data_normalized_velocity.py` | 29-dim | **29-dim** (direction only) |

---

## Prerequisites

### 1. AMASS Dataset
Download from [AMASS website](https://amass.is.tue.mpg.de/). Recommended subsets:
- **KIT** - High quality walking motions
- **CMU** - Large variety of locomotion

Location: `~/projects/amass/smplx/KIT/`

### 2. GMR Framework
```bash
cd ~/projects/GMR
uv sync  # Install dependencies
```

### 3. WildRobot Model
Ensure the robot URDF/XML is configured in GMR:
- Robot config: `~/projects/GMR/GMR/robot_configs/wildrobot.yaml`
- URDF: `~/projects/wildrobot/assets/v1/wildrobot.xml`

---

## Step 1: Retarget AMASS Motions to WildRobot

### Option A: Single Motion (Interactive)
```bash
cd ~/projects/GMR
uv run python scripts/smplx_to_robot.py \
    --smplx_file ~/projects/amass/smplx/KIT/3/walking_slow01_stageii.npz \
    --robot wildrobot \
    --save_path ~/projects/wildrobot/assets/motions/walking_slow01.pkl
```

### Option B: Batch Processing (Headless)
```bash
cd ~/projects/GMR
uv run python scripts/batch_retarget_walking.py
```

This processes predefined walking motions from KIT dataset:
- Slow, medium, fast walking speeds
- Multiple subjects for variety
- Output: `~/projects/wildrobot/assets/motions/*.pkl`

### Validation
```bash
# Visualize retargeted motion
uv run python scripts/vis_robot_motion.py \
    --motion ~/projects/wildrobot/assets/motions/walking_slow01.pkl
```

**Check for:**
- ✅ Feet contact ground properly (no floating/sinking)
- ✅ Natural walking gait
- ✅ No joint limit violations (red markers in MuJoCo)

---

## Step 2: Convert to AMP Features

### Single File
```bash
cd ~/projects/GMR
uv run python scripts/convert_to_amp_format.py \
    --input ~/projects/wildrobot/assets/motions/walking_slow01.pkl \
    --output ~/projects/wildrobot/training/data/walking_slow01_amp.pkl \
    --robot-config ~/projects/wildrobot/assets/v1/robot_config.yaml \
    --target_fps 50
```

### Batch Convert All Motions
```bash
uv run python scripts/batch_convert_to_amp.py \
    --robot-config ~/projects/wildrobot/assets/v1/robot_config.yaml \
    --input-dir ~/projects/wildrobot/assets/motions \
    --output-dir ~/projects/wildrobot/training/data \
    --target_fps 50
```

**Important:** The `--robot-config` flag is required. Joint order is loaded from the config file, not hardcoded.

### 29-dim Feature Format (Original)
| Index | Feature | Dims |
|-------|---------|------|
| 0-8 | Joint positions | 9 |
| 9-17 | Joint velocities | 9 |
| 18-20 | Root linear velocity | 3 |
| 21-23 | Root angular velocity | 3 |
| 24 | Root height | 1 |
| 25-28 | Foot contacts | 4 |

---

## Step 3: Merge Multiple Motions

If you have multiple motion files, merge them:

```python
import pickle
import numpy as np

# Load all AMP files
all_features = []
for pkl_file in motion_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    all_features.append(data['features'])

# Concatenate
merged = np.concatenate(all_features, axis=0)

# Save
with open('walking_motions_merged.pkl', 'wb') as f:
    pickle.dump({'features': merged, 'feature_dim': 29}, f)
```

---

## Step 4: Normalize Root Linear Velocity (Critical!)

### Why Normalize Velocity?

The AMASS reference motions have different speeds than what we command the robot to walk:

| Source | Velocity |
|--------|----------|
| Reference data mean | ~0.27 m/s |
| Commanded range | 0.5-1.0 m/s |

**Problem:** The discriminator penalizes the policy for walking at the commanded speed because it doesn't match the slow reference motions.

**Solution:** Normalize root linear velocity to unit direction vector. This preserves *direction* (forward/backward/sideways) while removing *speed* magnitude.

### Convert to Normalized Velocity Direction
```bash
cd ~/projects/wildrobot
uv run python scripts/convert_ref_data_normalized_velocity.py
```

This script:
1. Loads `training/data/walking_motions_merged.pkl` (29-dim)
2. Normalizes indices 18-20 (root_linvel) to unit direction: `v_dir = v / (||v|| + ε)`
3. Saves `training/data/walking_motions_normalized_vel.pkl` (29-dim)

### 29-dim Feature Format (Final - with normalized direction)
| Index | Feature | Dims |
|-------|---------|------|
| 0-8 | Joint positions | 9 |
| 9-17 | Joint velocities | 9 |
| 18-20 | Root linear velocity **DIRECTION** (normalized) | 3 |
| 21-23 | Root angular velocity | 3 |
| 24 | Root height | 1 |
| 25-28 | Foot contacts | 4 |

**Note:** Root linear velocity is **NORMALIZED** to unit direction vector. Discriminator learns "walking forward should face forward" without penalizing speed.

---

## Step 5: Verify Final Dataset

### Check Feature Statistics
```bash
cd ~/projects/wildrobot
uv run python scripts/analyze_reference_velocities.py
```

### Verify Dataset Format
```python
import pickle
import numpy as np

with open('training/data/walking_motions_normalized_vel.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Features shape: {data['features'].shape}")  # Should be (N, 29)
print(f"Feature dim: {data['feature_dim']}")         # Should be 29
print(f"Velocity normalized: {data.get('velocity_normalized', False)}")  # Should be True
```

---

## Step 6: Update Training Config

Ensure `ppo_amass_training.yaml` points to the normalized velocity dataset:

```yaml
amp:
  dataset_path: training/data/walking_motions_normalized_vel.pkl
```

**Critical:** The reference data joint order MUST match `robot_config.yaml`. This is ensured by:
- GMR's `convert_to_amp_format.py` reads joint order from `--robot-config`
- No hardcoded joint names anywhere in the pipeline

And verify `robot_config.yaml` has correct dimension:

```yaml
dimensions:
  amp_feature_dim: 29  # With normalized velocity direction
```

---

## Complete Example Workflow

```bash
# 1. Retarget AMASS motions
cd ~/projects/GMR
uv run python scripts/batch_retarget_walking.py

# 2. Convert to AMP format (joint order from robot_config.yaml)
uv run python scripts/batch_convert_to_amp.py \
    --robot-config ~/projects/wildrobot/assets/v1/robot_config.yaml \
    --input-dir ~/projects/wildrobot/assets/motions \
    --output-dir ~/projects/wildrobot/training/data \
    --merged-output walking_motions_merged.pkl

# 3. Normalize velocity to direction only
cd ~/projects/wildrobot
uv run python scripts/convert_ref_data_normalized_velocity.py

# 4. Verify joint order matches
uv run python -c "
import pickle, yaml
with open('training/data/walking_motions_normalized_vel.pkl', 'rb') as f:
    ref = pickle.load(f)
with open('assets/v1/robot_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
assert ref['joint_names'] == cfg['actuators']['joints'], 'Joint order mismatch!'
print('✓ Joint order verified')
"

# 5. Train!
uv run python training/train.py
```

---

## Troubleshooting

### Issue: `disc_acc` stuck at 0.97-0.99
**Cause:** Discriminator too strong, easily distinguishes policy from reference.
**Solutions:**
- Lower `disc_lr` (e.g., 5e-5)
- Reduce `update_steps` (e.g., 2)
- Increase `disc_input_noise_std` (e.g., 0.05)

### Issue: `disc_acc` stuck at 0.50
**Cause:** Discriminator collapsed, not learning.
**Solutions:**
- Increase `disc_lr` (e.g., 5e-5 → 1e-4)
- Increase `update_steps` (e.g., 1 → 2)
- Reduce `r1_gamma`

### Issue: AMP reward low despite good locomotion
**Cause:** Speed mismatch between reference and policy.
**Solution:** Use normalized velocity features (direction only, not magnitude).

### Issue: Feet floating or sinking in retargeted motion
**Cause:** Height scaling mismatch in GMR.
**Solution:** Adjust `s_root` scale factor in GMR config.

### Issue: Joint limit violations
**Cause:** Human motion exceeds robot's joint range.
**Solution:** Enable `use_velocity_limit=True` in GMR and check joint limits.

---

## File Locations Summary

| File | Location | Description |
|------|----------|-------------|
| AMASS source | `~/projects/amass/smplx/KIT/` | Raw human MoCap |
| GMR scripts | `~/projects/GMR/scripts/` | Retargeting tools |
| Robot motions | `~/projects/wildrobot/assets/motions/` | Retargeted `.pkl` |
| AMP data | `~/projects/wildrobot/training/data/` | Training-ready features |
| Training config | `training/configs/ppo_amass_training.yaml` | Points to dataset |
| Robot config | `assets/v1/robot_config.yaml` | Feature dimensions |

---

## References

- [GMR: General Motion Retargeting](https://github.com/YanjieZe/GMR)
- [AMASS Dataset](https://amass.is.tue.mpg.de/)
- [AMP: Adversarial Motion Priors](https://xbpeng.github.io/projects/AMP/)
