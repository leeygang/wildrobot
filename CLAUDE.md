# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WildRobot is a humanoid robotics project integrating MuJoCo simulation, reinforcement learning (RL) training, and hardware deployment. The codebase follows a unified structure supporting both MuJoCo MJX/JAX training (primary) and IsaacLab training (experimental).

The project implements the development roadmap outlined in `mujoco_playground_plan.md`, focusing on:
- **Phase I**: Digital Twin & Asset Prep
- **Phase II**: Physics Validation (torque capacity, stability)
- **Phase III**: RL Training with PPO + AMP
- **Phase IV**: Domain Randomization for sim2real
- **Phase V**: Hardware Deployment

### Core Components

- **assets/**: Robot models (MJCF, USD) and CAD sources
- **mujoco/**: MuJoCo training pipelines (MJX-accelerated + legacy AMP)
- **scripts/**: Physics validation, model conversion, visualization utilities
- **outputs/**: Training checkpoints and logs (gitignored)
- **data/**: AMP motion datasets (gitignored)

## Key Commands

### Model Development

View model in MuJoCo viewer:
```bash
python -m mujoco.viewer assets/mjcf/wildrobot.xml
```

Update model from Onshape CAD:
```bash
cd assets/onshape/wildrobot
python -m onshape_to_robot.main robot
# Post-process meshes
python ../post_process.py
```

Convert MJCF to USD (for IsaacLab):
```bash
bash scripts/conversion/mjcf_to_usd.sh
```

### Physics Validation (Critical - Run Before Training!)

Validate torque capacity (4Nm limit check):
```bash
python scripts/physics/validate_torque.py assets/mjcf/wildrobot.xml
```

Passive stability test:
```bash
python scripts/physics/drop_test.py assets/mjcf/wildrobot.xml
```

### Training

**Option 1: mujoco_playground (Recommended - GPU-accelerated MJX)**

Train walking with PPO:
```bash
cd mujoco/playground
python train.py --config configs/base.yaml
```

Train with AMP (human-like motion priors):
```bash
python train.py --config configs/amp.yaml
```

Quick test run:
```bash
python train.py --config configs/quick.yaml
```

Visualize trained policy:
```bash
python visualize_policy.py --checkpoint ../../outputs/checkpoints/policy_latest.pkl
```

Export to ONNX:
```bash
python export_onnx.py --policy ../../outputs/checkpoints/policy_latest.pkl --output wildrobot_v1.onnx
```

**Option 2: loco-mujoco (Legacy - AMP)**

Train AMP policy:
```bash
cd mujoco/amp
python train.py
```

### Hardware Deployment

Deploy policy to robot:
```bash
cd mujoco/playground
python deploy.py --policy ../../outputs/checkpoints/policy_latest.pkl
```

Test runtime on hardware:
```bash
cd mujoco/runtime
python -m wildrobot_runtime.test_motors
python -m wildrobot_runtime.test_imu
```

## Architecture

### Robot Configuration

The robot has **9 actuated DOFs** (11 joints total, 2 passive):
- Left leg: hip_pitch, hip_roll, knee, ankle (4 DOFs) + passive foot_roll
- Right leg: hip_pitch, hip_roll, knee, ankle (4 DOFs) + passive foot_roll
- Torso: waist (1 DOF)

**Note**: The foot_roll joints exist for passive compliance but are NOT actuated.

**Critical Constraint**: 4.41Nm torque limit per joint (HTD-45H servos @ 12V)
- Stall torque: 45 kg·cm = 4.41 Nm
- Safe continuous: ~60% = 2.65 Nm
- Peak dynamic: ~85% = 3.75 Nm

Control frequency: 50 Hz (matches training frequency)

### Project Structure

```
wildrobot/
├── assets/                           # Robot models and CAD sources
│   ├── mjcf/                         # MuJoCo MJCF models (PRIMARY)
│   │   ├── wildrobot.xml             # Main robot model
│   │   ├── scene_flat.xml            # Flat terrain scene
│   │   ├── scene_rough.xml           # Rough terrain scene
│   │   ├── meshes/                   # STL/OBJ mesh files
│   │   │   ├── collision/            # Simplified collision meshes
│   │   │   └── visual/               # High-res visual meshes
│   │   └── textures/                 # Material textures
│   ├── onshape/                      # Onshape CAD source
│   │   ├── wildrobot/                # onshape-to-robot output
│   │   └── post_process.py           # Mesh optimization script
│   └── usd/                          # USD format for IsaacLab
│       ├── wildrobot.usd             # Converted USD model
│       └── configuration/            # USD configs
│
├── mujoco/                           # MuJoCo training pipelines
│   ├── playground/                   # MJX/JAX training (PRIMARY)
│   │   ├── wildrobot/                # Environment package
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Base environment class
│   │   │   ├── locomotion.py         # Locomotion task
│   │   │   ├── constants.py          # Robot constants (DOFs, limits)
│   │   │   └── config_utils.py       # Config loading utilities
│   │   ├── configs/                  # Training configurations
│   │   │   ├── base.yaml             # Default training settings
│   │   │   ├── amp.yaml              # AMP-specific config (NEW)
│   │   │   ├── quick.yaml            # Quick test config
│   │   │   └── domain_rand.yaml      # Domain randomization (NEW)
│   │   ├── train.py                  # Main training script
│   │   ├── evaluate.py               # (TODO) Evaluation script
│   │   ├── export_onnx.py            # Policy export for deployment
│   │   ├── deploy.py                 # Hardware deployment script
│   │   ├── visualize_policy.py       # Policy visualization
│   │   └── README.md                 # Playground documentation
│   ├── amp/                          # Legacy AMP training
│   │   ├── train.py                  # AMP training script
│   │   └── walk_env.py               # AMP environment
│   ├── runtime/                      # Hardware deployment layer
│   │   ├── wildrobot_runtime/        # Runtime package
│   │   │   ├── hardware/             # Motor/IMU drivers
│   │   │   ├── control/              # 50Hz control loop
│   │   │   └── inference/            # Policy inference
│   │   └── configs/                  # Hardware configs
│   └── common/                       # Shared utilities
│       ├── base_env.py               # Common environment base
│       ├── constants.py              # Shared constants
│       └── video_renderer.py         # Rendering utilities
│
├── scripts/                          # Validation & utility scripts
│   ├── physics/                      # Physics validation (CRITICAL)
│   │   ├── validate_torque.py        # Torque capacity validation
│   │   ├── drop_test.py              # Passive stability test
│   │   └── check_dynamics.py         # General physics checks
│   ├── conversion/                   # Model conversion scripts
│   │   ├── onshape_to_mjcf.sh        # Onshape → MJCF
│   │   └── mjcf_to_usd.sh            # MJCF → USD
│   ├── visualize_usd.py              # USD visualization
│   ├── validate_usd.py               # USD validation
│   └── inspect_usd.py                # USD inspection
│
├── data/                             # Training data (gitignored)
│   └── amp/                          # AMP motion datasets
│       └── amass/                    # AMASS human motion capture
│
├── outputs/                          # Training outputs (gitignored)
│   ├── checkpoints/                  # Model checkpoints
│   ├── logs/                         # TensorBoard logs
│   └── videos/                       # Evaluation videos
│
├── tests/                            # Unit tests
│   ├── test_physics.py               # Physics validation tests
│   └── test_mujoco_env.py            # Environment tests
│
├── README.md                         # Project overview
├── CLAUDE.md                         # This file
├── mujoco_playground_plan.md         # Development roadmap
├── requirements.txt                  # Python dependencies
└── pyproject.toml                    # Package configuration
```

### Training Approaches

**1. mujoco_playground (PRIMARY - MJX/JAX accelerated)**
- GPU-accelerated using MuJoCo MJX (JAX backend)
- 10-50x faster than CPU-based training (4,000+ parallel envs)
- PPO algorithm with optional AMP (Adversarial Motion Priors)
- Domain Randomization for sim2real transfer
- Direct ONNX export for hardware deployment
- Located in: `mujoco/playground/`

**2. loco-mujoco (Legacy - CPU-based AMP)**
- CPU-based with multiprocessing
- AMP for human-like locomotion
- Requires external loco-mujoco library
- Located in: `mujoco/amp/`

### Runtime Architecture

The runtime package (`mujoco/runtime/wildrobot_runtime`) follows a modular HAL design:
- `hardware/`: Motor controllers (HTD-45H), IMU drivers, sensors
- `control/`: 50Hz control loop matching training frequency
- `inference/`: Policy inference (PKL/JAX or ONNX)
- Located in: `mujoco/runtime/`

### Model Generation Pipeline

1. **CAD Design**: Edit robot in Onshape
2. **Export**: `onshape-to-robot` generates MJCF + meshes → `assets/onshape/wildrobot/`
3. **Post-process**: Simplify meshes for physics performance
4. **Validate**: Run `scripts/physics/validate_torque.py` to verify 4Nm constraint
5. **Convert**: Generate USD for IsaacLab (optional)

## Integration with loco-mujoco

For legacy AMP training, symlink the model to loco-mujoco:

```bash
ln -s /Users/ygli/projects/wildrobot/assets/mjcf \
      /path/to/loco-mujoco/loco_mujoco/environments/humanoids/wildrobot
```

This ensures model changes propagate to training without manual copying.

## Development Workflow (mujoco_playground_plan.md)

### Phase I: Digital Twin & Asset Prep
1. Export MJCF from Onshape → `assets/onshape/wildrobot/`
2. Post-process meshes for performance
3. Verify mass/inertia tensors are accurate
4. Add sensor sites (IMU, foot contacts)

### Phase II: Physics Validation (CRITICAL - Run Before Training!)
1. **Passive Drop Test**: `python scripts/physics/drop_test.py`
   - Validates model stability and contact solver
2. **Active Torque Test**: `python scripts/validate_torque.py`
   - **MUST PASS**: Max torque < 2.65Nm (~60% of 4.41Nm, leaving 1.76Nm headroom)
   - **WARN**: 2.65-3.75Nm (marginal, risky for dynamic motion)
   - **FAIL**: Max torque > 3.75Nm (~85% of limit) → Return to CAD to reduce mass
   - Tests deep squat and single-leg stance poses
   - Prints body dimensions (mass, inertia, COM) for design optimization

### Phase III: RL Training Pipeline
1. Configure observation space (44-dim: gravity, joints, velocities, prev action)
2. Configure reward function (velocity tracking, AMP style, torque penalty)
3. Set domain randomization parameters
4. Train with MJX (4,000+ parallel envs)
5. Monitor torque usage during training

### Phase IV: Domain Randomization
- Friction: [0.4, 1.25]
- Mass scale: [0.85, 1.15]
- Motor strength: [0.9, 1.1]
- Joint damping: [0.5, 1.5]
- Control latency: [0, 0.02s]
- Push forces: [0, 10N]

### Phase V: Hardware Deployment
1. Export policy to ONNX
2. Test in simulation with latency/noise
3. Deploy to robot hardware
4. Safety monitors: temperature, joint limits, emergency stop

## Git Workflow

**Commit these:**
- Models: `assets/mjcf/`, `assets/onshape/`
- Training configs: `mujoco/playground/configs/`
- Environment code: `mujoco/playground/wildrobot/`
- Scripts: `scripts/`
- Runtime code: `mujoco/runtime/`
- Documentation: `README.md`, `CLAUDE.md`, `mujoco_playground_plan.md`
- Tests: `tests/`

**Do NOT commit (add to .gitignore):**
- Training outputs: `outputs/`
- AMP datasets: `data/amp/`
- Large checkpoints (except curated production policies)
- Python artifacts: `__pycache__/`, `*.pyc`, `.venv/`
- IDE configs: `.vscode/`, `.idea/`

## GMR (General Motion Retargeting) Integration

**IMPORTANT PRINCIPLE**: When working with GMR project, **DO NOT modify GMR code**. Instead:

1. **Understand and leverage GMR's existing flow** - GMR has been tested with many robots and provides a robust IK-based retargeting pipeline
2. **Only modify configuration files** (e.g., `configs/smplx_to_wildrobot.json`) to adapt to wildrobot
3. **If code changes seem necessary**, discuss with the team first - there may be a configuration-based solution
4. **Joint limits are automatic** - GMR uses `mink.ConfigurationLimit(model)` which reads limits directly from the MuJoCo XML
5. **Use `*_mimic` bodies** in IK configs - these are special sites placed at key joint locations for IK matching

### Key GMR Concepts

- **IK-based retargeting**: GMR matches body positions (pelvis, hip, knee, foot), not joint angles directly
- **Position/rotation weights**: Control IK priority (e.g., `knee_mimic: [50, 10]` means position weight=50, rotation weight=10)
- **human_scale_table**: Scales human body parts to robot proportions (wildrobot uses 0.6 scale)
- **ik_match_table1/2**: Two-stage IK solving (table2 is optional refinement)

### Wildrobot GMR Config Location

- GMR config: `/Users/ygli/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_wildrobot.json`
- Local copy: `/Users/ygli/projects/wildrobot/configs/smplx_to_wildrobot.json`

### Running GMR for Wildrobot

```bash
cd /Users/ygli/projects/GMR
python scripts/smplx_to_robot.py \
    --robot wildrobot \
    --smplx_file /path/to/motion.npz \
    --save_path /path/to/output.pkl
```

## Development Notes

- **4.41Nm Torque Limit**: This is the PRIMARY constraint (HTD-45H @ 12V: 45 kg·cm). Always validate with `validate_torque.py` before training
  - Safe continuous operation: ~2.65 Nm (60% of stall)
  - Peak dynamic: ~3.75 Nm (85% of stall)
  - Design target: Keep static poses < 2.65 Nm to leave headroom for dynamic acceleration
- **Model updates**: When modifying CAD, re-run `onshape-to-robot` and post-processing
- **Mesh optimization**: Use simplified collision meshes for physics performance
- **Control frequency**: 50Hz matches both training and hardware
- **Sensor suite**: Multiple IMUs (chest, knees, pelvis) for state estimation
- **Safety**: Hardware includes temperature monitoring, joint limits, and emergency stop
- **AMP**: Use human motion priors to learn energy-efficient gaits (critical for 4.41Nm limit)
- **Domain Randomization**: Essential for sim2real transfer - randomize physics parameters every episode
