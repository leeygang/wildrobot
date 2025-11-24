# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WildRobot is a humanoid robotics project integrating MuJoCo simulation, reinforcement learning (RL) training, and hardware deployment. The codebase is organized into three main components:

- **models/**: MuJoCo MJCF XML models and 3D assets
- **training/**: RL training pipelines (AMP, PPO)
- **runtime/**: Hardware deployment code for embedded systems

## Key Commands

### Model Development

View model in MuJoCo viewer:
```bash
python -m mujoco.viewer models/v1/wildrobot.xml
```

Update model from Onshape CAD:
```bash
cd models/v1
./update_xml.sh
```

### Training

**Option 1: mujoco_playground (Recommended - GPU-accelerated)**

Quick start:
```bash
cd training/wildrobot_playground
./quickstart.sh standing flat 10000000
```

Train standing balance:
```bash
cd training/wildrobot_playground
python train.py --task standing --terrain flat --num_timesteps 10000000
```

Train walking:
```bash
python train.py --task walking --terrain flat --num_timesteps 20000000
```

Export to ONNX:
```bash
python export_onnx.py --policy training/logs/.../policy.pkl --output wildrobot_v1.onnx
```

**Option 2: loco-mujoco (Legacy - AMP)**

Train AMP policy:
```bash
cd training/amp
python scripts/train.py --config configs/wildrobot_amp_amass.yaml
```

Visualize trained policy:
```bash
python scripts/visualize.py --path outputs/2024-11-23/14-30-00/AMPJax_saved.pkl
```

Export policy to ONNX:
```bash
python scripts/export_onnx.py --policy outputs/.../AMPJax_saved.pkl --output wildrobot_v1.onnx
```

### Runtime/Hardware

Install runtime package:
```bash
cd runtime
pip install -e .
```

Test hardware:
```bash
python scripts/test_motors.py
python scripts/test_imu.py
```

Run policy on hardware:
```bash
python scripts/run_policy.py --policy ../training/policies/wildrobot_v1.onnx --config configs/wildrobot_pi5.json
```

Calibrate joints:
```bash
python scripts/calibrate_joints.py
python scripts/calibrate_imu.py
```

## Architecture

### Robot Configuration

The robot has **11 DOFs** (no head):
- Left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle (5 DOFs)
- Right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle (5 DOFs)
- Torso: waist (1 DOF)

Default motor controller: Dynamixel XM430-W350
Control frequency: 50 Hz (matches training frequency)

### Model Structure

The MuJoCo model (`models/v1/wildrobot.xml`) is generated from Onshape CAD using `onshape-to-robot`. Key files:
- `wildrobot.xml`: Main MJCF model
- `config.json`: Onshape conversion config
- `sensors.xml`: Sensor definitions (IMU, joint sensors)
- `joints_properties.xml`: Joint parameters for HTD-45H servos
- `scene.xml`: Environment setup
- `assets/`: STL meshes (simplified for performance)

### Training Pipeline

WildRobot supports two training approaches:

**1. mujoco_playground (Recommended)**
- GPU-accelerated using MuJoCo MJX (JAX)
- 10-50x faster than CPU-based training
- PPO algorithm with Brax
- Direct sim-to-real with ONNX export
- Located in: `training/wildrobot_playground/`

**2. loco-mujoco (Legacy)**
- CPU-based with multiprocessing
- AMP (Adversarial Motion Priors) for human-like locomotion
- Requires model symlink to loco-mujoco library
- Located in: `training/amp/`, `training/ppo/`

**Directory structure:**
- `training/wildrobot_playground/`: mujoco_playground setup (recommended)
  - `wildrobot/`: Custom environment implementation
  - `train.py`: Training script
  - `export_onnx.py`: Policy export for deployment
  - `deploy.py`: Hardware deployment script
- `models/v1/`: MuJoCo models (shared with all training methods)
  - `wildrobot.xml`: Robot MJCF model
  - `scene_flat_terrain.xml`: Flat terrain scene
  - `scene_rough_terrain.xml`: Rough terrain scene
  - `assets/`: STL meshes
- `training/amp/`: AMP training (legacy)
- `training/ppo/`: PPO training (legacy)
- `training/shared/`: Shared utilities
- `training/policies/`: Production-ready curated policies (.pkl and .onnx)
- `training/logs/`: Training outputs (gitignored)

Training outputs are saved to `training/logs/YYYY-MM-DD-HH-MM-SS/` and are gitignored.

### Runtime Architecture

The runtime package (`wildrobot_runtime`) follows a modular HAL (Hardware Abstraction Layer) design:

- `hardware/`: Motor controllers, IMU drivers, sensors
- `control/`: 50Hz control loop, observation builder, action processor
- `inference/`: Policy inference engines (PKL/JAX or ONNX)
- `utils/`: Kinematics, filters, config loading
- `io/`: Optional peripherals (camera, gamepad)

Configuration files (`runtime/configs/*.json`) specify hardware settings, motor parameters, joint offsets, and safety limits.

## Integration with loco-mujoco

The WildRobot model needs to be accessible to loco-mujoco for training. Recommended approach:

**Symlink (for active development):**
```bash
ln -s /Users/ygli/projects/wildrobot/models/v1 \
      /path/to/loco-mujoco/loco_mujoco/environments/humanoids/wildrobot
```

This ensures model changes immediately propagate to training without manual copying.

## Git Workflow

**Commit these:**
- Models: `models/`
- Training configs/scripts: `training/*/configs/`, `training/*/scripts/`
- Runtime code: `runtime/wildrobot_runtime/`, `runtime/scripts/`
- Curated policies: `training/policies/*.onnx` (tested, production-ready)
- Documentation and tests

**Do NOT commit:**
- Training outputs: `training/*/outputs/`
- Large PKL files (except curated policies)
- Python artifacts: `__pycache__/`, `*.pyc`, `.venv/`
- IDE configs: `.vscode/`, `.idea/`

## Development Notes

- **Model updates**: When modifying the Onshape CAD, run `update_xml.sh` to regenerate the MJCF model
- **STL simplification**: Models use simplified meshes (`max_stl_size: 0.1MB`) for performance
- **Joint properties**: HTD-45H servo parameters are defined in `joints_properties.xml` (kp=21.1, damping=0.08516, frictionloss=0.022275)
- **Sensor suite**: Robot has multiple IMUs (chest, knees, pelvis) for state estimation
- **Safety**: Runtime includes temperature monitoring, joint limits, and emergency stop functionality
- **Deployment formats**: PKL (JAX) for maximum performance or ONNX for lightweight deployment
