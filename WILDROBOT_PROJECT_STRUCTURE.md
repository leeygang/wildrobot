# WildRobot Project - Recommended Folder Structure

This document describes the complete folder structure for the WildRobot humanoid robotics project, designed for development, training, and deployment.

## Project Overview

```
wildrobot/
├── models/                    # MuJoCo XML models and 3D assets
├── training/                  # RL training pipelines
├── runtime/                   # Hardware deployment code
├── experiments/               # Research experiments
├── tools/                     # Utilities
├── docs/                      # Documentation
├── tests/                     # Unit and integration tests
├── docker/                    # Docker configurations
└── .github/                   # CI/CD workflows
```

---

## 1. `models/` - Simulation Models

MuJoCo MJCF XML models and associated assets.

```
models/
├── wildrobot_v1/              # Current robot model
│   ├── wildrobot.xml          # Main MJCF file
│   ├── scene.xml              # Scene with ground/lighting
│   ├── scene_position.xml     # Scene with position control
│   ├── actuators.xml          # Actuator definitions
│   ├── sensors.xml            # Sensor definitions
│   └── materials.xml          # Visual materials
│
├── wildrobot_v2/              # Future iterations
│
├── assets/                    # Shared assets
│   ├── meshes/                # STL/OBJ mesh files
│   │   ├── pelvis.stl
│   │   ├── femur_left.stl
│   │   ├── tibia_left.stl
│   │   └── ...
│   ├── textures/              # Texture images
│   └── collision/             # Simplified collision meshes
│
└── README.md                  # Model documentation
```

**Purpose**: Version-controlled robot models used for both training (loco-mujoco) and simulation (MuJoCo viewer).

---

## 2. `training/` - RL Training Pipelines

All training-related code, organized by algorithm.

```
training/
├── amp/                       # AMP training (Adversarial Motion Priors)
│   ├── configs/               # Hydra configs
│   │   ├── wildrobot_amp_amass.yaml        # Train with AMASS data
│   │   ├── wildrobot_amp_custom.yaml       # Train with custom data
│   │   └── wildrobot_amp_walk.yaml         # Walking-specific
│   │
│   ├── scripts/               # Training scripts
│   │   ├── train.py           # Main training entry point
│   │   ├── eval.py            # Evaluation script
│   │   ├── export_onnx.py     # Convert PKL → ONNX
│   │   ├── export_pkl.py      # Extract deployment PKL
│   │   ├── visualize.py       # Visualize trained policy
│   │   └── benchmark.py       # Benchmark inference speed
│   │
│   ├── expert_data/           # Expert demonstrations
│   │   ├── wildrobot_walk.npz
│   │   ├── wildrobot_run.npz
│   │   └── README.md          # Data collection process
│   │
│   └── outputs/               # Training outputs (gitignored)
│       └── YYYY-MM-DD/
│           └── HH-MM-SS/
│               ├── AMPJax_saved.pkl
│               ├── main.log
│               ├── .hydra/
│               └── videos/
│
├── ppo/                       # PPO training (pure RL)
│   ├── configs/
│   ├── scripts/
│   └── outputs/
│
├── shared/                    # Shared training utilities
│   ├── observation_builders.py
│   ├── reward_functions.py
│   ├── data_utils.py
│   └── utils.py
│
├── policies/                  # Production-ready policies (curated)
│   ├── wildrobot_walk_v1.pkl
│   ├── wildrobot_walk_v1.onnx
│   ├── wildrobot_run_v1.pkl
│   ├── wildrobot_run_v1.onnx
│   ├── CHANGELOG.md           # Policy versions and performance
│   └── README.md              # Usage instructions
│
└── README.md                  # Training documentation
```

**Key Principles**:
- Training configs live here, NOT in loco-mujoco
- Scripts are wrappers around loco-mujoco training
- Curated policies in `policies/` for deployment
- Raw training outputs in `outputs/` (gitignored)

---

## 3. `runtime/` - Hardware Deployment

Robot control code for embedded hardware (Raspberry Pi, Jetson Nano).

**Based on Open_Duck_Mini_Runtime structure:**

```
runtime/
├── wildrobot_runtime/         # Main Python package
│   ├── __init__.py
│   │
│   ├── hardware/              # Hardware Abstraction Layer (HAL)
│   │   ├── __init__.py
│   │   ├── motor_controller.py    # Abstract motor interface
│   │   ├── dynamixel_controller.py  # Dynamixel implementation
│   │   ├── hiwonder_controller.py   # Hiwonder implementation (optional)
│   │   ├── imu.py                 # IMU driver (BNO085/ICM45686)
│   │   ├── imu_bno085.py          # BNO085 specific
│   │   ├── imu_icm45686.py        # ICM45686 specific
│   │   ├── encoders.py            # Joint encoders
│   │   ├── height_estimator.py    # Height from leg kinematics
│   │   └── feet_contacts.py       # Foot contact sensors
│   │
│   ├── control/               # Control loops
│   │   ├── __init__.py
│   │   ├── policy_runner.py       # Main 50Hz control loop
│   │   ├── observation_builder.py # Build obs from sensors
│   │   ├── action_processor.py    # Process policy actions
│   │   └── safety_monitor.py      # Safety checks and limits
│   │
│   ├── inference/             # Policy inference engines
│   │   ├── __init__.py
│   │   ├── pkl_infer.py           # JAX/PKL inference
│   │   └── onnx_infer.py          # ONNX Runtime inference
│   │
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── kinematics.py          # Forward/inverse kinematics
│   │   ├── filters.py             # Low-pass, Kalman filters
│   │   ├── config.py              # Config file loading
│   │   └── logging.py             # Logging utilities
│   │
│   └── io/                    # Optional peripherals
│       ├── __init__.py
│       ├── camera.py
│       ├── speaker.py
│       ├── leds.py
│       └── gamepad.py             # Xbox controller input
│
├── scripts/                   # Executable deployment scripts
│   ├── run_policy.py          # Main deployment script
│   ├── run_policy_pkl.py      # Run with PKL model
│   ├── run_policy_onnx.py     # Run with ONNX model
│   │
│   ├── calibrate_joints.py    # Joint offset calibration
│   ├── calibrate_imu.py       # IMU calibration
│   ├── find_offsets.py        # Interactive offset finder
│   │
│   ├── test_motors.py         # Motor diagnostics
│   ├── test_imu.py            # IMU diagnostics
│   ├── test_sensors.py        # All sensors test
│   ├── test_control_loop.py   # Control loop benchmark
│   │
│   ├── benchmark_inference.py # Benchmark policy speed
│   └── emergency_stop.py      # Emergency stop utility
│
├── configs/                   # Hardware configurations
│   ├── wildrobot_jetson.json      # Jetson Nano config
│   ├── wildrobot_pi5.json         # Raspberry Pi 5 config
│   ├── wildrobot_pi4.json         # Raspberry Pi 4 config
│   ├── example_config.json        # Template config
│   └── README.md                  # Config documentation
│
├── tests/                     # Integration tests
│   ├── test_hardware.py
│   ├── test_inference.py
│   ├── test_observation.py
│   ├── test_control_loop.py
│   └── test_safety.py
│
├── setup.py                   # Python package setup
├── pyproject.toml             # Modern Python packaging
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Development dependencies
└── README.md                  # Deployment guide
```

**Key Design Principles** (from Open_Duck_Mini):
- **Hardware abstraction**: Easy to swap motor controllers
- **Modular**: Each component testable independently
- **Safety-first**: Built-in safety monitors and e-stop
- **50 Hz control loop**: Exact match with training frequency

---

## 4. `experiments/` - Research Experiments

Experimental code and prototypes (not production).

```
experiments/
├── sim/                       # Simulation experiments
│   ├── terrain_adaptation/
│   ├── push_recovery/
│   ├── stairs_climbing/
│   ├── sloped_walking/
│   └── uneven_terrain/
│
├── hardware/                  # Hardware experiments
│   ├── motor_characterization/
│   ├── sensor_fusion/
│   ├── state_estimation/
│   └── system_identification/
│
└── README.md                  # Experiments index
```

---

## 5. `tools/` - Utilities and Tools

Helper scripts and utilities.

```
tools/
├── calibration/
│   ├── camera_calibration.py
│   ├── imu_calibration.py
│   └── motor_calibration.py
│
├── visualization/
│   ├── plot_training.py
│   ├── compare_policies.py
│   ├── visualize_trajectories.py
│   └── plot_sensors.py
│
├── conversion/
│   ├── pkl_to_onnx.py
│   ├── urdf_to_mjcf.py
│   └── stl_optimizer.py
│
└── data/
    ├── process_mocap.py
    ├── create_expert_data.py
    └── validate_dataset.py
```

---

## 6. `docs/` - Documentation

```
docs/
├── hardware/
│   ├── assembly.md            # Step-by-step assembly
│   ├── bom.md                 # Bill of materials
│   ├── wiring.md              # Wiring diagrams
│   ├── motors.md              # Motor specifications
│   ├── sensors.md             # Sensor specifications
│   └── troubleshooting.md     # Hardware issues
│
├── software/
│   ├── setup_development.md   # Dev environment setup
│   ├── setup_robot.md         # Robot software setup (Pi/Jetson)
│   ├── deployment.md          # Deployment guide
│   ├── pkl_vs_onnx.md         # PKL vs ONNX comparison
│   └── troubleshooting.md     # Software issues
│
├── training/
│   ├── amp_training.md        # AMP training guide
│   ├── ppo_training.md        # PPO training guide
│   ├── expert_data.md         # Creating expert data
│   └── hyperparameters.md     # Tuning guide
│
├── api/
│   ├── runtime_api.md         # Runtime API reference
│   └── training_api.md        # Training API reference
│
└── images/                    # Images for documentation
```

---

## 7. `tests/` - Testing

```
tests/
├── models/                    # Model validation
│   ├── test_mjcf_valid.py
│   └── test_model_physics.py
│
├── training/                  # Training pipeline tests
│   ├── test_observation.py
│   ├── test_reward.py
│   └── test_training.py
│
├── runtime/                   # Runtime tests
│   ├── test_hardware.py
│   ├── test_inference.py
│   ├── test_control_loop.py
│   └── test_safety.py
│
└── integration/               # End-to-end tests
    └── test_full_pipeline.py
```

---

## 8. Root Files

```
wildrobot/
├── .gitignore                 # Git ignore patterns
├── .github/
│   └── workflows/
│       ├── test.yml           # Run tests on push
│       └── deploy.yml         # Deploy to robot
│
├── README.md                  # Project overview
├── SETUP.md                   # Quick setup guide
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # License file
│
├── setup.py                   # Python package setup (legacy)
├── pyproject.toml             # Modern Python packaging
├── requirements.txt           # All dependencies
│
├── Makefile                   # Common tasks (optional)
└── docker-compose.yml         # Docker setup (optional)
```

---

## Development Workflows

### 1. Model Development

```bash
cd models/wildrobot_v1/
# Edit wildrobot.xml
vim wildrobot.xml

# Test in MuJoCo viewer
python -m mujoco.viewer wildrobot.xml

# Test in loco-mujoco
cd ../../training/amp
python scripts/train.py --config configs/test.yaml
```

### 2. Training Workflow

```bash
cd training/amp/

# Train policy
python scripts/train.py --config configs/wildrobot_amp_amass.yaml

# Visualize result
python scripts/visualize.py --path outputs/2024-11-23/14-30-00/AMPJax_saved.pkl

# Export for deployment
python scripts/export_onnx.py --policy outputs/.../AMPJax_saved.pkl --output wildrobot_v1.onnx

# Copy to policies/ for deployment
cp wildrobot_v1.onnx ../policies/
cp outputs/.../AMPJax_saved.pkl ../policies/wildrobot_v1.pkl
```

### 3. Deployment Workflow

```bash
cd runtime/

# Install on robot
pip install -e .

# Copy policy to robot
scp ../training/policies/wildrobot_v1.onnx robot@wildrobot.local:~/policies/

# SSH to robot
ssh robot@wildrobot.local

# Test hardware
python scripts/test_motors.py
python scripts/test_imu.py

# Calibrate
python scripts/find_offsets.py

# Run policy
python scripts/run_policy.py --policy policies/wildrobot_v1.onnx --config configs/wildrobot_pi5.json
```

---

## Git Workflow

### Branches
- `main` - Stable, tested code (deployable)
- `develop` - Development branch (integration)
- `feature/*` - Feature branches
- `experiment/*` - Experimental branches
- `hotfix/*` - Emergency fixes

### What to Commit ✅

- Models: `models/`
- Training configs: `training/*/configs/`
- Training scripts: `training/*/scripts/`
- Runtime code: `runtime/wildrobot_runtime/`
- Runtime scripts: `runtime/scripts/`
- Documentation: `docs/`
- Tests: `tests/`
- Curated policies: `training/policies/*.onnx` (small, tested policies)

### What NOT to Commit ❌ (.gitignore)

```gitignore
# Training outputs
training/*/outputs/
*.pkl  # Except in training/policies/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.venv/
venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Data
*.hdf5
data/raw/
data/processed/

# Temporary
tmp/
temp/
*.tmp
```

---

## Integration with loco-mujoco

Your WildRobot model needs to be accessible to loco-mujoco for training.

### Option 1: Symlink (Recommended for Development)

```bash
# Create symlink from loco-mujoco to your model
ln -s /Users/ygli/projects/wildrobot/models/wildrobot_v1 \
      /Users/ygli/projects/loco-mujoco/loco_mujoco/environments/humanoids/wildrobot
```

**Advantages**: Changes to model immediately reflected in training

### Option 2: Copy (Cleaner)

```bash
# Copy model to loco-mujoco
cp -r models/wildrobot_v1/* \
      /path/to/loco-mujoco/loco_mujoco/environments/humanoids/wildrobot/

# Update when model changes
# (create a Makefile target for this)
```

**Advantages**: Cleaner separation, no symlink dependencies

### Training Integration

```python
# training/amp/scripts/train.py
from loco_mujoco import ImitationFactory
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="wildrobot_amp_amass")
def train(cfg: DictConfig):
    # Create environment (uses model from loco-mujoco)
    env = ImitationFactory.make(
        "MjxWildRobot",  # Registered in loco-mujoco
        **cfg.env_params
    )

    # Train...
    # (rest of training code)

if __name__ == "__main__":
    train()
```

---

## Comparison with Open_Duck_Mini

| Aspect | Open_Duck_Mini | WildRobot |
|--------|----------------|-----------|
| **Model Location** | `mini_bdx/robots/` | `models/` (separate) |
| **Training** | `experiments/` | `training/` (organized by algo) |
| **Runtime** | `Open_Duck_Mini_Runtime` (separate repo) | `runtime/` (same repo) |
| **Policies** | Root `.onnx` files | `training/policies/` |
| **DOFs** | 14 (with head, antennas) | 11 (no head) |
| **Hardware** | Hiwonder servos | Dynamixel (configurable) |
| **Organization** | Two repos (model + runtime) | Single repo (all-in-one) |

**Advantages of WildRobot structure:**
- ✅ Everything in one repo (easier version control)
- ✅ Clear separation of concerns
- ✅ Better organized training pipelines
- ✅ Reusable runtime architecture

---

## Dependencies

### Development Environment (Mac/PC)

```bash
# Python 3.10+
pip install loco-mujoco jax flax wandb hydra-core
pip install mujoco
pip install jax2tf tf2onnx tensorflow onnx onnxruntime
```

### Robot Environment (Raspberry Pi/Jetson)

**Option A: PKL (JAX)**
```bash
pip install numpy jax jaxlib flax
pip install dynamixel-sdk  # Or your motor controller
```

**Option B: ONNX (Lightweight)**
```bash
pip install numpy onnxruntime
pip install dynamixel-sdk  # Or your motor controller
```

---

## Next Steps

1. **Create directory structure** (use the setup script below)
2. **Copy WildRobot MJCF model** to `models/wildrobot_v1/`
3. **Set up training configs** in `training/amp/configs/`
4. **Create runtime package** in `runtime/wildrobot_runtime/`
5. **Write documentation** in `docs/`

See `SETUP.md` for detailed setup instructions.

---

## Quick Setup Script

```bash
# Run this in /Users/ygli/projects/wildrobot/

# Create directory structure
mkdir -p models/wildrobot_v1 models/assets/{meshes,textures,collision}
mkdir -p training/{amp,ppo,shared,policies}/{configs,scripts,outputs}
mkdir -p runtime/{wildrobot_runtime/{hardware,control,inference,utils,io},scripts,configs,tests}
mkdir -p experiments/{sim,hardware}
mkdir -p tools/{calibration,visualization,conversion,data}
mkdir -p docs/{hardware,software,training,api,images}
mkdir -p tests/{models,training,runtime,integration}
mkdir -p .github/workflows
mkdir -p docker/{training,deployment}

echo "✓ Directory structure created!"
```

---

## Summary

This structure provides:
- **Clear organization** for models, training, and deployment
- **Separation of concerns** (sim vs hardware)
- **Reusable patterns** from Open_Duck_Mini
- **Scalability** for future robot versions
- **Professional layout** for collaboration

Start with this structure and adapt as your project evolves!
