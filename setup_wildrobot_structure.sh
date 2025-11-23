#!/bin/bash
# WildRobot Project Setup Script
#
# This script creates the complete directory structure for the WildRobot project.
#
# Usage:
#   cd /Users/ygli/projects/wildrobot
#   bash setup_structure.sh

set -e  # Exit on error

echo "========================================="
echo "WildRobot Project Structure Setup"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create main directories
echo -e "${BLUE}Creating main directories...${NC}"

mkdir -p models/wildrobot_v1
mkdir -p models/wildrobot_v2
mkdir -p models/assets/{meshes,textures,collision}

mkdir -p training/amp/{configs,scripts,expert_data,outputs}
mkdir -p training/ppo/{configs,scripts,outputs}
mkdir -p training/shared
mkdir -p training/policies

mkdir -p runtime/wildrobot_runtime/{hardware,control,inference,utils,io}
mkdir -p runtime/scripts
mkdir -p runtime/configs
mkdir -p runtime/tests

mkdir -p experiments/{sim,hardware}

mkdir -p tools/{calibration,visualization,conversion,data}

mkdir -p docs/{hardware,software,training,api,images}

mkdir -p tests/{models,training,runtime,integration}

mkdir -p .github/workflows

mkdir -p docker/{training,deployment}

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Create __init__.py files for Python packages
echo -e "${BLUE}Creating Python package files...${NC}"

touch runtime/wildrobot_runtime/__init__.py
touch runtime/wildrobot_runtime/hardware/__init__.py
touch runtime/wildrobot_runtime/control/__init__.py
touch runtime/wildrobot_runtime/inference/__init__.py
touch runtime/wildrobot_runtime/utils/__init__.py
touch runtime/wildrobot_runtime/io/__init__.py

touch training/shared/__init__.py

echo -e "${GREEN}✓ Python packages initialized${NC}"
echo ""

# Create README files
echo -e "${BLUE}Creating README files...${NC}"

cat > models/README.md << 'EOF'
# WildRobot Models

This directory contains MuJoCo MJCF XML models for the WildRobot humanoid.

## Structure

- `wildrobot_v1/` - Current robot model
- `wildrobot_v2/` - Future iterations
- `assets/` - Shared meshes, textures, collision shapes

## Usage

Test model in MuJoCo viewer:
```bash
python -m mujoco.viewer models/wildrobot_v1/wildrobot.xml
```
EOF

cat > training/README.md << 'EOF'
# WildRobot Training

RL training pipelines for WildRobot locomotion.

## Algorithms

- `amp/` - Adversarial Motion Priors (human-like motion)
- `ppo/` - Proximal Policy Optimization (pure RL)

## Quick Start

```bash
cd amp
python scripts/train.py --config configs/wildrobot_amp_amass.yaml
```

## Policies

Trained policies are saved in `policies/` directory.
EOF

cat > runtime/README.md << 'EOF'
# WildRobot Runtime

Hardware deployment code for running trained policies on real WildRobot hardware.

## Installation

```bash
cd runtime
pip install -e .
```

## Quick Start

```bash
# Test hardware
python scripts/test_motors.py
python scripts/test_imu.py

# Run policy
python scripts/run_policy.py --policy ../training/policies/wildrobot_v1.onnx
```

## Structure

- `wildrobot_runtime/` - Core Python package
- `scripts/` - Deployment and testing scripts
- `configs/` - Hardware configurations
EOF

cat > docs/README.md << 'EOF'
# WildRobot Documentation

Complete documentation for WildRobot development and deployment.

## Contents

- `hardware/` - Assembly, BOM, wiring
- `software/` - Setup, deployment
- `training/` - Training guides
- `api/` - API reference
EOF

echo -e "${GREEN}✓ README files created${NC}"
echo ""

# Create .gitignore
echo -e "${BLUE}Creating .gitignore...${NC}"

cat > .gitignore << 'EOF'
# Training outputs
training/*/outputs/
*.pkl
!training/policies/*.pkl  # Keep curated policies

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
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
.AppleDouble
.LSOverride

# Data
*.hdf5
data/raw/
data/processed/

# Temporary
tmp/
temp/
*.tmp

# Jupyter
.ipynb_checkpoints/
*.ipynb

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json

# Weights & Biases
wandb/
EOF

echo -e "${GREEN}✓ .gitignore created${NC}"
echo ""

# Create requirements files
echo -e "${BLUE}Creating requirements files...${NC}"

cat > requirements.txt << 'EOF'
# WildRobot Requirements

# Core
numpy>=1.24.0
scipy>=1.10.0

# Runtime (choose one based on deployment)
# Option A: JAX (for PKL models, heavier)
# jax[cpu]>=0.4.0
# jaxlib>=0.4.0
# flax>=0.7.0

# Option B: ONNX Runtime (for ONNX models, lighter)
onnxruntime>=1.15.0

# Hardware interfaces
# dynamixel-sdk>=3.7.0  # If using Dynamixel motors
# RPi.GPIO>=0.7.1  # For Raspberry Pi GPIO
# adafruit-circuitpython-bno08x  # For BNO085 IMU

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
EOF

cat > runtime/requirements.txt << 'EOF'
# Runtime-specific requirements (minimal for deployment)

numpy>=1.24.0
scipy>=1.10.0
onnxruntime>=1.15.0  # Or onnxruntime-gpu
pyyaml>=6.0

# Hardware (uncomment based on your setup)
# dynamixel-sdk>=3.7.0
# RPi.GPIO>=0.7.1
# adafruit-circuitpython-bno08x
EOF

cat > runtime/requirements-dev.txt << 'EOF'
# Development requirements

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Linting
black>=23.0.0
flake8>=6.0.0
mypy>=1.3.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0
EOF

echo -e "${GREEN}✓ Requirements files created${NC}"
echo ""

# Create pyproject.toml
echo -e "${BLUE}Creating pyproject.toml...${NC}"

cat > runtime/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "wildrobot-runtime"
version = "0.1.0"
description = "Runtime deployment for WildRobot humanoid robot"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "onnxruntime>=1.15.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
]

[project.scripts]
wildrobot-run = "wildrobot_runtime.control.policy_runner:main"
wildrobot-test = "wildrobot_runtime.scripts.test_hardware:main"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
EOF

echo -e "${GREEN}✓ pyproject.toml created${NC}"
echo ""

# Create example config
echo -e "${BLUE}Creating example hardware config...${NC}"

cat > runtime/configs/example_config.json << 'EOF'
{
  "robot_name": "wildrobot_v1",
  "start_paused": false,
  "control_frequency": 50,

  "imu": {
    "type": "bno085",
    "upside_down": false,
    "i2c_bus": 1,
    "address": "0x4a",
    "sample_rate": 100
  },

  "motors": {
    "type": "dynamixel",
    "serial_port": "/dev/ttyUSB0",
    "baudrate": 1000000,
    "protocol": 2.0,
    "model": "XM430-W350",
    "joint_names": [
      "hip_yaw_left", "hip_roll_left", "hip_pitch_left",
      "knee_left", "ankle_left",
      "hip_yaw_right", "hip_roll_right", "hip_pitch_right",
      "knee_right", "ankle_right",
      "waist"
    ],
    "pid": {
      "kp": 30,
      "ki": 0,
      "kd": 0
    }
  },

  "joints_offsets": {
    "hip_yaw_left": 0.0,
    "hip_roll_left": 0.0,
    "hip_pitch_left": 0.0,
    "knee_left": 0.0,
    "ankle_left": 0.0,
    "hip_yaw_right": 0.0,
    "hip_roll_right": 0.0,
    "hip_pitch_right": 0.0,
    "knee_right": 0.0,
    "ankle_right": 0.0,
    "waist": 0.0
  },

  "control": {
    "action_scale": 0.25,
    "max_motor_velocity": 5.24
  },

  "safety": {
    "joint_limits_enabled": true,
    "temperature_monitoring": true,
    "max_temperature": 70,
    "emergency_stop_enabled": true
  },

  "peripherals": {
    "camera": false,
    "speaker": false,
    "leds": false,
    "gamepad": false
  }
}
EOF

echo -e "${GREEN}✓ Example config created${NC}"
echo ""

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Directory structure created for:"
echo "  • Models and simulation assets"
echo "  • Training pipelines (AMP, PPO)"
echo "  • Hardware runtime deployment"
echo "  • Experiments and tools"
echo "  • Documentation and tests"
echo ""
echo "Next steps:"
echo "  1. Copy your WildRobot MJCF model to models/wildrobot_v1/"
echo "  2. Set up training configs in training/amp/configs/"
echo "  3. Implement runtime hardware drivers in runtime/wildrobot_runtime/hardware/"
echo "  4. Write documentation in docs/"
echo ""
echo "See PROJECT_STRUCTURE.md for detailed documentation."
echo ""
