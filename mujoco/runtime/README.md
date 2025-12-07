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
