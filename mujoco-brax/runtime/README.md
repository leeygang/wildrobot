# WildRobot Runtime

Hardware deployment code for running trained policies on real WildRobot hardware.

## Installation

```bash
cd runtime
uv sync

# Or (explicit editable install)
uv venv
uv pip install -e .
```

## Quick Start

```bash
# Test hardware
uv run python scripts/test_motors.py
uv run python scripts/test_imu.py

# Run policy
uv run python scripts/run_policy.py --policy ../training/policies/wildrobot_v1.onnx
```

## Structure

- `wildrobot_runtime/` - Core Python package
- `scripts/` - Deployment and testing scripts
- `configs/` - Hardware configurations
