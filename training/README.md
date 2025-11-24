# WildRobot Training

RL training pipeline for WildRobot locomotion using Google DeepMind's mujoco_playground.

## Directory Structure

```
training/
├── wildrobot_playground/    # Active training pipeline (mujoco_playground + MJX)
│   ├── configs/             # YAML configuration files
│   ├── wildrobot/           # WildRobot environment package
│   ├── train.py             # Main training script with W&B logging
│   ├── export_onnx.py       # Export policy to ONNX
│   ├── deploy.py            # Deploy to hardware
│   └── README.md            # Comprehensive documentation
│
└── policies/                # Production-ready trained policies
    └── (*.onnx files)       # Curated policies for deployment
```

## Quick Start

Train with default configuration:

```bash
cd wildrobot_playground
python train.py
```

Train with production settings:

```bash
python train.py --config configs/production.yaml
```

Quick test run:

```bash
python train.py --config configs/quick.yaml
```

## Training Pipeline

The `wildrobot_playground/` directory contains the complete training setup:

- **Framework**: mujoco_playground (MJX - JAX-based MuJoCo)
- **Algorithm**: PPO (Proximal Policy Optimization) from Brax
- **Task**: Unified locomotion (velocity-conditioned: standing → walking)
- **Logging**: Weights & Biases integration
- **Speed**: 10-50x faster than CPU-based training (GPU-accelerated)

See `wildrobot_playground/README.md` for detailed documentation.

## Trained Policies

Production-ready policies should be saved in the `policies/` directory:

```bash
# After training
cd wildrobot_playground
python export_onnx.py \
  --policy training/logs/wildrobot_*/policy.pkl \
  --output ../policies/wildrobot_v1.onnx
```

These ONNX files are used for hardware deployment.
