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
