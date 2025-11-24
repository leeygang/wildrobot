# WildRobot RL Training with mujoco_playground

GPU-accelerated RL training for WildRobot using Google DeepMind's mujoco_playground framework.

## Directory Structure

```
wildrobot_playground/
├── wildrobot/              # WildRobot environment package
│   ├── base.py            # Base environment class
│   ├── constants.py       # Robot configuration and constants
│   ├── locomotion.py      # Unified locomotion task
│   └── config_utils.py    # YAML config utilities
├── default.yaml           # Default training config (30M steps)
├── quick.yaml             # Quick testing config (5M steps)
├── production.yaml        # Production config (50M steps)
├── train.py               # Training script with W&B logging
├── export_onnx.py         # Export policy to ONNX
└── deploy.py              # Deploy to hardware
```

## Installation

1. **Install mujoco_playground**:
```bash
cd /Users/ygli/projects/mujoco_playground
uv venv --python 3.11
source .venv/bin/activate
uv pip install -U "jax[cuda12]"  # For GPU support
uv pip install -e ".[all]"
```

2. **Install WildRobot runtime** (for deployment):
```bash
cd /Users/ygli/projects/wildrobot/runtime
pip install -e .
```

## Training

### Using Config Files

Train with default configuration (30M steps, 2048 envs):

```bash
cd /Users/ygli/projects/wildrobot/training/wildrobot_playground
python train.py
```

Use a specific config file:

```bash
# Quick training (5M steps, smaller network, no W&B)
python train.py --config quick.yaml

# Production training (50M steps, large network, full logging)
python train.py --config production.yaml
```

### Available Configs

- **`default.yaml`** - Balanced config (30M steps, 2048 envs, W&B enabled)
- **`quick.yaml`** - Fast prototyping (5M steps, 1024 envs, small network, no W&B)
- **`production.yaml`** - Best performance (50M steps, 4096 envs, large network)

### Command-Line Overrides

Override any config parameter from the command line:

```bash
# Override terrain and timesteps
python train.py --terrain rough --num_timesteps 50000000

# Override multiple parameters
python train.py --config quick.yaml \
  --terrain rough \
  --num_envs 4096 \
  --learning_rate 0.0005 \
  --batch_size 1024

# Enable W&B on quick config
python train.py --config quick.yaml --use_wandb=true
```

### Available Command-Line Flags

All flags override config file values:

**Config**:
- `--config` - Path to YAML config file (default: `default.yaml`)

**Environment**:
- `--terrain` - Terrain type: `flat` or `rough`

**Training**:
- `--num_timesteps` - Total training steps
- `--num_envs` - Parallel environments
- `--seed` - Random seed
- `--episode_length` - Max steps per episode

**PPO**:
- `--batch_size` - Batch size
- `--learning_rate` - Learning rate
- `--entropy_cost` - Entropy bonus

**Rendering**:
- `--render` - Render videos after training (true/false)
- `--num_videos` - Number of videos to render

**Logging**:
- `--use_wandb` - Enable W&B logging (true/false)
- `--wandb_project` - W&B project name

**Checkpointing**:
- `--load_checkpoint` - Path to checkpoint to resume from

### Example Commands

```bash
# Default training
python train.py

# Quick test with custom parameters
python train.py --config quick.yaml --terrain rough

# Production training on rough terrain
python train.py --config production.yaml --terrain rough

# Resume from checkpoint
python train.py --load_checkpoint training/logs/wildrobot_*/checkpoints/10000000
```

## Training Outputs

Training creates a timestamped directory in `training/logs/`:

```
training/logs/wildrobot_locomotion_flat_20251123-143000/
├── checkpoints/
│   ├── config.json        # Training configuration
│   ├── 1000000/           # Checkpoint at 1M steps
│   ├── 2000000/
│   └── ...
├── policy.pkl             # Final trained policy (JAX)
└── rollout_*.mp4          # Evaluation videos
```

## Weights & Biases Integration

Training automatically logs to W&B when enabled:

### Key Metrics Logged

- `train/step` - Current training step
- `train/elapsed_time` - Total training time
- `eval/episode_reward` - Average episode reward
- `eval/episode_length` - Average episode length
- `training/*` - Training metrics (policy loss, value loss, entropy)
- `videos/rollout_*` - Evaluation videos

After training starts, you'll see:
```
W&B initialized: https://wandb.ai/username/wildrobot/runs/xxxxx
```

Disable W&B with `--use_wandb=false`.

## Sim-to-Real Transfer

### 1. Export to ONNX

Convert the trained JAX policy to ONNX for deployment:

```bash
python export_onnx.py \
  --policy training/logs/wildrobot_locomotion_flat_20251123-143000/policy.pkl \
  --output wildrobot_v1.onnx
```

### 2. Deploy to Hardware

Run the policy on Raspberry Pi 5:

```bash
# Test motor communication first
python deploy.py \
  --policy wildrobot_v1.onnx \
  --config runtime/configs/wildrobot_pi5.json \
  --test-motors

# Run the policy
python deploy.py \
  --policy wildrobot_v1.onnx \
  --config runtime/configs/wildrobot_pi5.json \
  --duration 30.0
```

## Environment Design

### Observation Space (58D)

The observation vector matches hardware sensors and includes velocity command:

```python
obs = [
    qpos,           # 11 - Joint positions
    qvel,           # 11 - Joint velocities
    gravity,        # 3  - Gravity vector (from pelvis IMU)
    angvel,         # 3  - Angular velocity
    gyro,           # 9  - Gyro readings (3 IMUs × 3)
    accel,          # 9  - Accelerometer readings (3 IMUs × 3)
    prev_action,    # 11 - Previous action
    velocity_cmd,   # 1  - Commanded forward velocity (0-1 m/s)
]
```

### Action Space (11D)

Direct position control for 11 DOFs:

```python
action = [
    right_hip_pitch, right_hip_roll, right_knee_pitch, right_ankle_pitch, right_foot_roll,
    left_hip_pitch, left_hip_roll, left_knee_pitch, left_ankle_pitch, left_foot_roll,
    waist_yaw
]
```

### Reward Functions

**Unified Locomotion Task** (velocity-conditioned):
- Velocity tracking: Match commanded velocity (0-1 m/s)
- Height reward: Maintain target height (0.45m ± 0.05m)
- Upright reward: Keep gravity vector aligned with z-axis
- Lateral penalty: Minimize sideways drift
- Angular velocity penalty: Minimize wobbling
- Action smoothness: Penalize large actions
- Energy penalty: Encourage efficient movement (scaled by velocity)

The same reward function works for all velocities:
- `velocity_cmd = 0.0` → Standing balance
- `velocity_cmd = 0.5` → Normal walking
- `velocity_cmd = 1.0` → Fast walking

## Troubleshooting

### GPU Issues

If JAX doesn't detect GPU:
```bash
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
python -c "import jax; print(jax.default_backend())"  # Should print 'gpu'
```

### Import Errors

Make sure mujoco_playground is in your Python path:
```bash
export PYTHONPATH="/Users/ygli/projects/mujoco_playground:$PYTHONPATH"
```

Or install in development mode:
```bash
cd /Users/ygli/projects/mujoco_playground
pip install -e .
```

## References

- [mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [Brax PPO](https://github.com/google/brax)
