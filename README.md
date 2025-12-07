# WildRobot

A humanoid robot locomotion project with training pipelines for both MuJoCo/Brax and IsaacLab frameworks.

## Project Structure

```
wildrobot/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Optional: installable package config
│
├── assets/                          # Robot model files
│   ├── mjcf/                       # MuJoCo model files
│   │   ├── wildrobot.xml           # Main MuJoCo robot model
│   │   ├── meshes/                 # STL/OBJ mesh files
│   │   └── textures/               # Texture files (if any)
│   │
│   └── usd/                        # USD format for IsaacLab
│       ├── wildrobot.usd           # Converted USD robot model
│       └── meshes/                 # USD mesh references
│
├── configs/                         # Configuration files
│   ├── mujoco/                     # MuJoCo-specific configs
│   │   └── train_config.yaml       # MuJoCo training hyperparameters
│   └── isaaclab/                   # IsaacLab-specific configs
│       ├── env_cfg.py              # IsaacLab environment configuration
│       └── train_cfg.py            # IsaacLab training configuration
│
├── mujoco/                          # MuJoCo/Brax training pipeline
│   ├── __init__.py
│   ├── envs/                       # Environment implementations
│   │   ├── __init__.py
│   │   └── wildrobot_env.py        # MuJoCo environment wrapper
│   ├── train.py                    # MuJoCo/Brax PPO training script
│   └── evaluate.py                 # Model evaluation script
│
├── isaaclab/                        # IsaacLab training pipeline
│   ├── __init__.py
│   ├── envs/                       # Environment implementations
│   │   ├── __init__.py
│   │   └── wildrobot_env.py        # IsaacLab environment wrapper
│   ├── tasks/                      # Task definitions
│   │   ├── __init__.py
│   │   └── locomotion.py           # Locomotion task configuration
│   ├── train.py                    # IsaacLab training script
│   └── evaluate.py                 # Model evaluation script
│
├── shared/                          # Shared components across frameworks
│   ├── __init__.py
│   ├── rewards.py                  # Common reward functions
│   ├── observations.py             # Common observation processing
│   └── utils.py                    # Utility functions
│
├── models/                          # Trained model checkpoints
│   ├── mujoco/                     # MuJoCo/Brax trained models
│   │   └── checkpoints/
│   └── isaaclab/                   # IsaacLab trained models
│       └── checkpoints/
│
├── logs/                            # Training logs and metrics
│   ├── mujoco/                     # MuJoCo training logs
│   └── isaaclab/                   # IsaacLab training logs
│
├── scripts/                         # Utility scripts
│   ├── convert_mjcf_to_usd.sh     # MJCF to USD conversion script
│   ├── visualize.py                # Visualization utilities
│   └── export_policy.py            # Policy export utilities
│
└── tests/                           # Unit tests
    ├── test_mujoco_env.py
    └── test_isaaclab_env.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- MuJoCo (for MuJoCo/Brax training)
- IsaacLab (for IsaacLab training) - should be installed at `../isaaclab/` relative to this project

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd wildrobot

# Install dependencies
pip install -r requirements.txt

# Optional: Install as editable package
pip install -e .
```

### Converting Robot Model

Convert the MuJoCo model to USD format for IsaacLab:

```bash
# Run the conversion script
bash scripts/convert_mjcf_to_usd.sh

# Or manually:
cd ../isaaclab
./isaaclab.sh -p scripts/tools/convert_mjcf.py \
  ../wildrobot/assets/mjcf/wildrobot.xml \
  ../wildrobot/assets/usd/wildrobot.usd \
  --import-sites \
  --make-instanceable
```

## Training

### MuJoCo/Brax Training

```bash
# Train with MuJoCo and Brax PPO
python mujoco/train.py --config configs/mujoco/train_config.yaml

# Evaluate trained model
python mujoco/evaluate.py --checkpoint models/mujoco/checkpoints/best_model.pt
```

### IsaacLab Training

```bash
# Train with IsaacLab
python isaaclab/train.py --config configs/isaaclab/train_cfg.py

# Evaluate trained model
python isaaclab/evaluate.py --checkpoint models/isaaclab/checkpoints/best_model.pt
```

## Visualization

```bash
# Visualize trained policy
python scripts/visualize.py --framework mujoco --checkpoint models/mujoco/checkpoints/best_model.pt
python scripts/visualize.py --framework isaaclab --checkpoint models/isaaclab/checkpoints/best_model.pt
```

## Project Goals

- Develop a bipedal humanoid robot capable of robust locomotion
- Compare training efficiency and performance between MuJoCo/Brax and IsaacLab
- Implement shared reward functions and observation spaces for fair comparison
- Achieve stable walking across various terrains

## Development Workflow

1. **Robot Design**: Define robot model in `assets/mjcf/wildrobot.xml`
2. **Conversion**: Convert to USD for IsaacLab using conversion script
3. **Environment Setup**: Implement environments in both frameworks
4. **Training**: Run parallel experiments with both simulators
5. **Comparison**: Analyze results and iterate on design

## Key Features

- **Dual Framework Support**: Train with both MuJoCo/Brax and IsaacLab
- **Shared Components**: Common reward functions and observations for consistent comparison
- **Modular Design**: Easy to extend with new tasks or environments
- **Organized Checkpoints**: Separate storage for models from different frameworks

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your License Here]

## Acknowledgments

- MuJoCo and Brax teams
- NVIDIA Isaac Lab team
- [Any other acknowledgments]
