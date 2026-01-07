# WildRobot

A humanoid robot locomotion project using MuJoCo/MJX with JAX-based PPO and Adversarial Motion Priors (AMP) training.

## Project Structure

```
wildrobot/
├── README.md                           # This file
├── AGENTS.md                           # Agent guidelines
├── CLAUDE.md                           # Claude AI guidelines
├── pyproject.toml                      # Python package configuration (uv)
├── assets/                             # Robot model files
│   ├── wildrobot.xml                   # Main MuJoCo robot model
│   ├── wildrobot.urdf                  # URDF version
│   ├── scene_flat_terrain.xml          # Flat terrain scene
│   ├── scene_rough_terrain.xml         # Rough terrain scene
│   ├── joints_properties.xml           # Joint configuration
│   ├── sensors.xml                     # Sensor definitions
│   ├── robot_config.yaml               # Robot parameters
│   ├── config.json                     # Onshape configuration
│   ├── post_process.py                 # post-process during model generation
│   ├── update_xml.sh                   # generate and update the wildrobot.xml/urdf from Onshape
│   └── assets/                         # Generated Mesh files (STL)
│       ├── *.stl                       # Robot part meshes
│       └── *.part                      # Part definitions
│
├── training/                          # Main training codebase
│   ├── train.py                        # Training entry point
│   ├── CHANGELOG.md                    # Training version history
│   │
│   ├── cal/                            # v0.11.0: Control Abstraction Layer (CAL)
│   │   ├── __init__.py                 # Export CAL, specs, types
│   │   ├── cal.py                      # ControlAbstractionLayer class
│   │   ├── specs.py                    # JointSpec, ActuatorSpec, Pose3D, Velocity3D
│   │   ├── types.py                    # ActuatorType, VelocityProfile, CoordinateFrame
│   │   └── cal.md                      # CAL documentation
│   │
│   ├── amp/                            # Adversarial Motion Priors
│   │   ├── discriminator.py            # AMP discriminator network
│   │   ├── discriminator_diagnostics.py # Discriminator debugging
│   │   ├── policy_features.py          # Policy feature extraction
│   │   ├── ref_features.py             # Reference motion features
│   │   ├── ref_buffer.py               # Reference motion buffer
│   │   ├── replay_buffer.py            # Experience replay buffer
│   │   └── amp_mirror.py               # Motion mirroring utilities
│   │
│   ├── configs/                        # Configuration system
│   │   ├── __init__.py
│   │   ├── training_config.py          # Training config schema
│   │   ├── training_runtime_config.py  # Runtime config
│   │   ├── feature_config.py           # Feature configuration
│   │   ├── ppo_walking.yaml            # Walking task config
│   │   └── ppo_standing.yaml           # Standing task config
│   │
│   ├── data/                           # Reference motion data
│   │   ├── gmr/                        # GMR motion data
│   │   ├── gmr_to_physics_ref_data.py  # Motion data conversion
│   │   └── debug_gmr_physics.py        # Debug utilities
│   │
│   ├── docs/                           # Training documentation
│   │   ├── learn_first_plan.md         # Training guide
│   │   ├── TRAINING_SYSTEM_DESIGN.md   # System architecture
│   │
│   ├── envs/                           # Environment implementation
│   │   ├── wildrobot_env.py            # Main JAX environment
│   │   ├── env_info.py                 # Typed info + metrics schema
│   │   └── disturbance.py              # Push disturbance utilities
│   │
│   ├── exports/                        # Export utilities
│   │   ├── export_onnx.py              # Checkpoint → ONNX
│   │   ├── export_policy_bundle.py     # Bundle writer (library)
│   │   └── export_policy_bundle_cli.py # Bundle export CLI
│   │
│   ├── policy_contract/                # Training-side spec helpers
│   │   └── spec_builder.py             # Build PolicySpec for export
│   │
│   ├── eval/                           # Evaluation tools
│   │   ├── eval_policy.py              # Headless evaluation
│   │   └── visualize_policy.py         # MuJoCo viewer playback
│   │
│   ├── scripts/                        # Analysis scripts
│   │
│   ├── sim_adapter/                    # Sim signal extraction
│   │   ├── mjx_signals.py              # MJX Signals adapter
│   │   └── mujoco_signals.py           # Native MuJoCo Signals adapter
│   │
│   ├── core/                           # Training infrastructure (loop/rollout/metrics)
│   │   ├── __init__.py
│   │   ├── training_loop.py            # Main training loop
│   │   ├── rollout.py                  # Rollout collection
│   │   ├── checkpoint.py               # Checkpoint save/load
│   │   ├── metrics_registry.py         # Metrics tracking
│   │   └── experiment_tracking.py      # W&B integration
│   │
│   ├── algos/                          # Algorithm components
│   │   └── ppo/                         # PPO-specific core
│   │       ├── __init__.py
│   │       └── ppo_core.py              # PPO networks/losses/GAE
│   │
│   ├── tests/                          # All tests (consolidated)
│   │   ├── TEST_STRATEGY.md            # Comprehensive test strategy
│   │   ├── conftest.py                 # Shared test fixtures
│   │   ├── robot_schema.py             # Schema validation utilities
│   │   ├── run_validation.py           # Validation runner
│   │   │
│   │   │ # Physics & Environment Tests
│   │   ├── test_physics_validation.py  # Physics correctness
│   │   ├── test_reward_components.py   # Reward computation
│   │   ├── test_foot_contacts.py       # Contact force validation
│   │   ├── test_env_info_schema.py     # Environment info schema
│   │   │
│   │   │ # Training Component Tests
│   │   ├── test_training_components.py # Training infrastructure
│   │   ├── test_trainer_invariance.py  # Training invariance
│   │   ├── test_ppo_core.py            # PPO algorithm tests
│   │   ├── test_amp_features.py        # AMP feature tests
│   │   │
│   │   │ # Math Utility Tests
│   │   ├── test_quat_edges.py          # Quaternion edge cases
│   │   ├── test_quat_normalize.py      # Quaternion normalization
│   │   │
│   │   │ # Environment State Tests
│   │   ├── envs/                       # JAX environment tests
│   │   │   ├── test_jax_env_fns.py
│   │   │   ├── test_jax_equiv_small.py
│   │   │   ├── test_jax_equiv_expanded.py
│   │   │   ├── test_jax_fullstate_small.py
│   │   │   ├── test_jax_contact_equiv.py
│   │   │   └── test_termination_checks.py
│   │   │
│   │   └── debug_contact_forces.ipynb  # Contact force debugging
│   │
│   ├── scripts/                        # Debug & diagnostic scripts
│   │   ├── debug_amp_features.py
│   │   ├── diagnose_amp_features.py
│   │   ├── diagnose_torque.py
│   │   ├── mocap_retarget.py           # Motion capture retargeting
│   │   └── test_actuator_range.py
│   │
│   ├── checkpoints/                    # Saved model checkpoints
│   │   └── wildrobot_ppo_*/            # Training run checkpoints
│   │
│   └── wandb/                          # W&B experiment logs
│
├── mujoco-brax/                        # Legacy mujoco brax training version.(reference only not used)
│   ├── README.md
│   ├── amp/                            # Reference AMP code
│   ├── common/                         # Shared utilities
│   ├── playground/                     # Experimental code
│   └── runtime/                        # Runtime utilities
│
├── isaac/                              # IsaacLab integration (WIP, not used)
│   ├── isaac_lab_strategy.md           # Integration strategy
│   ├── assets/                         # USD assets
│   └── scripts/                        # Conversion scripts
│
└── scripts/                            # Project-level scripts
    ├── scp_to_remote.sh                # Upload to remote server
    └── scp_from_remote.sh              # Download from remote server
```

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- MuJoCo 3.x
- JAX with GPU support (optional but recommended)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd wildrobot

# Install dependencies with uv
uv sync

# Or install in development mode
uv pip install -e .
```

### Running Tests

All tests are consolidated under `training/tests/`:

```bash
# Run all tests
uv run pytest training/tests -v

# Run specific test categories
uv run pytest training/tests/test_physics_validation.py -v  # Physics tests
uv run pytest training/tests/test_reward_components.py -v   # Reward tests
uv run pytest training/tests/test_ppo_core.py -v            # PPO tests
uv run pytest training/tests/envs/ -v                        # Environment tests

# Run with coverage
uv run pytest training/tests --cov=training --cov-report=html
```

## Training

### Walking Task (PPO + AMP)

```bash
# Start training with default config
uv run python training/train.py

# Train with specific config
uv run python training/train.py --config training/configs/ppo_walking.yaml

# Resume from checkpoint
uv run python training/train.py --resume training/checkpoints/<checkpoint_dir>
```

### Configuration

Training is configured via YAML files in `training/configs/`:

- `ppo_walking.yaml` - Walking locomotion task
- `ppo_standing.yaml` - Standing balance task

Key configuration options:

```yaml
# Training parameters
training:
  num_envs: 4096
  rollout_steps: 32
  num_iterations: 1000
  learning_rate: 3e-4

# Reward weights
reward:
  velocity_tracking: 1.0
  orientation: 0.5
  action_smoothness: 0.1

# AMP settings
amp:
  enabled: true
  reward_weight: 0.5
```

### Monitoring Training

Training progress is logged to [Weights & Biases](https://wandb.ai/):

```bash
# View live training metrics
wandb login
# Then check your W&B dashboard
```

## Visualization

```bash
# Visualize trained policy
uv run python training/eval/visualize_policy.py \
  --checkpoint training/checkpoints/<checkpoint_dir>
```

## Project Goals

- Develop a bipedal humanoid robot capable of robust locomotion
- Implement physics-based Adversarial Motion Priors (AMP)
- Achieve stable walking across flat and rough terrains
- Validate simulation accuracy with comprehensive test suite

## Key Features

- **JAX/MJX Acceleration**: Fast parallel simulation with JAX
- **PPO + AMP Training**: Combines PPO with adversarial motion priors
- **Comprehensive Testing**: 10-layer test strategy from schema contracts to integration
- **Schema Contracts**: Prevents silent breakage from XML changes
- **W&B Integration**: Full experiment tracking and visualization

## Development Workflow

1. **Robot Model**: Define/update robot in `assets/wildrobot.xml`
2. **Configuration**: Adjust training params in `training/configs/`
3. **Training**: Run experiments with `training/train.py`
4. **Testing**: Validate with `pytest training/tests/`
5. **Analysis**: Review metrics in W&B dashboard

## Documentation

Detailed documentation is available in `training/docs/`:

- [Training Guide](training/docs/TRAINING_README.md) - How to train policies
- [System Design](training/docs/TRAINING_SYSTEM_DESIGN.md) - Architecture overview
- [Test Strategy](training/tests/TEST_STRATEGY.md) - Comprehensive test plan
- [AMP Design](training/docs/AMP_FEATURE_PARITY_DESIGN.md) - AMP implementation details
- [E2E Runbook](docs/e2e_training_to_sim2real.md) - Train → eval → export → sim2real

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your License Here]

## Acknowledgments

- MuJoCo and MJX teams at DeepMind
- JAX team at Google
- [Any other acknowledgments]
