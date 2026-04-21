# WildRobot

A humanoid robot locomotion project centered on MuJoCo/JAX training, with an
industry-style RL deployment path as the current mainline for WildRobot.

Current repo reality:
- the repo is pivoting from standing-first PPO exploration toward a
  locomotion-first `v0.19.x` platform
- the deployed runtime artifact remains a **learned policy**
- reduced-order references and IK are moving into top-level `control/` as
  nominal-motion and training-support infrastructure
- `training/` remains the home for RL, eval, export, and locomotion curricula
- `runtime/` remains the hardware-facing layer and will absorb locomotion
  policy running plus shared logging
- `tools/` becomes the explicit home for SysID, replay, and sim2real analysis
- OCS2 / `wb_humanoid_mpc` and URDF migration remain long-term reference
  directions, not the active implementation target for current hardware

Architecture references:
- [docs/system_architecture.md](docs/system_architecture.md)
- [training/docs/walking_training.md](training/docs/walking_training.md) - active locomotion roadmap
- [training/docs/ToddlerBot_direction.md](training/docs/ToddlerBot_direction.md) - ToddlerBot-style pivot rationale
- [training/docs/standing_training.md](training/docs/standing_training.md) - standing branch history and regression gates
- [training/docs/footstep_planner_rl_adoption.md](training/docs/footstep_planner_rl_adoption.md)
- [training/docs/ocs2_humanoid_mpc_adoption.md](training/docs/ocs2_humanoid_mpc_adoption.md)

## Project Structure

Mainline platform split after the locomotion pivot:

```
wildrobot/
├── README.md                           # This file
├── AGENTS.md                           # Agent guidelines
├── CLAUDE.md                           # Claude AI guidelines
├── pyproject.toml                      # Python package configuration (uv)
├── docs/                               # Repo-level architecture and process docs
│   ├── system_architecture.md          # Mainline system architecture
│   ├── URDF_migration_plan.md          # Deferred URDF migration notes
│   ├── e2e_training_to_sim2real.md     # Sim2real flow notes
│   └── sim2real_policy_contract.md     # Policy contract docs
│
├── assets/                             # Digital twin and robot model files
│   ├── v2/                             # Active WildRobot asset variant
│   │   ├── wildrobot.xml               # Main MuJoCo robot model
│   │   ├── mujoco_robot_config.json    # Generated robot metadata
│   │   ├── scene_flat_terrain.xml      # Training scene
│   │   ├── sensors.xml                 # Sensor definitions
│   │   └── assets/                     # Generated Mesh files (STL)
│   ├── post_process.py                 # Post-process during model generation
│   ├── update_xml.sh                   # Regenerate XML/config from Onshape
│   └── robot_config.py                 # Robot config helpers
│
├── control/                            # Reduced-order references, IK, adapters, reusable non-PPO helpers
│   ├── interfaces.py                   # Shared controller / reference-side interfaces
│   ├── robot_model/                    # Robot-model interfaces and conventions
│   ├── reduced_model/                  # LIPM / DCM / ALIP-style helpers
│   ├── references/                     # Walking reference generators (Milestone 2+)
│   ├── kinematics/                     # Leg IK and nominal-motion utilities (Milestone 2+)
│   ├── locomotion/                     # Unified walking controller (v0.19.5)
│   │   ├── __init__.py
│   │   ├── walking_controller.py       # Orchestrator: ref → IK → [optional residual] → ctrl
│   │   │                               #   mode: nominal_only (v0.19.4) | residual_ppo (v0.19.5)
│   │   └── nominal_ik_adapter.py       # IK adapter: task-space targets → joint angles (NumPy)
│   ├── adapters/                       # Reference -> joint-target adapters
│   ├── execution/                      # Servo-friendly execution / tracking layer
│   └── mpc/                            # Reserved for future advanced planner work
│
├── training/                           # RL, eval, export, locomotion curricula
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
│   ├── configs/                        # Configuration system
│   │   ├── __init__.py
│   │   ├── training_config.py          # Training config schema
│   │   ├── training_runtime_config.py  # Runtime config
│   │   ├── ppo_walking_v0201_smoke.yaml # v0.20.1 walking smoke
│   │   └── ppo_standing.yaml           # Standing task config
│   │
│   ├── docs/                           # Training documentation
│   │   ├── walking_training.md         # Active locomotion roadmap
│   │   ├── ToddlerBot_direction.md     # ToddlerBot-style pivot note
│   │   ├── standing_training.md        # Standing branch history / regression gates
│   │   ├── footstep_planner_rl_adoption.md # RL-first + optional teacher plan
│   │   ├── ocs2_humanoid_mpc_adoption.md # Deferred OCS2 / humanoid MPC notes
│   │   └── archive/                    # Archived / superseded training docs
│   │       └── learn_first_plan.md     # Historical walking-first roadmap
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
│   ├── eval/                           # Evaluation tools
│   │   ├── eval_policy.py              # Headless evaluation
│   │   ├── eval_ladder_v0170.py        # Standing fixed ladder
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
│   │   ├── diagnose_torque.py
│   │   ├── gait_diagnostic.py
│   │   └── test_actuator_range.py
│   │
│   ├── checkpoints/                    # Saved model checkpoints
│   │   └── wildrobot_ppo_*/            # Training run checkpoints
│   │
│   └── wandb/                          # W&B experiment logs
│
├── policy_contract/                    # Shared sim2real contract (spec + JAX/NumPy implementations)
│   ├── spec.py                         # PolicySpec schema + validators + bundle loader
│   ├── spec_builder.py                 # Canonical PolicySpec builder (used by training + export + eval)
│   ├── jax/                            # JAX backend (training)
│   └── numpy/                          # NumPy backend (runtime + tooling)
│
├── runtime/                            # Hardware runtime and shared execution logs
│   ├── README.md                       # Runtime install + run instructions
│   ├── configs/                        # Runtime JSON config templates (hardware calibration source of truth)
│   │   ├── runtime_config_v2.json      # Standing/general runtime config
│   │   ├── walking_v0194.json          # Nominal-only walking deploy config (v0.19.4)
│   │   └── walking_v0195.json          # Residual PPO walking deploy config (v0.19.5)
│   ├── scripts/                        # Calibration tools (IMU/servo/footswitch)
│   └── wr_runtime/                     # Importable runtime package + CLIs
│       ├── control/
│       │   ├── run_policy.py           # Standing/general policy runner
│       │   ├── run_walking.py          # Walking controller entry point (v0.19.5)
│       │   ├── loc_ref_runtime.py      # Walking ref v1 runtime wrapper
│       │   └── loc_ref_runtime_v2.py   # Walking ref v2 runtime wrapper (v0.19.5)
│       └── locomotion/                 # Planned locomotion runner / command / log schema layer
│
├── tools/                              # Sim2real and platform tooling (Milestone 0+)
│   ├── sysid/                          # Actuator characterization and fitting tools
│   └── sim2real_eval/                  # Replay, comparison, and regression plotting tools
│
├── mujoco-brax/                        # Legacy mujoco brax training version.(reference only not used)
│   ├── README.md
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

## Architecture Split

Use this rule of thumb:

- `assets/` owns the digital twin
  - MJCF / scene files
  - generated robot metadata
  - sensor and asset definitions

- `control/` owns locomotion priors and reusable nominal-motion logic
  - reduced-order models
  - reference generation
  - IK and adapters
  - execution / tracking helpers
  - optional future planning experiments

- `training/` owns RL and evaluation
  - locomotion and standing experiments
  - training loops
  - fixed eval ladders
  - reward / curriculum work
  - logging and checkpointing
  - exportable learned policies
  - wrappers that call into `control/`

- `runtime/` owns hardware execution
  - runtime configs
  - policy runners
  - hardware-facing calibration scripts
  - shared runtime log schema

- `tools/` owns sim2real engineering support
  - SysID capture / fitting
  - replay and trace comparison
  - calibration and regression utilities

`training/` may depend on `control/`.  
`control/` should not depend on PPO-specific training code.

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

The commands below are for the PPO training stack in `training/`.

### Walking Task

```bash
# v0.20.1 walking smoke
uv run python training/train.py --config training/configs/ppo_walking_v0201_smoke.yaml

# Resume from checkpoint
uv run python training/train.py --resume training/checkpoints/<checkpoint_dir>
```

### Configuration

Training is configured via YAML files in `training/configs/`:

- `ppo_walking_v0201_smoke.yaml` - v0.20.1 walking smoke (offline ZMP prior + bounded residual)
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
- Achieve stable walking across flat and rough terrains
- Validate simulation accuracy with comprehensive test suite

## Key Features

- **JAX/MJX Acceleration**: Fast parallel simulation with JAX
- **Prior-Guided PPO**: Offline ZMP reference + bounded residual policy
- **Comprehensive Testing**: 10-layer test strategy from schema contracts to integration
- **Schema Contracts**: Prevents silent breakage from XML changes
- **W&B Integration**: Full experiment tracking and visualization

## Development Workflow

1. **Robot Model**: Define/update robot in `assets/v1/wildrobot.xml` (default) or `assets/v2/wildrobot.xml` (variant)
2. **Configuration**: Adjust training params in `training/configs/`
3. **Training**: Run experiments with `training/train.py`
4. **Testing**: Validate with `pytest training/tests/`
5. **Analysis**: Review metrics in W&B dashboard

## Documentation

Detailed documentation is available in `training/docs/`:

- [Walking Training Plan](training/docs/walking_training.md) - v0.20.x locomotion roadmap
- [Reference Design](training/docs/reference_design.md) - Offline ZMP prior + bounded residual contract
- [Test Strategy](training/tests/test_plan.md) - Comprehensive test plan
- [E2E Runbook](docs/e2e_training_to_sim2real.md) - Train → eval → export → sim2real

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your License Here]

## Acknowledgments

- MuJoCo and MJX teams at DeepMind
- JAX team at Google
- [Any other acknowledgments]
