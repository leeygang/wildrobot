# Control Stack

This directory is the future mainline controller path for WildRobot.

Ownership:
- `control/` owns controller logic
- `training/` owns experiments, PPO baselines, evals, and training integration

Planned subdirectories:
- `robot_model/`
- `reduced_model/`
- `mpc/`
- `execution/`
- `adapters/`

This directory is intentionally minimal at the architecture-doc stage.
Implementation begins from the `v0.17.2+` OCS2 / humanoid MPC adoption plan in:
- `docs/system_architecture.md`
- `training/docs/ocs2_humanoid_mpc_adoption.md`
