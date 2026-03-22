# Control Stack

This directory is the non-PPO controller and planner path for WildRobot.

Ownership:
- `control/` owns controller logic
- `training/` owns experiments, PPO baselines, evals, and training integration

Planned subdirectories:
- `robot_model/`
- `reduced_model/`
- `mpc/` (reserved for advanced planning work if needed later)
- `execution/`
- `adapters/`

This directory is intentionally minimal at the architecture-doc stage.

Current mainline implementation starts from the hybrid planner + RL plan in:
- `docs/system_architecture.md`
- `training/docs/footstep_planner_rl_adoption.md`

Deferred long-term references:
- `training/docs/ocs2_humanoid_mpc_adoption.md`
