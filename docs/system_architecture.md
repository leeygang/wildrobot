# WildRobot System Architecture

**Status:** ToddlerBot-style locomotion pivot active for `v0.19.x`  
**Last updated:** 2026-03-31

---

## Purpose

This document captures the repo-level architecture decision after the
standing-only exploration and the later ToddlerBot / LeRobot comparison.

The project is pivoting from a standing-first PPO branch toward a
locomotion-first robot-learning platform on current WildRobot v2 hardware.

The repo keeps three layers:
- `runtime/` and `assets/` own the hardware interface and digital twin
- `training/` owns RL, eval, export, and locomotion curricula
- top-level `control/` owns reduced-order models, references, IK, and reusable
  non-PPO helpers

The active mainline for current WildRobot hardware is now:
- **MJCF / MuJoCo remains the main simulation path**
- **the deployed controller remains a learned policy**
- **reference-guided RL becomes the active locomotion recipe**
- **reduced-order reference + IK provide nominal motion, not the final
  controller**
- **sim2real becomes an explicit calibration + SysID + evaluation loop**
- **walking becomes the primary product goal on current hardware**
- **standing remains a regression gate, not the top-level destination**

This is the closest match to the practical industry recipe for this robot class:
- simulator-native RL
- policy deployment to hardware
- structured locomotion priors
- actuator realism and transfer hardening
- explicit sim2real tooling

Active planning docs:
- [walking_training.md](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)
- [TodderlBot_direction.md](/home/leeygang/projects/wildrobot/training/docs/TodderlBot_direction.md)
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)
- [open_duck_op3_comparison.md](/home/leeygang/projects/wildrobot/training/docs/open_duck_op3_comparison.md)

Reference tiers for the active branch:

1. **Primary architecture and training reference**
   - ToddlerBot:
     - <https://toddlerbot.github.io/>
     - <https://arxiv.org/abs/2502.00893>
   This defines the mainline recipe shape:
   - digital twin quality
   - actuator realism and SysID
   - structured locomotion prior
   - learned runtime policy

2. **Reduced-order locomotion reference**
   - LIPM / DCM / capture-point
   - ALIP-inspired task-space learning:
     - <https://arxiv.org/abs/2309.15442>
   These define the reference-generator shape:
   - task-space step targets
   - reduced-order balance logic
   - nominal motion prior for RL

3. **Direct RL stack references**
   - MuJoCo Playground:
     - <https://github.com/google-deepmind/mujoco_playground>
   - Unitree RL Lab:
     - <https://github.com/unitreerobotics/unitree_rl_lab>
   These define the training and deployment recipe:
   - simulator-native RL
   - direct policy deployment
   - transfer hardening through training

4. **Infrastructure reference**
   - LeRobot:
     - <https://github.com/huggingface/lerobot>
   This is the reference for:
   - dataset and replay tooling
   - robot interface cleanup
   - reusable train/eval/deploy workflows

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Why The Architecture Changed

The standing exploration established useful baseline facts:
- quiet standing is learnable on current hardware
- PPO itself is not the main blocker
- the hard part is coordinated locomotion and whole-body recovery

The remaining gap is no longer "can RL stabilize standing?" The remaining gap
is:
- reliable step timing
- task-space foot placement
- trunk and pelvis coordination
- repeatable sim2real transfer for locomotion

Three directions were compared:

1. **Keep escalating free PPO**
   - low confidence after the standing branch already exposed the mechanism gap

2. **Jump to full OCS2 / humanoid MPC**
   - still a poor fit for current hardware:
     - position-controlled hobby servos
     - `50 Hz` runtime
     - thin sensing stack
     - no ankle roll

3. **Follow a ToddlerBot-style structured-learning route**
   - highest confidence on current hardware
   - keeps a learned runtime policy
   - adds a stronger locomotion prior and better sim2real discipline

That leads to the current conclusion:
- keep the learned-policy deployment recipe
- add reduced-order reference generation and IK as nominal-motion tools
- train a residual policy around that nominal motion
- make sim2real an explicit engineering loop rather than a later patch

---

## Current Mainline Architecture

The active mainline is reference-guided RL.

```text
MJCF / MuJoCo simulation
    ->
Calibration and SysID
    ->
Reduced-order walking reference
  (LIPM / DCM / capture-point first, ALIP later if needed)
    ->
IK nominal motion generator
    ->
Residual RL policy
    ->
Joint position targets
    ->
Current runtime / hardware path
```

### Layer 1: Simulation and digital twin

Responsibilities:
- keep `assets/v2/wildrobot.xml` as the main simulation model
- preserve MuJoCo contact tuning and actuator semantics
- provide the state path used by training, eval, and runtime parity checks
- absorb actuator, sensor, and latency realism from SysID work

### Layer 2: Calibration and SysID

Responsibilities:
- identify actuator delay, backlash, frictionloss, armature, and torque limits
- calibrate sensor noise and bias
- provide versioned realism parameters for the digital twin

### Layer 3: Reduced-order reference generation

Responsibilities:
- generate task-space locomotion references from command velocity and robot state
- provide gait phase, foothold targets, pelvis targets, and swing-foot targets
- begin with `LIPM / DCM / capture-point`
- later allow `ALIP` if angular-momentum effects matter enough to justify it

### Layer 4: IK nominal motion generation

Responsibilities:
- map task-space reference targets into nominal joint targets
- keep the nominal motion morphology-aware for WildRobot v2
- provide a clear baseline motion for policy residual learning

### Layer 5: RL training stack

Responsibilities:
- policy optimization
- rollout collection
- curriculum
- fixed evaluation
- exportable runtime-compatible policy contract
- residual correction around the nominal reference path

The deployed artifact remains the policy.

### Layer 6: Evaluation and deployment

Responsibilities:
- fixed locomotion and standing regression ladders
- sim2sim checks
- sim2real replay and mismatch analysis
- policy export and runtime deployment

The artifact being evaluated and deployed stays the policy. Reference and IK
layers are part of the locomotion stack, but they are not treated as a separate
standalone controller product.

---

## Architecture Principles

1. **Learned policy remains the deployed controller**
   - The runtime artifact remains a learned policy that matches the current
     hardware path.

2. **Reference-guided RL replaces reward-first discovery**
   - Reduced-order references and IK narrow the search space.
   - The policy learns robustness and adaptation around a nominal motion.

3. **IK is a nominal-motion tool, not the final controller**
   - IK provides the base posture and step realization.
   - The deployed control contract still centers on the learned policy.

4. **Keep MJCF / MuJoCo as the mainline sim path**
   - Do not force URDF or a new simulator into the active branch.

5. **Follow the existing industry recipe**
   - simulator-native RL
   - policy deployment
   - domain randomization / latency / noise hardening
   - structured locomotion prior
   - explicit sim2real tooling

6. **Keep heavyweight model-based stacks as long-term references**
   - OCS2, `wb_humanoid_mpc`, OpenLoong, and MJPC remain reference material.
   - They are not the immediate implementation contract for current hardware.

---

## What We Reuse From The Current Repo

These parts remain mainline assets:
- `assets/v2/wildrobot.xml`
- `assets/v2/mujoco_robot_config.json`
- current scene files in `assets/v2/`
- `training/envs/`
- `training/core/`
- `training/eval/`
- `runtime/configs/runtime_config_v2.json`
- current export/runtime flow
- the standing regression targets in
  [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)

These remain baseline comparison lines:
- `v0.14.6` hard-push standing baseline
- `v0.17.1` final PPO standing baseline

These remain references, not mainline dependencies:
- OCS2
- `wb_humanoid_mpc`
- URDF-first migration work

---

## What Stops Being Mainline

These paths stay documented, but they are not the active implementation plan:
- full OCS2 / humanoid MPC adoption on current hardware
- URDF-first migration as the mainline path
- standing-first PPO as the top-level product path
- unrestricted joint-space gait discovery from rewards alone
- mocap-first retargeting as the initial locomotion route
- open-ended reward-family growth as the main solution

---

## Folder Ownership

The controller-assist path should still live in **top-level `control/`**, not
`training/control/`.

Ownership rule:
- `control/` owns planner/teacher/helper logic that is not PPO-specific
- `training/` owns experiments, PPO baselines, evals, exports, and runtime
  integration

Practical boundary:
- `training/` may depend on `control/`
- `control/` should not depend on PPO-specific training code

What belongs in `control/`:
- robot model interfaces
- reduced-order helper logic
- locomotion references
- IK / kinematics helpers
- optional teacher or target generators
- adapters and reusable non-PPO support code
- reserved room for future advanced planner experiments

What stays in `training/`:
- PPO baseline stack
- reward and curriculum experiments
- training configs
- evaluation ladder
- W&B / logging / checkpointing
- policy export and runtime contract
- wrappers that call into `control/`

---

## Expected Repo Shape

```text
docs/
  system_architecture.md
  URDF_migration_plan.md            # deferred / conditional

training/docs/
  walking_training.md
  TodderlBot_direction.md
  standing_training.md
  footstep_planner_rl_adoption.md
  ocs2_humanoid_mpc_adoption.md     # deferred / reference only

control/
  README.md
  robot_model/
  reduced_model/
  references/
    locomotion_contract.py
  kinematics/
    interfaces.py
  mpc/                              # reserved for future advanced planner work
  execution/
  adapters/
    locomotion_action_adapter.py

runtime/wr_runtime/
  logging/
    locomotion_log_schema.py

tools/
  sysid/
  sim2real_eval/
```

In the active branch, `control/` exists to support training and deployment of
the locomotion policy stack, not to replace it with a heavyweight controller.

---

## Milestone Mapping

### `v0.19.0`
- architecture pivot
- locomotion-first roadmap
- shared observation / action contract

### `v0.19.1`
- actuator SysID
- digital twin realism
- sim2real replay tooling

### `v0.19.2`
- `walking_ref_v1`
- `LIPM / DCM / capture-point` reference
- IK nominal motion path

### `v0.19.3`
- reference-guided walking PPO
- forward walk and stop
- residual policy export

### `v0.19.4`
- transfer hardening
- history and delay
- domain randomization expansion

### `v0.19.5`
- in-place yaw
- broader command range
- repeatable zero-shot or near-zero-touch transfer

### `v0.19.6`
- disturbance-aware stepping
- moderate push recovery during standing and slow walking
- optional `ALIP` upgrade if the simpler reference saturates

---

## Decision Record

Mainline decision as of `2026-03-31`:
- **Use a ToddlerBot-style locomotion platform direction on current v2
  hardware**
- **Keep the deployed artifact as a learned policy**
- **Use reduced-order references + IK to provide nominal locomotion**
- **Train residual reference-guided RL rather than free gait discovery**
- **Keep MJCF / MuJoCo as the mainline sim and eval stack**
- **Make SysID and sim2real evaluation part of the mainline architecture**
- **Keep OCS2 / `wb_humanoid_mpc` as long-term references only**
