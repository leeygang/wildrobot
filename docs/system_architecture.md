# WildRobot System Architecture

**Status:** Active mainline updated at `v0.17.3`  
**Last updated:** 2026-03-22

---

## Purpose

This document captures the repo-level architecture decision after the `v0.17.1`
standing result and the later hardware-fit review.

The repo keeps two paths:
- the existing PPO/AMP stack in `training/` as baseline, runtime, and eval
  infrastructure
- a new set of non-PPO helpers under `control/` for bounded teacher/planner
  utilities when needed

The active mainline for current WildRobot hardware is now:
- **MJCF / MuJoCo remains the main simulation path**
- **the deployed controller remains a direct RL policy**
- **planner or capture-point logic, if used, is training-time teacher/scaffold
  only**
- **standing push recovery is validated first**
- **walking is added later on the same deployed policy stack**

This is the closest match to the practical industry recipe for this robot class:
- simulator-native RL
- policy deployment to hardware
- domain-randomized transfer hardening
- optional teacher or privileged training signals

Active planning docs:
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)

Reference tiers for the active branch:

1. **Direct RL recipe references**
   - MuJoCo Playground:
     - <https://github.com/google-deepmind/mujoco_playground>
   - Unitree RL Lab:
     - <https://github.com/unitreerobotics/unitree_rl_lab>
   These define the mainline recipe shape:
   - simulator-native RL
   - direct policy deployment
   - transfer hardening through training

2. **Teacher / scaffold reference**
   - ModelBasedFootstepPlanning-IROS2024:
     - <https://github.com/hojae-io/ModelBasedFootstepPlanning-IROS2024>
   This is the closest open-source reference for:
   - model-based footstep targets
   - model-free RL execution
   - teacher-like geometry assistance without making the deployed artifact a
     full planner stack

3. **Closest hardware-class implementation reference**
   - adjacent Open Duck Mini project / runtime
   This is the grounding reference for:
   - hobby-servo deployment reality
   - similar runtime constraints
   - what the deployed policy path can plausibly support on small hardware

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Why The Architecture Changed

`v0.17.1` answered the PPO baseline question:
- quiet standing was stable
- PPO optimization itself was not the blocker
- internal push probes looked strong
- the fixed ladder still failed the true hard-push gate

The remaining gap was:
- hard-push step quality
- touchdown geometry
- post-touchdown arrest of collapse

Two follow-up ideas were examined and rejected as the mainline:

1. **Keep escalating PPO without changing the recipe**
   - low confidence after the fixed ladder already rejected the branch

2. **Jump to full OCS2 / humanoid MPC**
   - poor fit for current hardware:
     - position-controlled hobby servos
     - `50 Hz` runtime
     - thin sensing stack
     - no ankle roll

That leaves the more practical conclusion:
- keep the industry-standard RL deployment recipe
- improve training structure
- add step-geometry hints only as bounded teachers or privileged scaffolds

---

## Current Mainline Architecture

The active mainline is RL-first.

```text
MJCF / MuJoCo simulation
    ->
RL training stack
    ->
Optional teacher / scaffold path
  (need_step labels, capture-point targets, privileged signals)
    ->
Direct policy
    ->
Joint position targets
    ->
Current runtime / hardware path
```

### Layer 1: Simulation and robot model

Responsibilities:
- keep `assets/v2/wildrobot.xml` as the main simulation model
- preserve MuJoCo contact tuning and actuator semantics
- provide the state path used by training, eval, and runtime parity checks

### Layer 2: RL training stack

Responsibilities:
- policy optimization
- rollout collection
- curriculum
- fixed evaluation
- exportable runtime-compatible policy contract

This remains the main deployed-control path.

### Layer 3: Optional teacher / scaffold path

Responsibilities:
- provide bounded training-time assistance when task geometry is hard to learn
  from scratch

Allowed forms:
- `need_step` labels
- target step placement
- capture-margin signal
- privileged training inputs
- auxiliary reward targets

Disallowed role:
- becoming a required runtime controller dependency

### Layer 4: Evaluation and deployment

Responsibilities:
- fixed ladder comparison against `v0.14.6` and `v0.17.1`
- sim2sim checks
- policy export and runtime deployment

The artifact being evaluated and deployed stays the policy.

---

## Architecture Principles

1. **RL-first deployed controller**
   - The runtime artifact should remain a direct policy that matches the current
     hardware path.

2. **Teacher second, not planner-first**
   - Step geometry can be injected as labels, targets, or privileged signals.
   - It should not become a required runtime dependency unless later evidence
     justifies that jump.

3. **Keep MJCF / MuJoCo as the mainline sim path**
   - Do not force URDF or a new simulator into the active branch.

4. **Follow the existing industry recipe**
   - simulator-native RL
   - policy deployment
   - domain randomization / latency / noise hardening
   - optional teacher or imitation support

5. **Keep heavyweight model-based stacks as long-term references**
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
- planner-dependent runtime controller execution
- PPO-only escalation without recipe changes
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
  standing_training.md
  footstep_planner_rl_adoption.md
  ocs2_humanoid_mpc_adoption.md     # deferred / reference only

control/
  README.md
  robot_model/
  reduced_model/
  mpc/                              # reserved for future advanced planner work
  execution/
  adapters/
```

In the active branch, `control/` exists to support training and deployment of
the RL stack, not to replace it with a heavyweight controller.

---

## Milestone Mapping

### `v0.17.2`
- architecture review
- folder split and scaffolding
- documentation

### `v0.17.3`
- RL recipe alignment groundwork
- explicit runtime-vs-privileged input boundary
- prepare optional teacher/scaffold hooks

### `v0.17.4`
- bounded step-target teacher / scaffold integration

### `v0.17.5`
- standing push recovery on the RL mainline

### `v0.17.6`
- walking on the same deployed policy stack

### `v0.17.7`
- optional stronger teachers only if the simpler branch saturates

---

## Decision Record

Mainline decision as of `2026-03-22`:
- **Use an industry-style RL recipe as the active path**
- **Keep the deployed artifact as a policy**
- **Use step-target logic only as optional training-time assistance**
- **Keep MJCF / MuJoCo as the mainline sim and eval stack**
- **Keep OCS2 / `wb_humanoid_mpc` as long-term references only**
