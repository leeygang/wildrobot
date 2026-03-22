# WildRobot Step-Target Teacher + RL Plan

**Status:** Active mainline plan for `v0.17.3+`  
**Last updated:** 2026-03-22

---

## Purpose

This document defines the active post-`v0.17.1` implementation path for
WildRobot.

The key decision is:
- keep the existing MJCF / MuJoCo / JAX RL stack as the mainline
- keep the deployed artifact as a direct policy, not a planner-dependent
  controller
- use step-target or capture-point heuristics only as optional training-time
  teachers, labels, or reward scaffolds

This is the closest fit to the current industry recipe for practical humanoid
locomotion on position-controlled hardware:
- simulator-native training
- direct RL policy deployment
- domain randomization and sim2real hardening
- optional privileged or teacher signals during training

Read together with:
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)
- [system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Industry Recipe We Are Following

For current hardware, the mainline should look like:
- MJCF / MuJoCo or equivalent simulator-native model
- RL policy as the deployed runtime controller
- domain randomization / latency / noise modeling for transfer
- optional privileged training channels
- optional teacher or imitation signals when task geometry is hard to discover

Reference tiers for this plan:

### 1. Direct RL recipe references

- MuJoCo Playground:
  - https://github.com/google-deepmind/mujoco_playground
- Unitree RL Lab:
  - https://github.com/unitreerobotics/unitree_rl_lab

These define the mainline pattern:
- simulator-native RL
- direct policy deployment
- transfer hardening through training

### 2. Teacher / scaffold reference

- ModelBasedFootstepPlanning-IROS2024:
  - https://github.com/hojae-io/ModelBasedFootstepPlanning-IROS2024

This is the closest open-source reference for the exact structural idea we want:
- model-based step/foothold geometry
- model-free RL execution
- teacher-like guidance without turning the runtime stack into full MPC

### 3. Closest hardware-adjacent implementation reference

- adjacent Open Duck Mini project / runtime in `/home/leeygang/projects`

This is the practical hardware reference for:
- hobby-servo deployment reality
- similar runtime constraints
- policy export and runtime simplicity

Useful secondary references:
- OP3-style servo-humanoid walking stacks
- MuJoCo MPC only as a possible simulation teacher later

Not the active mainline:
- full OCS2 / `wb_humanoid_mpc`
- URDF-first migration
- runtime planner dependency for standing and walking

---

## Why This Is The New Mainline

`v0.17.1` showed:
- PPO can stabilize quiet standing
- PPO can learn useful recovery behavior
- the true hard-push gate still fails
- the remaining failure is still mostly `term_height_low`

What that means:
- the deployed control style is not obviously wrong
- the missing piece is likely training structure, step geometry supervision, or
  stronger sim2real-style hardening
- the next step should still look like the recipe used by practical RL humanoid
  projects, not a controller-stack pivot the hardware cannot exploit

So the active answer is:
- **RL-first mainline**
- **planner or capture-point logic only as optional training-time assistance**

---

## Hardware-Fit Constraints

Current hardware assumptions:
- position-controlled hobby servos
- `50 Hz` runtime
- one IMU
- four binary foot switches
- no ankle roll
- limited onboard compute

Implications:
- deployed policy should keep issuing position-friendly targets
- do not add a runtime dependency on torque-level controllers
- do not require whole-body force control
- do not assume precise contact wrench estimation
- keep the hardware runtime contract simple

This is why the active branch remains:
- RL-native
- servo-friendly
- deployable through the current export/runtime path

---

## Target Architecture

The active architecture is training-first and policy-first.

```text
MJCF / MuJoCo simulation
    ->
RL training stack
    ->
Optional teacher / scaffold path
  (capture-point target, need_step label, step-target reward, privileged signal)
    ->
Direct policy
    ->
Joint position targets
    ->
Current runtime / hardware path
```

### 1. Mainline deployed controller

This stays a policy.

Responsibilities:
- stabilize standing
- execute recovery on current actuators
- later support slow commanded walking

### 2. Teacher / scaffold path

This is optional and training-time only.

Allowed roles:
- `need_step` label
- target foot placement
- capture-margin scalar
- auxiliary reward target
- privileged teacher feature

Disallowed role:
- becoming the required runtime controller for deployment

### 3. Evaluation path

This remains policy evaluation, not planner evaluation.

Use:
- current fixed standing ladder
- direct comparison against `v0.14.6` and `v0.17.1`

---

## What The Step-Target Teacher May Provide

If used, the teacher should stay small and explicit.

Minimum outputs:
- `need_step`
- `swing_foot`
- local touchdown target
- optional capture urgency / capture margin

Allowed first model:
- capture-point / LIP heuristic

Allowed later escalation only if clearly needed:
- ALIP-lite

Not the first target:
- full runtime footstep planner
- full MPC
- phase-heavy handcrafted controller logic

---

## What The RL Policy Still Owns

The policy remains responsible for:
- joint-space execution quality
- posture stabilization
- landing softness and recovery transients
- compensation for actuation limits and model mismatch
- later walking behavior on the same deployed stack

This is the key rule:
- the teacher may shape learning
- the policy remains the deployed behavior generator

---

## Required Training-Recipe Features

To align with the industry recipe, the active branch should emphasize:

1. **Runtime-available actor observations**
   - actor input must stay compatible with deployment

2. **Privileged training channels only when justified**
   - teacher features, if used, should be clearly separated from runtime policy
     inputs

3. **Domain randomization and transfer hardening**
   - actuator variation
   - latency / delay
   - sensor noise
   - contact variation

4. **Sim2sim and deployment discipline**
   - keep checkpoint evaluation and export compatible with the existing runtime

5. **Bounded teacher logic**
   - step-target heuristics should stay small, inspectable, and removable

---

## Reuse From The Current Repo

Reuse directly:
- `assets/v2/wildrobot.xml`
- `assets/v2/mujoco_robot_config.json`
- current MuJoCo scenes
- `training/envs/wildrobot_env.py`
- disturbance generation and fixed eval ladder
- current runtime calibration/config path
- current policy export path

Reuse as baselines:
- `v0.14.6`
- `v0.17.1`

Do not delete:
- PPO training path
- current eval ladder
- current runtime path

---

## Implementation Plan

### `v0.17.3` - RL Recipe Alignment Groundwork

Deliver:
- document and enforce the active RL-first recipe
- separate runtime actor inputs from any privileged teacher inputs
- audit or add the needed training hooks for:
  - domain randomization
  - latency / delay modeling
  - sensor-noise modeling
  - sim2sim / export compatibility
- no deployed-controller change yet

Exit criteria:
- active training recipe is explicit
- deployment contract stays policy-only
- repo is ready for bounded step-target teacher experiments

### `v0.17.4` - Step-Target Teacher / Scaffold Integration

Deliver:
- optional capture-point or similar step-target heuristic
- privileged labels or auxiliary targets for the policy
- metrics and debug signals for teacher engagement
- no required runtime planner dependency

Exit criteria:
- teacher signals are inspectable and bounded
- policy can train with or without the teacher path

### `v0.17.5` - Standing Push Recovery

Deliver:
- hard-push standing branch on the industry-style RL path
- direct fixed-ladder comparison against `v0.17.1` and `v0.14.6`

Exit criteria:
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` materially below `v0.17.1`

### `v0.17.6` - Walking On The Same Policy Stack

Deliver:
- slow / conservative forward walking
- stop / hold
- gentle turn
- preserve standing push recovery as a regression suite

Exit criteria:
- one deployed policy stack supports both standing recovery and conservative
  walking

### `v0.17.7` - Optional Stronger Teachers

Possible directions only if needed:
- privileged teacher-student path
- imitation / reference-motion assistance
- MJPC-generated teacher data in simulation

Do not skip straight here unless the simpler branch clearly saturates.

---

## Diagnostics To Add

Policy / transfer diagnostics:
- latency sensitivity
- action saturation
- clean-standing regression
- export/runtime parity

Teacher diagnostics:
- teacher active fraction
- `need_step` frequency
- target-step statistics
- target-step tracking error if defined

Recovery diagnostics:
- first step latency
- touchdown count
- support-foot changes
- post-push residual velocity
- `term_height_low_frac`

---

## Immediate Next Actions

1. Make the RL-first deployment contract explicit in configs and docs.
2. Separate runtime actor observations from any privileged teacher path.
3. Add the minimum hooks for latency/noise/domain-randomization review.
4. Define the optional step-target teacher interface.
5. Keep the standing fixed ladder as the first hard acceptance test.

---

## Out Of Scope For The Active Branch

Do not make these the mainline right now:
- runtime planner-dependent controller execution
- full OCS2 adoption
- URDF-first migration
- full humanoid MPC / WBC
- open-ended reward-family expansion without recipe discipline
