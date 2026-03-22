# WildRobot OCS2 / Humanoid MPC Adoption Notes

**Status:** Deferred reference, not the active mainline for `v0.17.3+`  
**Last updated:** 2026-03-22

---

## Purpose

This document records why OCS2 / `wb_humanoid_mpc` was investigated, why it is
not the current mainline for WildRobot, and what parts of that architecture
remain useful as long-term reference material.

Active mainline doc:
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)

Repo-level context:
- [system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)

---

## Why It Was Investigated

After `v0.17.1` failed the true hard-push gate, the first interpretation was:
- pure PPO is not discovering good enough step geometry
- a more explicit model-based locomotion stack might solve both standing
  recovery and walking under one architecture

OCS2 / `wb_humanoid_mpc` was attractive because it offered:
- planner / execution separation
- switched-contact reasoning
- a standing-to-walking architectural path
- a more explicit handling of step placement than PPO-only training

That made it a reasonable architectural candidate to evaluate.

---

## Why It Is Not The Current Mainline

The hardware-fit review changed the decision.

Current WildRobot constraints:
- position-controlled hobby servos
- no torque-control interface
- `50 Hz` runtime loop
- one IMU and binary foot switches
- no ankle roll
- thin dynamic torque margin

Those constraints conflict with what the reference stacks are best at:
- torque or force-oriented execution
- higher-bandwidth control loops
- richer contact-state estimation
- stronger lateral stabilization hardware

Main blockers for direct adoption:

1. **Actuation mismatch**
   - Whole-body MPC / WBC stacks assume much more direct control authority than
     current hobby servos provide.

2. **Bandwidth mismatch**
   - WildRobot runtime is much slower than the controller loops typically
     assumed by these stacks.

3. **Lateral stabilization limits**
   - missing ankle roll materially reduces the classical balance toolbox.

4. **Infrastructure cost**
   - URDF / Pinocchio / OCS2 plumbing is substantial work before controller
     validation can even begin.

So for current hardware, the OCS2 path is architecturally interesting but not
the best near-term implementation target.

---

## What We Still Want To Borrow

Even though the full adoption is deferred, several ideas remain useful:

1. **Planner / execution separation**
   - step geometry should be explicit
   - joint-space execution should be a separate concern

2. **Standing and walking on one stack**
   - avoid building an unrelated walking controller later

3. **Model-based first where it helps**
   - planner logic should encode obvious locomotion structure
   - RL should not be forced to invent all of that from scratch

4. **Clear interfaces**
   - robot model
   - reduced-order planner
   - execution layer
   - simulation/runtime adapters

These ideas still shape the `control/` folder and the active hybrid plan.

---

## Repository Role Now

This document is now a reference note, not an implementation plan.

Use it for:
- long-term architecture thinking
- future hardware-upgrade planning
- naming the abstractions that should remain clean in `control/`

Do not use it to drive:
- `v0.17.3`
- `v0.17.4`
- `v0.17.5`

Those milestones are owned by the active hybrid plan in
[footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md).

---

## When To Revisit This Path

Revisit direct OCS2 / humanoid MPC adoption only if at least one of these is
true:

1. WildRobot hardware is upgraded materially:
   - higher-bandwidth actuators
   - torque or impedance control
   - richer contact sensing

2. The project chooses a more capable robot class:
   - one where whole-body MPC / WBC is actually a good fit

3. The current hybrid planner + RL branch clearly saturates:
   - planner geometry is still insufficient
   - RL tracking is no longer the main bottleneck

4. A simulation-only advanced controller branch becomes worthwhile as a
   separate research effort.

---

## What To Preserve From This Investigation

Keep these decisions:
- `control/` stays top-level
- controller logic stays separate from PPO-specific code
- reduced-order planning remains a valid abstraction
- standing-first validation still comes before walking

Keep these expectations realistic:
- full humanoid MPC is not a free upgrade for hobby hardware
- URDF migration is only worth it if the controller stack can actually exploit
  it

---

## Bottom Line

For current WildRobot hardware:
- OCS2 / `wb_humanoid_mpc` is **not** the active mainline
- it remains a **long-term reference**
- the active path is **simple footstep planner + RL tracking**
