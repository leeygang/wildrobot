# WildRobot Option B: URDF-First Migration Plan

**Status:** Proposed  
**Last updated:** 2026-03-22

---

## Goal

Adopt a **URDF-first control model** for WildRobot so the new mainline controller stack can align cleanly with:
- OCS2 as the optimal-control backend
- humanoid MPC stacks such as `wb_humanoid_mpc`

This does **not** mean deleting MuJoCo or abandoning the current MJCF assets immediately.

The intended end state is:
- **URDF is canonical for control**
- **MJCF remains available for MuJoCo simulation and regression**
- both descriptions stay synchronized

---

## Why Option B

Option B is the cleanest long-term architecture because:
- it aligns with the expected model path of OCS2 / Pinocchio-based humanoid control stacks
- it reduces controller-side friction compared with building a custom MuJoCo bridge first
- it supports the new project goal of using one architecture for:
  - standing push recovery
  - walking

The tradeoff is:
- more upfront model migration work
- lower short-term continuity than staying purely MuJoCo-first

---

## Core Decision

Make a clear separation:

- **Control model**
  - canonical source for controller-facing kinematics/dynamics
  - URDF-first

- **Simulation model**
  - canonical source for MuJoCo simulation behavior
  - MJCF remains active during migration

The migration succeeds only if these two stay consistent enough that:
- controller failures are real controller failures
- simulation failures are not caused by model drift

---

## What Changes

### Becomes canonical for control

- URDF robot description
- frame conventions used by the controller stack
- joint and contact naming contract
- control-model metadata required by Pinocchio / OCS2 / humanoid MPC

### Remains active during migration

- `assets/v2/wildrobot.xml`
- MuJoCo scenes
- current standing eval ladder
- current PPO baselines

### Stops being mainline

- MJCF as the only authoritative model for locomotion-control design
- direct-action PPO as the main architecture for standing+walking

---

## Migration Principles

1. **Do not delete MuJoCo early**
   - MuJoCo remains useful for simulation, regression, and comparison.

2. **Do not try to keep two independent robot models by hand**
   - URDF and MJCF must be synchronized through a documented source-of-truth process.

3. **Standing-first validation remains**
   - quiet standing and push recovery come before walking in the new stack.

4. **Model consistency before controller complexity**
   - the first risk is model drift, not planner sophistication.

5. **Prefer staged integration**
   - model audit
   - URDF/control path
   - standing bring-up
   - push recovery
   - walking

---

## Phase Plan

### Phase 0: Decision And Baseline Freeze

Objective:
- freeze the baseline comparison points before migration

Deliverables:
- record `v0.14.6` as historical hard-push baseline
- record `v0.17.1` as the final PPO standing baseline
- keep the fixed standing eval ladder unchanged

Exit criteria:
- baseline comparisons are stable and documented

---

### Phase 1: Model Inventory And Source-Of-Truth Decision

Objective:
- identify what currently exists and what becomes canonical

Tasks:
1. inventory active WildRobot assets:
   - `assets/v2/wildrobot.xml`
   - `assets/v2/mujoco_robot_config.json`
   - `runtime/configs/runtime_config_v2.json`
2. determine whether a maintained URDF already exists
3. define the model ownership rule:
   - either CAD/Onshape is the upstream source and both URDF/MJCF are generated
   - or one robot description becomes canonical and the other is generated/validated from it

Required outputs:
- explicit source-of-truth statement
- model artifact map
- conversion/validation ownership

Exit criteria:
- the repo has a written answer to:
  - what is canonical for control?
  - what is canonical for MuJoCo simulation?
  - how are they kept synchronized?

---

### Phase 2: URDF Control Model Bring-Up

Objective:
- establish a controller-facing URDF model path

Tasks:
1. create or validate the URDF control model for WildRobot
2. verify:
   - link hierarchy
   - joint directions
   - inertias
   - limits
   - foot frames
3. document frame conventions:
   - world
   - base/root
   - pelvis if distinct
   - left/right foot
   - heading-local standing frame

Required outputs:
- canonical URDF path
- frame convention document
- link/joint/contact naming map

Exit criteria:
- the controller-facing model is complete enough for Pinocchio/OCS2-style use

---

### Phase 3: MJCF <-> URDF Consistency Validation

Objective:
- ensure simulation and control models are close enough to trust

Tasks:
1. compare URDF and MJCF on:
   - total mass
   - link inertias
   - joint limits
   - foot positions in nominal stance
   - support polygon geometry
2. build a validation script/report for:
   - model drift
   - frame mismatch
   - contact geometry mismatch

Required outputs:
- consistency checklist
- tolerances for acceptable drift
- explicit list of known intentional differences

Exit criteria:
- the models are close enough that controller evaluation in MuJoCo is meaningful

---

### Phase 4: Control Stack Integration Groundwork

Objective:
- integrate the URDF-first model with the new control architecture without yet claiming stable standing

Tasks:
1. define top-level `control/` interfaces:
   - robot model
   - reduced model
   - planner
   - execution layer
   - adapters
2. make `control/robot_model/` own:
   - frame definitions
   - joint/contact maps
   - controller-facing model metadata
3. keep `training/` as integration/eval owner only

Required outputs:
- code skeleton in `control/`
- documented adapter boundary to simulation
- controller initialization path in simulation

Exit criteria:
- the repo has a clean controller/training split
- the controller path can execute in simulation for debug and integration work

---

### Phase 5: Quiet Standing

Objective:
- quiet standing on the new controller stack

Tasks:
1. initialize controller on URDF-first model
2. connect outputs to simulation path
3. validate:
   - stable neutral standing
   - coherent controller signals
   - no frame/sign convention mistakes

Exit criteria:
- quiet standing is stable and inspectable

---

### Phase 6: Standing Push Recovery

Objective:
- solve the original standing push-recovery task on the new stack

Tasks:
1. moderate push tests
2. hard-push evaluation with the existing ladder
3. compare against:
   - `v0.17.1`
   - `v0.14.6`

Exit criteria:
- `eval_medium` and `eval_hard` beat the PPO baseline line

---

### Phase 7: Walking On The Same Stack

Objective:
- extend the new stack from standing to walking

Tasks:
1. add walking references/commands in the model-based controller
2. preserve standing push recovery as a regression suite

Exit criteria:
- one architecture handles both standing recovery and walking

---

## Repository Changes Expected

### New or promoted paths

```text
docs/
  system_architecture.md
  URDF_migration_plan.md

control/
  robot_model/
  reduced_model/
  mpc/
  execution/
  adapters/
```

### Existing paths that remain important

```text
assets/v2/
training/eval/
training/docs/standing_training.md
training/CHANGELOG.md
runtime/configs/
```

---

## Risks

### 1. Model drift between URDF and MJCF

This is the biggest risk.

If the control model and simulation model diverge, the migration becomes hard to trust.

Mitigation:
- explicit validation scripts
- declared source-of-truth policy
- tolerances and drift checks

### 2. Hidden frame/sign mismatches

Typical failures:
- left/right foot swapped logically
- wrong heading-local sign convention
- joint direction mismatch

Mitigation:
- explicit frame map
- nominal pose checks
- controller signal inspection during quiet standing

### 3. Over-scoping the execution layer too early

Trying to build a full custom WBC stack immediately increases risk sharply.

Mitigation:
- start with the smallest execution layer that can validate standing

---

## Recommendation On MuJoCo During Migration

Do **not** drop MuJoCo early.

Recommended stance:
- keep MuJoCo as the main simulation and evaluation environment during migration
- move the **control model** to URDF-first
- use consistency validation between URDF and MJCF

This gives:
- cleaner controller alignment
- preserved simulation/eval investment
- lower transition risk than deleting the current sim path

---

## MuJoCo And URDF

MuJoCo does support URDF loading, and the native viewer can render the compiled model once loaded. Official MuJoCo docs state that:
- MuJoCo can load URDF files
- the `compile` sample converts between `MJCF`, `URDF`, and `MJB`
- URDF support is limited to a subset of what MuJoCo can represent
- in practice, MuJoCo recommends converting URDF to MJCF and then working in MJCF for full MuJoCo use

Sources:
- MuJoCo modeling docs: <https://mujoco.readthedocs.io/en/stable/modeling.html>
- MuJoCo code samples / compile tool: <https://mujoco.readthedocs.io/en/3.1.4/programming/samples.html>

Important caveats from the official docs:
- URDF support is a subset
- MuJoCo-specific URDF extensions are less rigorously checked than MJCF
- mesh path handling and other compiler conventions often need extra care

Practical conclusion:
- yes, MuJoCo viewer can be part of a URDF-first workflow
- but for robust simulation work, it is usually better to convert URDF to MJCF and keep using MJCF inside MuJoCo

---

## Other Simulator Options With Strong URDF Support

If the project later wants a more URDF-native simulator path than MuJoCo, the most relevant options are:

1. **Isaac Sim / Isaac Lab**
   - official importers for both URDF and MJCF
   - strong robotics tooling and modern humanoid workflow
   - sources:
     - <https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html>
     - <https://isaac-sim.github.io/IsaacLab/v2.0.0/source/refs/release_notes.html>

2. **Gazebo Sim**
   - official URDF spawn/import workflow
   - converts URDF through sdformat for simulation
   - source:
     - <https://gazebosim.org/docs/latest/spawn_urdf/>

3. **Drake**
   - strong URDF parsing and model-based robotics stack
   - more control/research oriented than MuJoCo-style interactive simulation
   - source:
     - <https://drake.mit.edu/doxygen_cxx/group__multibody__parsing.html>

These are alternatives, not immediate replacements for WildRobot.

For Option B, the recommended path is still:
- keep MuJoCo during migration
- make URDF canonical for control
- revisit simulator replacement only if MuJoCo becomes the bottleneck

---

## Immediate Next Actions

1. Decide the control-model source of truth.
2. Identify whether an authoritative WildRobot URDF already exists or must be generated.
3. Write the frame/contact convention document.
4. Define the URDF <-> MJCF consistency checks.
5. Start the `control/robot_model/` contract around the URDF-first model.
