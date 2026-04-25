# v0.17: Standing Stabilization with Reactive Stepping

**Version:** v0.17.0  
**Status:** Frozen. Final selected standing checkpoint is `v0.17.4t` iter `330`. Later `step`, `crouch`, and `crouch-teacher` branches are retained as historical negative results, not active work.
**Created:** 2026-03-20  
**Last updated:** 2026-03-31

---

## Purpose

This document records the completed standing-first roadmap for WildRobot.

This document is now **frozen**. The active roadmap has moved to
[walking_training.md](/home/leeygang/projects/wildrobot/training/docs/walking_training.md).
Older walking-first roadmap material remains archived under
`training/docs/archive/learn_first_plan.md`.

Standing success means:
- stand quietly under zero command
- survive pushes
- step only when bracing is no longer enough
- return to upright neutral standing after recovery

Walking remains part of the broader goal, but this document no longer defines
the active execution plan for that work.

The final project-level standing decision is:
- keep `v0.17.4t` as the selected standing baseline
- freeze further standing-only training on the current branch
- use the validated standing checkpoint as the quiet-stance and recovery
  baseline for the locomotion pivot
- move active development to the ToddlerBot-style locomotion roadmap

Active architecture docs:
- [docs/system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)
- [open_duck_op3_comparison.md](/home/leeygang/projects/wildrobot/training/docs/open_duck_op3_comparison.md)
- [teacher_training_design.md](/home/leeygang/projects/wildrobot/training/docs/teacher_training_design.md)

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Final Standing Result

The final selected standing checkpoint for WildRobot is:

- run:
  `training/wandb/run-20260324_225425-zn8r3l5g`
- checkpoint:
  `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`

Best checked fixed-ladder result:
- `eval_medium = 93.8%`
- `eval_hard = 70.6%`
- `eval_hard/term_height_low_frac = 29.4%`

Best internal push eval at the selected checkpoint:
- `eval_push/success_rate = 93.75%`
- `eval_push/episode_length = 478.9`
- `eval_push/term_height_low_frac = 6.25%`

Verdict:
- `v0.17.4t` cleared the real standing hard gate
- it is the best validated standing branch in the full `v0.17` sequence
- later RL follow-ons did not beat it on the fixed ladder
- this is the checkpoint to keep for standing baseline, viewer demos, and
  future standing regression tests

Diagnosis:
- quiet standing and fixed-ladder standing recovery are solved well enough to
  freeze the branch
- the remaining locomotion problem is no longer "pick a better standing
  checkpoint"
- the remaining work is locomotion architecture, nominal motion generation, and
  sim2real transfer for walking

Freeze decision:
- keep `v0.17.4t` iter `330` as the final standing checkpoint
- do not resume `v0.17.4t-step`, `v0.17.4t-crouch`, or
  `v0.17.4t-crouch-teacher`
- keep those later runs only as evidence about what did not improve the branch
- move active work to the walking roadmap beginning at `v0.19.0`

Standing regression references:
- selected checkpoint:
  `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`
- previous PPO baseline:
  `v0.17.1`
- historical hard-push PPO comparison:
  `v0.14.6`

---

## External References

v0.17 now uses two categories of references:
- historical references for the completed PPO baseline
- active references for the industry-style RL mainline

### Historical Baseline References: `v0.17.0` - `v0.17.1`

#### Recipe A: `legged_gym` / Rudin et al. 2022

- Paper: https://proceedings.mlr.press/v164/rudin22a.html
- Code: https://github.com/leggedrobotics/legged_gym

Used for:
- PPO structure
- curriculum-first training
- compact reward design
- fixed eval discipline

#### Recipe B: Li et al. 2024 Cassie

- Project: https://xbpeng.github.io/projects/Cassie_Locomotion/index.html

Used for:
- treating standing recovery as a first-class skill
- disturbance exposure
- history as a later escalation if needed

### Active Mainline References: `v0.17.3+`

#### Direct RL recipe references

- MuJoCo Playground:
  - https://github.com/google-deepmind/mujoco_playground
- Unitree RL Lab:
  - https://github.com/unitreerobotics/unitree_rl_lab

These matter because they define the active recipe shape:
- simulator-native training
- direct policy deployment
- transfer hardening through training

#### Teacher / scaffold reference

- ModelBasedFootstepPlanning-IROS2024:
  - https://github.com/hojae-io/ModelBasedFootstepPlanning-IROS2024

Relevant because:
- it is the closest open-source reference for model-based step geometry helping
  model-free RL execution
- it matches the structural idea of bounded teacher assistance better than a
  full controller-stack pivot

#### Primary hardware-adjacent implementation reference

- adjacent Open Duck Mini project / runtime

Relevant because:
- it is much closer to WildRobot's hardware class than large torque-controlled
  humanoid stacks
- it keeps attention on what can actually deploy on current hardware

#### Long-term architecture references

Useful, but not the active mainline:
- OCS2
- `wb_humanoid_mpc`
- OpenLoong
- MJPC

---

## Repo Starting Point

The standing history still gives a clear baseline.

### What Worked

- `v0.13.8` established strong moderate-push standing robustness
- `v0.13.10` added explicit posture return
- `v0.14.6` pure PPO at the calibrated hard-push regime reached the best
  historical hard-push checkpoint:
  - `eval_push/success_rate = 60.84%` at `10N x 10`
  - dominant failure still `term_height_low`

### What Did Not Work

- `v0.14.7` privileged critic did not beat pure PPO
- `v0.14.8` actor-side capture-point observation did not beat pure PPO
- `v0.14.9` guide-only FSM underperformed and remained weakly engaged
- `v0.17.1` cleaned PPO baseline still failed the true hard gate

### Quantitative Boundary Data

Existing force sweep:

| Push Regime | Success Rate | Episode Length | Interpretation |
|---|---:|---:|---|
| `8N x 10` | `74.6%` | `431.8` | mostly bracing-manageable |
| `9N x 10` | `58.0%` | `382.9` | boundary / transition regime |
| `10N x 10` | `47.3%` | `346.7` | hard-push regime |

Planning implication:
- `8N x 10` is often recoverable by bracing
- `9N x 10` is where stepping should start to matter
- `10N x 10` is the real hard-push target

---

## Hardware Reality

Current hardware matters to the architecture choice.

Current constraints:
- position-controlled hobby servos
- `50 Hz` runtime
- one IMU
- four binary foot switches
- no ankle roll

Implications:
- keep the deployed controller simple
- avoid torque-control assumptions
- avoid planner-dependent runtime architectures
- use training-time structure before controller-stack replacement

Aggressive whole-body dynamic recovery is **not** the immediate target on this
hardware.

---

## Strategy

The completed `v0.17` branch used this sequence:

1. Use `v0.17.1` as the final PPO standing baseline result.
2. Keep MJCF / MuJoCo and the current RL infrastructure.
3. Fix the proven recipe gaps from the closest small-servo RL baselines:
   - residual action around a standing default pose
   - recovery-friendly termination
   - dominant alive / survival reward
   - domain randomization
   - action / IMU delay and sensor noise
4. Re-evaluate the hard-push gate on the recipe-fixed pure-RL branch.
5. Add bounded teacher or scaffold signals only if the geometry gap remains
   after that rerun.
6. Validate standing push recovery first.
7. Freeze the standing winner and move to locomotion work.

This avoids four failure modes:
- endless PPO-only reward patching without recipe discipline
- premature teacher / hierarchy work before the execution layer is fixed
- a controller pivot the hardware cannot use well
- splitting standing and walking into unrelated deployment stacks

Current branch decision:
- standing work is frozen at `v0.17.4t`
- the active roadmap is now [walking_training.md](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)

---

## Core Design

### Task Definition

Task:
- command starts at zero velocity
- robot starts from neutral standing
- pushes arrive during the episode
- success means surviving the push and returning to upright standing

Later walking extension:
- add conservative forward / turn / stop commands on the same policy stack

### Observation Design

The deployed actor should stay runtime-compatible.

Required runtime observations:
- projected gravity
- base angular velocity
- heading-local base linear velocity
- joint positions
- joint velocities
- previous action
- foot contact signals

Optional training-only signals:
- teacher labels
- privileged capture-margin signals
- target-step annotations

Rule:
- do not make runtime deployment depend on planner-only signals unless the
  architecture decision is revisited explicitly

### Teacher / Scaffold Design

If used, teacher logic should stay small and removable.

Allowed first teacher forms:
- `need_step` labels
- target foot placement
- capture-point / LIP targets
- auxiliary reward alignment

Not the first branch:
- full runtime footstep planner
- full MPC
- full WBC

### Policy Design

The policy remains the deployed controller.

Responsibilities:
- joint-space execution
- posture stabilization
- recovery transients
- compensation for actuation limits and model mismatch
- later conservative walking behavior

### Training Recipe Design

The mainline should emphasize:
- residual actions around a standing default pose
- recovery-friendly termination
- dominant survival reward
- curriculum
- domain randomization
- actuator/noise/delay hardening
- fixed eval ladder
- optional privileged/teacher channels only after the recipe-fixed rerun if
  justified

### Evaluation

Keep the same standing ladder as the first gate:
- `eval_clean`
- `eval_easy`
- `eval_medium`
- `eval_hard`
- `eval_hard_long`

Primary decision metrics:
- `eval_medium`
- `eval_hard`
- `term_height_low_frac`

---

## Milestones

### `v0.17.2`: Planning / Scaffolding

Objective:
- architecture review, folder split, and doc/scaffolding work

Status:
- complete

### `v0.17.3`: Recipe-Fixed Pure RL

Objective:
- align the active branch with the closest proven RL recipe on similar hardware
- fix 5 recipe gaps identified in `open_duck_op3_comparison.md` that explain why
  pure PPO cannot learn push recovery on this hardware
- the planner+RL hybrid path (original v0.17.3 plan) is **postponed** — the
  recipe gaps would block that approach too

Sequence:
```
v0.17.3a: Core recipe fix (home-centered action + relaxed termination + alive bonus)
  result: eval_medium = 83.1%, eval_hard = 51.4% on best checked checkpoint
v0.17.3a-metrics: Recovery-window diagnostics patch
  gate: recovery metrics visible and interpretable in training/eval
v0.17.3a-pitchfix: Cheap sim-side pitch-arrest probe
  gate: eval_hard > 60%?
    yes → v0.17.5
    no  → v0.17.3a-arrest
v0.17.3a-arrest: Recovery-gated step-quality rewards
  gate: eval_hard > 60%?
    yes → v0.17.5
    no  → v0.17.4t
v0.17.3b: Sim2real hardening (domain randomization + delay/noise)
  only after the sim hard-push gate is already passed
```

#### `v0.17.3a`: Core Recipe Fix

Three tightly-coupled changes applied together.

**Change 1: Home-centered action mapping**

Current: `ctrl = action * policy_action_sign * per_joint_span + per_joint_center`
New: `ctrl = clip(home + action * policy_action_sign * per_joint_span, range_min, range_max)`

Where:
- `per_joint_span = max(abs(range_min - home), abs(range_max - home))`
- `home ≈ 0 rad` for all joints (confirmed from `assets/v2/keyframes.xml`)
- So effectively: `per_joint_span = max(abs(range_min), abs(range_max))`

Why:
- The current mapping puts action=0 at the midpoint of each joint range, which
  is a crouched/bent pose for most joints. The policy must learn a non-zero
  constant just to stand.
- With the new mapping, action=0 = home = standing pose.
- Per-joint spans preserve action resolution. A universal span (e.g., 180°)
  would waste most of the [-1, +1] range on joints like knee [0°, 120°]
  (post-2026-04-25 widening from 80°) where ~33% of actions would clip.
  Per-joint spans ensure action=±1 reaches the further limit, with only
  the short side clipping (minimally).
- The runtime `target_rad → servo units` path is completely unchanged — it uses
  its own `motor_center_mujoco_deg` and `motor_sign`, independent of the
  action mapping.

Per-joint span examples (home = 0 for all):

| Joint | range (rad) | per_joint_span | action=+1 | action=-1 |
|---|---|---|---|---|
| left_hip_pitch | [-0.087, 1.484] | 1.484 | 1.484 ✓ | -1.484 → clip -0.087 |
| left_knee_pitch | [0.0, 2.094] | 2.094 | 2.094 ✓ | -2.094 → clip 0.0 |
| left_ankle_pitch | [-0.698, 0.785] | 0.785 | 0.785 ✓ | -0.785 → clip -0.698 |
| waist_yaw | [-0.524, 0.524] | 0.524 | 0.524 ✓ | -0.524 ✓ |
| left_shoulder_pitch | [-0.524, 3.142] | 3.142 | 3.142 ✓ | -3.142 → clip -0.524 |

Implementation surface — the actual action→ctrl path:

The training/eval/export path uses `policy_contract`, NOT `training/cal/cal.py`.
The call chain is:
- `wildrobot_env.py:1125` → `JaxCalibOps.action_to_ctrl(spec=self._policy_spec, ...)`
- `wildrobot_env.py:829` → `JaxCalibOps.ctrl_to_policy_action(spec=..., ...)`
- Both dispatch to `policy_contract/jax/calib.py` which reads
  centers/spans from `spec.robot.joints`
- The spec is built by `build_policy_spec()` which currently hardcodes
  `mapping_id="pos_target_rad_v1"` — this must accept the new mapping_id.

The home position should use the actual `home_ctrl_rad` from the policy spec
(already supported by `spec_builder.py:22`), not assume zero. For the current
asset they are equivalent (~0 rad from keyframe), but using the spec value
keeps export/runtime parity exact.

The change must go through `policy_contract`, with CAL kept in sync.

Files — policy_contract layer:
- `policy_contract/spec.py` — add `"pos_target_home_v1"` to
  `SUPPORTED_MAPPING_IDS`
- `policy_contract/spec_builder.py` — accept `mapping_id` parameter, pass it
  through to `ActionSpec`; also pass `home_ctrl_rad` from keyframe
- `policy_contract/jax/calib.py` — add `pos_target_home_v1` branch in
  `action_to_ctrl()` / `ctrl_to_policy_action()`: center on `home_ctrl_rad`,
  per-joint span, clip to range. Keep `pos_target_rad_v1` for backward compat.
- `policy_contract/numpy/calib.py` — mirror for runtime
- `policy_contract/jax/obs.py` — update `normalize_joint_pos` to use the same
  home-centered span so obs=0 means "at home"
- `policy_contract/numpy/obs.py` — mirror for runtime

Files — `build_policy_spec()` callsites (must pass `mapping_id`):
- `training/envs/wildrobot_env.py:721` — env init
- `training/train.py:352` — checkpoint fingerprint
- `training/core/training_loop.py:1139` — training loop spec
- `training/eval/visualize_policy.py:87` — visualization
- `training/exports/export_policy_bundle.py:290` — ONNX export

All five callsites must read `mapping_id` from the training config and pass
it to `build_policy_spec()`. If any callsite uses a different mapping_id,
checkpoint fingerprinting will detect the drift.

Files — sync and docs:
- `training/cal/cal.py` — keep in sync (same span formula in
  `_precompute_arrays()`)
- `training/configs/training_config.py` — wire `mapping_id` from YAML
- `docs/joints_angle.md` — update section 2 equations
- Runtime path (`runtime/wr_runtime/hardware/actuators.py`) — no change needed

**Change 2: Relaxed termination**

Current: height < 0.40m OR pitch/roll > 0.8 rad
New: height < 0.15m (near-ground only, no orientation termination)

Rationale: the agent must experience and recover from large tilts during
training. Terminating at 0.8 rad pitch prevents learning recovery from pushes
that cause significant lean.

Files:
- `training/envs/wildrobot_env.py` (`_get_termination`) — add
  `use_relaxed_termination` flag; when enabled, only terminate on
  `height < min_height` (0.15m)
- `training/configs/training_runtime_config.py` — add config field

**Change 3: Large alive bonus**

Current: `base_height` weight = 1.0–2.0 (binary healthy)
New: dedicated `alive` reward term with weight = 15.0

Rationale: the dominant signal should be "stay alive as long as possible."
Standing quality, orientation, posture, etc. are secondary shaping terms. This
follows the Open Duck / standard RL recipe pattern.

Files:
- `training/envs/wildrobot_env.py` (`_get_reward`) — add `alive` term:
  constant 1.0 every step
- Config: `alive: 15.0`, reduce `base_height` to 0.5, `orientation` to -1.0,
  zero out `collapse_height` and `collapse_vz` (pre-collapse shapers at 0.40m
  are no longer useful with termination at 0.15m)

**Config: `ppo_standing_v0173a.yaml`**

Key settings:
```yaml
version: "0.17.3a"
version_name: "Recipe Fix - Residual Actions + Relaxed Term + Alive Bonus"

env:
  action_mapping_id: pos_target_home_v1  # action=0 → home, per-joint span
  use_relaxed_termination: true
  min_height: 0.15             # only terminate near ground
  action_filter_alpha: 0.3     # reduce from 0.6 — residual actions bounded

reward_weights:
  alive: 15.0                  # dominant survival signal
  base_height: 0.5             # reduced (alive handles survival)
  orientation: -1.0            # less punishing (relaxed term allows recovery)
  height_target: 0.5
  collapse_height: 0.0         # removed (termination at 0.15m)
  collapse_vz: 0.0             # removed
  torque: -0.001
  saturation: -0.1
  action_rate: -0.01
  joint_velocity: -0.001
  slip: -0.5
  posture: 0.5                 # pulls robot back to standing after recovery
  posture_sigma: 0.40
  step_event: 0.12
  foot_place: 0.30
```

**Verification:**
```bash
# Smoke test — verify config loads and env initializes
uv run python training/train.py \
  --config training/configs/ppo_standing_v0173a.yaml --verify

# Training run
uv run python training/train.py \
  --config training/configs/ppo_standing_v0173a.yaml
```

Expected: within 100 iterations the policy should learn quiet standing (since
action=0 is already standing). Push recovery should start emerging as training
progresses.

#### `v0.17.3a` exit gate

v0.17.3a passes if:
- `eval_clean/success_rate > 95%` (quiet standing not broken)
- `eval_hard/success_rate` is measured and logged (no minimum required —
  this is the baseline for comparison)
- Action=0 produces stable standing in visualization (no drift, no crouch)
- Policy export + runtime parity check passes

v0.17.3a fails if:
- Quiet standing regresses below 90%
- Training diverges within 100 iterations

After `v0.17.3a`, proceed to `v0.17.3a-metrics` and then the sim-side
follow-ups (`v0.17.3a-pitchfix`, `v0.17.3a-arrest`). `v0.17.3b` remains a
later sim2real-hardening branch only after the simulation hard-push gate is
already passed.

#### `v0.17.3a` result

Completed run:
- run: `training/wandb/run-20260323_011338-0gmualhi`
- best checked checkpoint:
  `training/checkpoints/ppo_standing_v0173a_v0173a_20260323_011340-0gmualhi/checkpoint_210_27525120.pkl`

Best checked fixed-ladder result:
- `eval_medium = 83.1%`
- `eval_hard = 51.4%`
- `eval_hard/term_height_low_frac = 48.6%`

Interpretation:
- `v0.17.3a` is a real pure-RL improvement over `v0.17.1`
- quiet standing is solved
- the remaining gap is still simulation competence, not transfer
- `v0.17.3b` is therefore **not** the next branch for improving fixed-ladder
  sim performance

#### `v0.17.3a-metrics`: Recovery-Window Diagnostics Patch

Objective:
- make the next standing run diagnosable at the step-quality level
- distinguish:
  - no step happened
  - a step happened but failed
  - a step happened and actually reduced the fall

Why this comes before the next long run:
- current `v0.17.3a` metrics show touchdown activity, but the recovery summary
  metrics were not usable in the completed run
- the next branch should not be evaluated only from global `term_pitch_frac`

Required metrics:
- `recovery/no_touchdown_frac`
- `recovery/touchdown_then_fail_frac`
- `recovery/first_liftoff_latency`
- `recovery/first_touchdown_latency`
- `recovery/first_step_dx`
- `recovery/first_step_dy`
- `recovery/first_step_target_err_x`
- `recovery/first_step_target_err_y`
- `recovery/pitch_rate_at_push_end`
- `recovery/pitch_rate_at_touchdown`
- `recovery/pitch_rate_reduction_10t`
- `recovery/capture_error_at_push_end`
- `recovery/capture_error_at_touchdown`
- `recovery/capture_error_reduction_10t`
- `recovery/touchdown_to_term_steps`
- `eval_clean/unnecessary_step_rate`

Exit criteria:
- recovery summaries appear in training and eval logs
- step failures can be separated into:
  - no-touchdown failures
  - touchdown-but-still-failed cases
- unnecessary stepping in quiet standing is measurable

#### `v0.17.3a-pitchfix`: Cheap Sim-Side Pitch-Arrest Probe

Objective:
- try one cheap pure-RL sim-side fix before changing the step reward structure
  or adding teacher support

Status:
- config landed:
  `training/configs/ppo_standing_v0173a_pitchfix.yaml`

Changes relative to `v0.17.3a`:
- keep:
  - home-centered action mapping
  - relaxed termination
  - alive bonus
  - `min_height = 0.15`
- add only mild shaping:
  - `pitch_rate: -0.2`
  - `collapse_height: -0.05`
  - `collapse_vz: -0.1`

Decision rule:
- if fixed-ladder `eval_hard > 60%`, stop here and keep the simpler RL branch
- if it still misses, proceed to `v0.17.3a-arrest`

#### `v0.17.3a-arrest`: Recovery-Gated Step-Quality Rewards

Objective:
- improve the quality of catch steps without creating a generic stepping habit

Diagnosis this branch targets:
- the agent is already producing touchdown events
- the missing trait is not "move a foot at all"
- the missing trait is "take a recovery step that actually arrests pitch and
  regains capturability"

Principles:
- step rewards should be active only when recovery is actually needed
- do not reward generic rocking or quiet-stance touch adjustments
- do not add an explicit unnecessary-step penalty at first; measure it first

Reward changes:
- gate step-quality rewards by `push_active || recovery_active` rather than
  broad rollout-wide stepping pressure
- keep `step_event` only as a weak auxiliary term
- strengthen outcome-linked step quality:
  - turn on `step_progress`
  - optionally turn on `step_length` if the geometry metrics show steps are too
    short
- add post-touchdown arrest rewards:
  - reduction in `|pitch_rate|` after touchdown
  - reduction in capture-point error after touchdown
  - short-horizon survival after the first recovery touchdown

Config:
- `training/configs/ppo_standing_v0173a_arrest.yaml`

Metrics required before or during this branch:
- the `v0.17.3a-metrics` recovery-window diagnostics above

Exit criteria:
- `eval_hard > 60%`, or
- the run shows that step timing/geometry improves but still plateaus below the
  gate, which justifies teacher support

Completed result:
- best internal `eval_push` matched `v0.17.3a-pitchfix` but did not beat it
- fixed ladder on `checkpoint_300_39321600.pkl`:
  - `eval_medium = 83.1%`
  - `eval_hard = 52.2%`
- interpretation:
  - the policy is stepping
  - arrest rewards are active
  - the remaining gap is still recovery-step quality, not step initiation
- decision:
  - do not spend another branch on local reward tuning here
  - proceed to `v0.17.4t`

#### `v0.17.3b`: Sim2real Hardening (deferred until sim gate passes)

Objective:
- harden a standing policy that already works in simulation for real hardware
  transfer

Important:
- `v0.17.3b` is **not** expected to fix a fixed-ladder sim miss
- domain randomization, delay, and sensor noise are transfer-hardening tools,
  not the next explanation for why `eval_hard` is still below 60% in sim

**Change 4: Domain randomization**

New file: `training/envs/domain_randomize.py`
- `randomize_physics(mjx_model, rng) -> mjx_model`
- Randomize per episode at reset:
  - Floor friction: U(0.4, 1.0)
  - Link masses: nominal × U(0.9, 1.1) per body
  - Actuator Kp/Kd (gainprm + biasprm): nominal × U(0.9, 1.1)
  - Joint frictionloss: nominal × U(0.8, 1.2)
  - Joint armature: nominal × U(0.8, 1.2)
  - Servo torque / speed limits: nominal × U(0.9, 1.1)
  - Servo backlash deadband: U(0.02, 0.10) rad
  - Joint zero offsets (qpos0): +U(-0.03, 0.03) rad
- Follow Open Duck pattern: modify `mjx_model` fields in-place (vmapped)
- Gated by config flag `domain_randomization_enabled: bool`

ToddlerBot-informed note:
- the highest-value additions for hobby-servo realism are backlash,
  torque/speed-limit randomization, and armature/frictionloss variation
- these are transfer-hardening levers, not a substitute for fixing the sim-side
  recovery mechanism first

Implementation: store randomized parameters in env info (per-env scalars),
apply them at reset/step time. This follows the Open Duck Playground pattern
and works with the existing vmap-over-state architecture (no model threading
changes needed). Specifically:
- At `env.reset()`, sample randomized scalars (friction scale, mass scales,
  kp/kd scale, frictionloss scale, armature scale, torque-limit scale,
  backlash deadband) and store them in `WildRobotInfo`
- At `env.step()`, apply the stored scales to the relevant `mjx.Data` fields
  before `mjx.step()`
- `qpos0` offsets are applied at reset by adding noise to `self._init_qpos`

Files:
- `training/envs/domain_randomize.py` — **New file**: sampling functions
- `training/envs/env_info.py` — add randomization state fields to
  `WildRobotInfo`
- `training/envs/wildrobot_env.py` — call randomization at reset/step

**Change 5: Action and IMU delay + sensor noise**

Implementation note: the current env state stores only a single `prev_action`
(`env_info.py:76`), not an action history ring buffer. Action delay requires
either:
- expanding env state to store an action history array of length N, or
- using a simpler 1-step fixed delay (apply `prev_action` instead of current
  action) as the first version

Start with 1-step fixed delay, escalate to ring buffer only if needed.

- IMU noise B1: `imu_gyro_noise_std: 0.05`, `imu_quat_noise_deg: 1.0`,
  `imu_latency_steps: 1` (1-step fixed delay first)
- IMU noise B2 if transfer gap remains: low-frequency bias-walk / AR(1)-style
  gyro + orientation noise with per-episode amplitude variation
- Joint position noise: per-joint Gaussian before normalization
- Observation history: keep the first version simple, but add an optional short
  observation / action history window after the 1-step delay path is working;
  ToddlerBot suggests that temporal context is often more valuable than a more
  complex one-off reward change during transfer hardening

Config: `ppo_standing_v0173b.yaml` — standalone config (the config loader uses
plain `yaml.safe_load` with no inheritance/merge mechanism, so this config must
be self-contained, not "inheriting" from v0.17.3a).

#### `v0.17.3b` exit gate

v0.17.3b passes if:
- `eval_clean/success_rate > 90%` (standing survives randomization)
- `eval_hard/success_rate` measured and compared against v0.17.3a baseline
- Domain randomization is confirmed active (logged randomized parameter stats)
- a scripted sim2real-style comparison can be run on quiet-standing logs:
  commanded action vs measured joint position / velocity / torque and IMU

v0.17.3b fails if:
- Quiet standing drops below 80% under randomization
- Training diverges

#### Files Summary

v0.17.3a (must modify):

| File | Change |
|---|---|
| `policy_contract/spec.py` | Add `pos_target_home_v1` to `SUPPORTED_MAPPING_IDS` |
| `policy_contract/spec_builder.py` | Accept `mapping_id` param, pass to `ActionSpec` |
| `policy_contract/jax/calib.py` | Add `pos_target_home_v1` branch: home center, per-joint span, clip |
| `policy_contract/numpy/calib.py` | Mirror for runtime |
| `policy_contract/jax/obs.py` | Home-centered joint position normalization |
| `policy_contract/numpy/obs.py` | Mirror for runtime |
| `training/envs/wildrobot_env.py` | Pass `mapping_id` to `build_policy_spec()` + alive reward + relaxed termination |
| `training/train.py` | Pass `mapping_id` to `build_policy_spec()` |
| `training/core/training_loop.py` | Pass `mapping_id` to `build_policy_spec()` |
| `training/eval/visualize_policy.py` | Pass `mapping_id` to `build_policy_spec()` |
| `training/exports/export_policy_bundle.py` | Pass `mapping_id` to `build_policy_spec()` |
| `training/cal/cal.py` | Keep in sync with policy_contract span formula |
| `training/configs/training_runtime_config.py` | New config fields |
| `training/configs/training_config.py` | Wire `mapping_id` + new fields from YAML |
| `training/configs/ppo_standing_v0173a.yaml` | **New config** |
| `docs/joints_angle.md` | Update section 2 action↔ctrl equations |

v0.17.3b (after the sim hard-push gate passes):

| File | Change |
|---|---|
| `training/envs/domain_randomize.py` | **New file** — sampling functions for physics randomization |
| `training/envs/env_info.py` | Add randomization state fields to `WildRobotInfo` |
| `training/envs/wildrobot_env.py` | Domain rand at reset/step, 1-step action delay |
| `training/configs/training_runtime_config.py` | Domain rand + delay + noise config fields |
| `training/configs/ppo_standing_v0173b.yaml` | **New standalone config** (no inheritance) |
| `scripts/eval_sim2real_gap.py` | **New file** — compare action / joint / IMU traces between sim and real logs |

Exit criteria:
- the key recipe gaps are addressed on the pure-RL branch
- deployment remains policy-only
- sim competence is judged before transfer hardening work

### `v0.17.4`: Conditional Teacher

This milestone is entered only if:
- `v0.17.3a-pitchfix` still misses the hard gate, and
- `v0.17.3a-arrest` still misses or clearly plateaus below the hard gate

At that point, the remaining gap is treated as a bounded step-geometry /
recovery-quality gap, not a reason to change the deployed architecture.

#### `v0.17.4t`: Conditional Step-Target Teacher (only if gate missed)

Objective:
- add bounded geometry assistance because the recipe-fixed pure-RL branch
  still misses the hard gate

Changes:
- generate compact training-time teacher targets during `push_active ||
  recovery_active` windows only:
  - `step_required`
  - `swing_foot`
  - `target_step_x`
  - `target_step_y`
- first teacher form:
  - use teacher targets for reward shaping / supervision only
  - keep them out of actor observations
  - keep the deployed artifact as the same single policy
- possible teacher source:
  - capture-point or similar heuristic target generator
- add metrics and debug signals for teacher engagement and agreement

Implementation plan:
- new config: `training/configs/ppo_standing_v0174t.yaml`
- new teacher target generator module:
  - `training/envs/teacher_step_target.py`
- env integration:
  - `training/envs/wildrobot_env.py`
  - compute teacher targets only during disturbed windows
  - keep teacher targets out of actor observations
- runtime config plumbing:
  - `training/configs/training_runtime_config.py`
  - `training/configs/training_config.py`
- first teacher weights should stay small and bounded:
  - `teacher_step_required`
  - `teacher_target_step_xy`
  - optional `teacher_swing_foot`
- logging/debug:
  - teacher target values
  - teacher-active gate
  - policy/teacher agreement metrics at touchdown

First success criteria for `v0.17.4t`:
- `eval_hard > 60%`, or a clear improvement over `52.2%`
- `eval_clean` remains saturated
- teacher metrics confirm that support is used mainly in disturbed windows
- deployed policy remains single-policy with no runtime teacher dependency

Exit criteria:
- teacher signals are bounded and inspectable
- the policy can train with or without them
- the teacher branch is compared directly against the recipe-fixed pure-RL
  baseline
- the deployed policy does not require runtime teacher inputs

Completed result:
- run: `training/wandb/run-20260324_225425-zn8r3l5g`
- best standing checkpoint:
  - `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`
- best internal push eval at iter `330`:
  - `eval_push/success_rate = 93.75%`
  - `eval_push/episode_length = 478.9`
  - `eval_push/term_height_low_frac = 6.25%`
- fixed ladder on the best checkpoint:
  - `eval_medium = 93.8%`
  - `eval_hard = 70.6%`
  - `eval_hard/term_height_low_frac = 29.4%`

Interpretation:
- `v0.17.4t` cleared the real standing hard gate
- the teacher remained bounded to disturbed windows
- runtime stayed a single-policy deployment path
- but viewer inspection suggests the policy still does not show a strong
  conditional recovery-step trait under hard pushes
- if visible stepping when bracing fails is a requirement, the branch is
  outcome-successful but mechanism-incomplete

Decision:
- freeze iter `330` as the winning standing checkpoint
- keep `v0.17.4t` as the current best standing baseline
- start a follow-on branch focused on conditional step initiation:
  `v0.17.4t-step`
- delay `v0.17.3b` transfer hardening until the policy shows the desired
  step-to-rebalance trait under hard pushes
- keep stronger teacher forms optional unless `v0.17.4t-step` also plateaus

### `v0.17.5`: Standing Push Recovery

Objective:
- solve standing push recovery on the RL mainline with the intended mechanism:
  brace when bracing is enough, step when bracing no longer works

Changes:
- keep the `v0.17.4t` fixed-ladder gain
- add explicit conditional step-initiation support on top of
  `teacher_target_step_xy`
- likely turn on:
  - `teacher_step_required`
  - `teacher_swing_foot`
- if step-initiation alone still plateaus, test a whole-body recovery variant:
  - conditional knee flexion / recovery crouch
  - modest center-of-mass lowering during disturbed windows
  - keep this bounded to `push_active || recovery_active`
  - do not turn it into a global lower standing posture
- add explicit step-trait diagnostics:
  - `visible_step_rate_hard`
  - `first_liftoff_latency`
  - `first_touchdown_latency`
  - `no_touchdown_frac`
  - `touchdown_then_fail_frac`
  - `eval_clean/unnecessary_step_rate`

Exit criteria:
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` materially below `v0.17.1`
- hard pushes usually trigger a real recovery step when bracing alone is
  insufficient
- if hard pushes are too large for an upright brace-only response, the policy
  should be allowed to use whole-body recovery:
  modest knee bend / CoM lowering plus a recovery step
- quiet standing does not develop a stepping habit

Current status:
- fixed-ladder competence gate is already satisfied by `v0.17.4t`
- mechanism requirement remains open because the current branch does not yet
  clearly show conditional visible stepping under hard pushes
- next work is `v0.17.4t-step`, not `v0.17.3b`
- current hypothesis under review:
  `v0.17.4t-step` may need a follow-on whole-body recovery branch
  (`v0.17.4t-step-crouch`) if the robot still falls with a tall straight body
  under large pushes
- expected behavior for that branch:
  brace for small disturbances, but under larger pushes use a bounded recovery
  crouch together with the capture step
- before implementation, get external review on whether this matches standard
  humanoid push-recovery practice

Data support for the current hypothesis:
- `v0.17.4t` already cleared the fixed ladder with:
  - best checkpoint:
    - `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`
  - `eval_medium = 93.8%`
  - `eval_hard = 70.6%`
  - `eval_hard/term_height_low_frac = 29.4%`
- however, visual evaluation still suggests many large-push failures happen with
  a relatively tall straight body rather than a clear whole-body recovery
- `v0.17.4t-step` evidence (run `20260328_155028-vtrx3sxe`):
  - best internal eval_push point is iter `440`
  - `eval_push/success_rate = 93.75%` (same plateau as best `v0.17.4t`)
  - `eval_push/visible_step_rate_hard = 0.537`
  - `eval_push/recovery_no_touchdown_frac = 0.104`
  - `eval_push/recovery_touchdown_then_fail_frac = 0.060`
  - `eval_clean/unnecessary_step_rate = 0.027` at the best point
  - later in the same run, `eval_clean/unnecessary_step_rate` degrades badly
    (`0.50+` by iter `530`), indicating over-stepping if the branch is trained
    too long unchanged
- fixed ladder on the best checkpoint (critical suites run):
  - checkpoint:
    - `training/checkpoints/ppo_standing_v0174t_step_v0174t-step_20260328_155031-vtrx3sxe/checkpoint_440_57671680.pkl`
  - `eval_medium = 83.1%`
  - `eval_hard = 52.2%`
  - `eval_hard_long = 27.6%`
  - `eval_hard/term_height_low_frac = 47.8%`
  - `eval_easy` was not run in this pass; the script's printed `0.0%` target
    line is not a measured branch result
- interpretation:
  - step teaching did produce some visible-step behavior internally
  - but that mechanism did not translate into a stronger fixed-ladder branch
  - `v0.17.4t-step` regressed materially from `v0.17.4t` on hard pushes
  - the branch landed back near the earlier `v0.17.3a` regime on fixed-ladder
    robustness
  - the current search problem is no longer just "should I step?" but "how do I
    coordinate trunk, knees, and step timing when bracing is insufficient?"

Comparison across key branches:

| Branch | eval_medium | eval_hard | eval_hard h_low |
| --- | ---: | ---: | ---: |
| `v0.17.1` | 71.6% | 39.2% | 60.8% |
| `v0.17.3a` | 83.1% | 51.4% | 48.6% |
| `v0.17.3a-pitchfix` | — | ~51.8% | — |
| `v0.17.4t` | 93.8% | 70.6% | 29.4% |
| `v0.17.4t-step` | 83.1% | 52.2% | 47.8% |
| `v0.17.4t-crouch` | 83.3% | 54.2% | 45.8% |
| `v0.17.4t-crouch-teacher` | 84.4% | 55.1% | 44.9% |

Reference benchmark:
- current best validated RL checkpoint remains:
  - `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`
- practical interpretation:
  - `v0.17.4t-step` is now fixed-ladder checked on the critical suites, and it
    is worse than `v0.17.4t`
  - `v0.17.4t-crouch` is not only worse than `v0.17.4t`
  - `v0.17.4t-crouch-teacher` B1 recovered slightly over `v0.17.4t-step` and
    `v0.17.4t-crouch`, but it still failed the hard gate and remained far below
    `v0.17.4t`
  - it falls back into the same general hard-push regime as the earlier
    `v0.17.3a` family rather than extending the `v0.17.4t` improvement

Decision after external review:

The RL-only teacher path has now been explored through the full `v0.17`
standing sequence (`v0.17.3a` → `pitchfix` → `arrest` → `v0.17.4t` →
`v0.17.4t-step` → `v0.17.4t-crouch` → `v0.17.4t-crouch-teacher` B1). The
fixed-ladder gate was cleared only at `v0.17.4t` (70.6%), but the robot still
falls with a stiff straight body — it does not show coordinated whole-body
recovery (knee flexion + stepping + trunk motion).

The core problem:
- The 19-joint coordination required for whole-body recovery is a large search
  space for end-to-end RL
- Every run so far penalized `height_target` and `posture` during recovery,
  actively preventing the crouch response the robot needs
- Even with teacher step targets, the RL policy discovered bracing/survival
  strategies rather than coordinated recovery motions

Plan:
1. One fast validation run to test whether reward relaxation alone is
   sufficient (cheapest possible test)
2. If it is not, try one compact whole-body teacher branch inside RL before a
   full architecture pivot
3. If that still fails, pivot to a position-level model-based control stack
   that reduces the coordination search space substantially

#### `v0.17.4t-crouch`: Fast Reward Relaxation Validation

Objective:
- test whether removing reward barriers is sufficient for whole-body recovery,
  before committing to a new control architecture

This is a single-run validation experiment, not an iterative tuning branch.

Changes relative to `v0.17.4t`:
- keep all existing teacher rewards
- during `push_active || recovery_active`:
  - `height_target`: 0.0 (was 0.5 — stop penalizing crouch)
  - `posture`: 0.0 (was 0.5 — stop penalizing knee flexion)
  - `orientation`: -0.3 (was -1.0 — allow trunk motion)
- add during recovery only:
  - `height_floor`: -2.0 below 0.20m (prevent full collapse)
  - `com_velocity_damping`: 1.0 (reward reducing horizontal CoM velocity)
- keep quiet-standing rewards unchanged outside disturbed windows

Resume from:
- `v0.17.4t-step` best checkpoint (iter 440) — this checkpoint has the
  strongest step-initiation trait. The crouch test builds on the step mechanism
  rather than discarding it.
- If the intent is to ablate crouch permission without the step teacher,
  that would be a separate experiment resuming from `v0.17.4t`.

Gate (all required before drawing conclusions):
- run fixed-ladder evaluation on the best checkpoint
- compare `eval_hard` against `v0.17.4t` (70.6%) — must not regress below 65%
- check `eval_clean/unnecessary_step_rate` — must not exceed 0.10
- check `recovery/visible_step_rate` — should increase over `v0.17.4t-step`
- check `recovery/no_touchdown_frac` and `recovery/touchdown_then_fail_frac`
- check `recovery/pitch_rate_reduction_10t` — is momentum arrest improving?
- visual inspection of hard-push rollouts in the MuJoCo viewer

Decision after gate:
- **if `eval_hard >= 70%` AND visual crouch+step under hard pushes**:
  the reward was the bottleneck. Continue with RL path, proceed to `v0.17.3b`
  sim2real hardening.
- **if `eval_hard >= 65%` but no visible coordinated recovery**:
  the RL path can survive pushes through subtle mechanisms but reward
  relaxation alone did not create coordinated whole-body recovery. Proceed to
  one compact whole-body teacher branch inside RL before a model-based pivot.
- **if `eval_hard < 65%`**:
  the reward relaxation actively hurt survival. Restore v0.17.4t as the best
  RL checkpoint and proceed to one compact whole-body teacher branch inside RL
  before `v0.18`.

Expected outcome:
- low confidence this will produce visible whole-body recovery
- high confidence this will answer whether reward gating is sufficient
- one run, ~10 hours

Observed result:
- screening run: `run-20260329_134942-k4xaef65`
- best checkpoint:
  - `training/checkpoints/ppo_standing_v0174t_crouch_v0174t-crouch_20260329_134944-k4xaef65/checkpoint_550_72089600.pkl`
- comparison reference:
  - best validated `v0.17.4t` checkpoint remains:
    - `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`
- fixed ladder:
  - `eval_clean = 100.0%`
  - `eval_easy = 100.0%`
  - `eval_medium = 83.3%`
  - `eval_hard = 54.2%`
  - `eval_hard_long = 32.1%`
- interpretation:
  - this branch failed the fixed-ladder screen (`eval_hard < 65%`)
  - it is materially worse than the validated `v0.17.4t` baseline (`70.6%`)
  - it also failed to beat the `v0.17.4t-step` mechanism branch on the intended
    whole-body evidence:
    - no convincing crouch-specific lowering
    - no convincing knee-flex increase
    - no clear increase in useful step trait
- conclusion:
  - reward relaxation alone was not sufficient
  - Branch A is closed
  - proceed to `v0.17.4t-crouch-teacher`

#### `v0.17.4t-crouch-teacher`: Compact Whole-Body Teacher Inside RL

Objective:
- add one compact whole-body recovery scaffold inside RL if reward relaxation
  alone is still too weak, while keeping the deployed artifact as one policy

This is the last planned RL-side mechanism experiment before a model-based
pivot.

Why this branch exists:
- reward gating alone may still leave too large a search space for coordinated
  trunk + knee + step behavior
- a compact task-space teacher can reduce that search space without prescribing
  every joint and without creating a runtime teacher dependency

Scope:
- keep the current step teacher as a bounded recovery-intent scaffold
  (`teacher_step_required`, `teacher_swing_foot`, `teacher_target_step_xy`)
- add one or two compact whole-body recovery targets during
  `push_active || recovery_active`, for example:
  - horizontal CoM-velocity reduction target
  - bounded recovery height band
  - bounded knee-flex / crouch indicator
- do **not** add per-joint supervision
- keep quiet standing unchanged outside disturbed windows

Rationale:
- current `v0.17.4t-step` data suggests step intent is no longer the only
  bottleneck
- current `v0.17.4t-crouch` hypothesis tests whether simple permission is
  enough
- if not, the next smallest step is compact task-space guidance, not a full new
  controller stack

Exit criteria:
- fixed ladder remains competitive with `v0.17.4t` (`eval_hard >= 65%` for a
  screening bar, target `>= 70%`)
- visible crouch + step behavior appears under hard pushes
- `eval_clean/unnecessary_step_rate <= 0.10`
- recovery metrics improve:
  - `visible_step_rate`
  - `no_touchdown_frac`
  - `touchdown_then_fail_frac`
  - `pitch_rate_reduction_10t`

If this branch also fails:
- the RL path has exhausted the planned bounded whole-body guidance options
- proceed to `v0.18`

Observed result (B1):
- screening run: `run-20260329_203029-rocg9xk1`
- best checkpoint:
  - `training/checkpoints/ppo_standing_v0174t_crouch_teacher_v0174t-crouch-teacher_20260329_203032-rocg9xk1/checkpoint_600_78643200.pkl`
- fixed ladder:
  - `eval_clean = 100.0%`
  - `eval_easy = 100.0%`
  - `eval_medium = 84.4%`
  - `eval_hard = 55.1%`
  - `eval_hard_long = 31.3%`
- interpretation:
  - this branch improved slightly over `v0.17.4t-step` (`52.2%`) and
    `v0.17.4t-crouch` (`54.2%`) on `eval_hard`
  - but it still failed the fixed-ladder screening bar (`eval_hard < 65%`)
  - training-time diagnostics suggested the whole-body teacher stayed too weak
    to create a convincing crouch-mediated recovery mechanism
- conclusion:
  - Branch B B1 is not a validated replacement for `v0.17.4t`
  - if the RL path is continued at all, only one bounded follow-up remains:
    add the second whole-body teacher term (`teacher_com_velocity_reduction`)
  - otherwise proceed to `v0.18`

#### External review update: ToddlerBot lessons

External review of ToddlerBot does **not** change the standing-branch verdict.
It strengthens it.

What the review supports:
- ToddlerBot succeeds with a more structured stack around learning:
  calibrated digital twin, actuator / joint SysID, explicit sim2real gap
  evaluation, observation history + delay modeling, and reference / planner
  scaffolding
- this supports moving `v0.18` forward as the main new branch once the bounded
  RL whole-body guidance options are exhausted
- it also upgrades `v0.17.3b`: transfer hardening should include backlash,
  armature, torque-limit randomization, and stronger IMU / delay realism

What the review does **not** support:
- do **not** copy a full ZMP walking reference stack directly into the current
  standing recovery branch
- do **not** assume that a scripted joint-space
  `crouch -> step -> stand` trajectory is the right next RL input
- do **not** treat lower `gamma` or longer training alone as the missing
  mechanism

Takeaway for WildRobot:
- for the current standing problem, the highest-confidence structured input is
  still compact task-space recovery guidance
  (`teacher_step_required`, `teacher_swing_foot`, `teacher_target_step_xy`,
  `teacher_com_velocity_reduction`)
- for the next architecture, the ToddlerBot lesson is to add structure at the
  right layer, not to keep stretching end-to-end PPO
- for later walking work, a simple capture-point / ZMP-style nominal step
  generator remains a plausible extension on top of `v0.18`

### `v0.18`: Position-Level Model-Based Control

Objective:
- replace end-to-end RL joint control with a position-level model-based stack
  that reduces the whole-body coordination problem from RL exploration to
  structured task-space IK design

This is a **new branch**, not a continuation of the RL teacher path. It is the
fallback after the bounded RL whole-body guidance options (`v0.17.4t-crouch`
and `v0.17.4t-crouch-teacher`) are exhausted. The standing-first validation
sequence is preserved.

Why a new branch:
- six iterations of RL reward and teacher tuning produced survival but not
  coordinated recovery
- the remaining gap is a 19-joint coordination problem that RL has not solved
- IK-based control reduces this search space substantially: the coordination
  problem moves from RL exploration in 19-joint space to structured task-space
  IK design, which is a more predictable engineering problem
- the same stack extends to walking (different gait targets, same planner + IK)

#### Architecture

```text
Layer 1: Task-space planner (10-50 Hz)
  inputs:  CoM state, foot positions, contact state, push detection
  outputs: target CoM position, target foot placement, recovery mode flag
  method:  capture-point / ALIP on LIP model

Layer 2: Whole-body IK (50 Hz)
  inputs:  task-space targets from Layer 1
  outputs: 19 joint position targets
  method:  Jacobian-based IK using MuJoCo kinematics (mj_jac, mj_comPos)
  constraints: joint limits, kinematic reachability

Layer 3: Servo interface (50 Hz)
  inputs:  joint position targets from Layer 2
  outputs: position commands to servos (or MuJoCo ctrl)
  method:  direct pass-through (servos have internal PD)
```

Key design points:
- **Position-level, not torque-level** — outputs joint angle targets, which is
  what the HTD-45H servos accept. No WBC, no inverse dynamics.
- **MuJoCo-native kinematics** — uses `mj_jac` for Jacobian, `mj_comPos` for
  CoM. No URDF, no Pinocchio, no OCS2.
- **50 Hz is sufficient** — IK and capture-point computation are microseconds
  of matrix math. This is not a high-bandwidth torque control loop.
- **Same MJCF model** — `assets/v2/wildrobot.xml` remains the single source of
  truth. The IK reads joint limits and kinematic chain from the same model.
- **ToddlerBot-informed transfer discipline** — keep a calibrated digital twin,
  explicit actuator realism, and a scripted sim-vs-real evaluation loop as part
  of this branch from the start rather than bolting them on at the end.
- **State feedback, not scripted pose phases** — recovery targets should be
  generated from current CoM / contact state. Do not hard-code a fixed
  `crouch -> step -> stand` sequence as the primary standing controller logic.

#### Why the original OCS2 rejection does not apply

The `v0.17.2` architecture review rejected model-based control for three
reasons. All three are addressed by the position-level approach:

| Original rejection reason | Position-level response |
|---|---|
| "Position servos can't do torque-based WBC" | This stack outputs position targets, not torques. IK computes joint angles; the servo PD handles force generation. |
| "No accurate dynamics model for real-time control" | IK needs kinematics, not dynamics. MuJoCo's kinematic model is exact. For the planner, a simple LIP model is sufficient. |
| "50 Hz is too slow for MPC inner loop" | IK + capture-point is microseconds of computation. 50 Hz is more than enough for position-level planning on a 4.1 kg robot. |

The earlier rejection was about OCS2/Pinocchio/URDF/torque-level WBC — a heavy
stack that does not match this hardware. The proposed stack is much lighter:
just MuJoCo kinematics + simple planner + position output.

#### How whole-body recovery works in this stack

How IK reduces the coordination problem:
- **CoM height target → knee flexion**: planner commands "lower CoM to 0.30m
  during recovery" → IK computes knee angles to achieve this → knees flex as
  a direct consequence of the task-space target. This is the highest-confidence
  part of the approach.
- **Foot placement target → swing leg coordination**: planner commands step
  target → IK computes swing leg joint angles. This is standard leg IK.
- **Trunk counter-rotation**: this does NOT fall out of basic IK automatically.
  The waist joint is not in the leg chains, so it requires an explicit
  objective — either an angular momentum regulation task, a CoM horizontal
  positioning task that involves the trunk, or a nullspace objective. This must
  be designed, not assumed.
- **Contact consistency**: while one foot swings, the stance foot must stay
  planted and the CoM must remain over the support polygon. This requires a
  prioritized IK formulation (stance constraint > CoM target > swing target).

The search space is substantially reduced compared to end-to-end RL, but
the IK formulation itself requires careful engineering — particularly for
trunk motion and task priority management.

For quiet standing:
- Planner sets target CoM at standing height (0.44m), feet at current position
- IK produces home-like joint angles
- No push → no recovery mode → robot stands still

#### Implementation plan

Phase 0: Servo / sim realism calibration
- characterize `HTD-45H` actuation with simple scripted tests
- fit or bound backlash, torque / speed limits, armature, and delay
- add a reusable sim2real evaluation script for action / joint / IMU traces
- deliverable: calibrated ranges that feed both `v0.17.3b` and `v0.18`

Phase 1: IK validation in simulation
- implement Jacobian-based IK for the leg chains using MuJoCo's `mj_jac`
- validate that IK can track a commanded CoM height + foot position on the
  static model (no physics step)
- validate joint limits and singularity handling
- deliverable: a standalone script that poses the robot at commanded task-space
  targets

Phase 2: Standing controller
- implement capture-point computation from CoM state
- implement standing balance: hold CoM over support polygon, adjust ankle/hip
  targets
- implement recovery mode: when capture point exits support polygon, lower CoM
  target + plan recovery step
- integrate with MuJoCo simulation loop
- deliverable: quiet standing and push recovery in simulation

Phase 3: Fixed-ladder validation
- run the same standing push ladder used for RL evaluation
- compare directly against `v0.17.4t` (70.6% eval_hard)
- visual inspection: does the robot show coordinated crouch + step?

Phase 4: Walking extension
- add gait timing (alternating support phases)
- use the same planner with periodic foot placement targets; if needed, add a
  simple nominal gait / step reference generator rather than asking RL to
  invent walking from scratch
- keep standing recovery as a regression gate

Phase 5: Sim2real
- calibrate IK against real robot kinematics
- run the explicit sim2real evaluation suite on joint / IMU traces before and
  after hardware standing tests
- deploy planner + IK on Raspberry Pi (no ML inference needed)
- hardware validation

#### Files

New code (under `control/`):
- `control/ik/jacobian_ik.py` — Jacobian-based IK using MuJoCo
- `control/planner/capture_point.py` — capture-point computation
- `control/planner/standing_planner.py` — standing balance + recovery planner
- `control/standing_controller.py` — top-level standing controller
- `control/sim_adapter.py` — adapter to run the controller in MuJoCo sim loop

Validation:
- `scripts/characterize_servo_response.py` — scripted actuator tests for delay,
  backlash, and speed / torque bounds
- `scripts/validate_ik.py` — standalone IK validation
- `scripts/validate_standing_controller.py` — standing + push test
- `scripts/eval_sim2real_gap.py` — compare action / joint / IMU traces

Reused from RL branch:
- `assets/v2/wildrobot.xml` — same MJCF model
- `training/eval/eval_ladder_v0170.py` — same fixed-ladder evaluation
- `runtime/` — same servo interface for deployment

#### Exit criteria

Phase 2 gate (standing controller):
- quiet standing is stable (no drift, no oscillation)
- push recovery shows visible knee flexion + stepping
- the controller survives the fixed-ladder eval_medium pushes

Phase 3 gate (fixed-ladder):
- `eval_medium > 75%`
- `eval_hard > 60%`
- visual inspection confirms whole-body recovery (crouch + step)

Phase 4 gate (walking):
- slow forward walking (0.05-0.15 m/s)
- stop and hold
- standing recovery regression passes

#### Relationship to the RL branch

The RL branch (`v0.17.x`) remains in the repo as:
- baseline comparison for the model-based stack
- fallback if the model-based stack fails on unexpected scenarios
- potential future hybrid: RL policy on top of the model-based lower stack for
  adaptive behavior

The `v0.17.4t` checkpoint (`eval_hard = 70.6%`) is the RL-path standing
baseline. The model-based stack should match or exceed this on the same ladder
before it replaces the RL path as the mainline.

#### Confidence assessment

The model-based stack is **medium confidence** overall. It is a more
structured approach to the coordination problem than end-to-end RL, but it is
not a guaranteed solution.

High confidence:
- CoM height target → knee flexion via Jacobian IK. This is standard
  kinematics and will work if the robot has sufficient workspace.
- Foot placement → swing leg IK. Analytical IK for 4-DOF legs is solved.
- 50 Hz position-level control is feasible for this computation.

Medium confidence:
- Trunk counter-rotation requires explicit IK design (angular momentum task
  or CoM horizontal positioning through the trunk). It will not emerge
  automatically from basic leg IK.
- Contact-consistent IK (stance foot stays planted, CoM stays over support)
  requires correct task prioritization. Getting this wrong produces unstable
  or infeasible solutions.
- The capture-point planner uses a simplified LIP model that may not match
  the actual robot dynamics well.

Open risks:
- **IK feasibility**: kinematic singularities or limited workspace may
  prevent the IK from tracking task-space targets. Phase 1 validates this.
- **Dynamic feasibility**: IK computes kinematically correct poses, but the
  4.41 Nm servos may not track them fast enough during recovery. This is the
  same torque budget constraint that limited the RL approach.
- **Planner model mismatch**: LIP/capture-point assumes point mass at constant
  height, which is a poor fit for a 4.1 kg robot with thick legs at
  variable height. Phase 2 validates this in simulation.
- **Sim2real gap**: the IK solution in simulation may not transfer to hardware
  if kinematic calibration is off. Phase 5 addresses this.
- **Walking complexity**: gait timing and contact scheduling are non-trivial
  even with IK. Phase 4 may require iteration.

The main advantage over the RL path is not that the model-based stack is
guaranteed to work, but that failures are diagnosable and fixable through
engineering rather than opaque training dynamics.

### `v0.19`: Walking On The Same Control Stack

Objective:
- extend the standing controller to slow / conservative walking
- same planner + IK architecture, different gait targets

Changes:
- add gait timing (periodic left/right support switching)
- add forward velocity command → footstep placement targets
- stop / hold command
- gentle turn command
- keep standing push recovery as a regression suite

Exit criteria:
- one control stack supports both standing recovery and conservative walking
- the robot walks at 0.05-0.15 m/s on flat ground
- standing push recovery regression passes after walking training

### Optional Future: Hybrid RL + Model-Based

If the model-based stack handles standing and walking but struggles with
robustness or adaptive behavior:
- use the model-based stack as the lower layer (IK + planner)
- add an RL policy on top that outputs task-space corrections
- the RL policy searches in 5-6 dimensional task space instead of 19-joint
  space — exponentially easier
- this is the Singh architecture (model-based geometry + RL execution) with
  the model-based layer providing the coordination that end-to-end RL could
  not discover

---

## Decision Rules

### After `v0.17.4t-crouch`

- if the robot shows coordinated crouch + step under hard pushes:
  - stay on the RL path
  - proceed to `v0.17.3b` sim2real hardening, then walking
- if the robot still falls stiff despite reward relaxation:
  - reward gating alone was insufficient
  - proceed first to `v0.17.4t-crouch-teacher`
- actual result:
  - fixed ladder regressed to `eval_hard = 54.2%`
  - branch did not show the desired whole-body mechanism
  - this branch is therefore closed and `v0.17.4t-crouch-teacher` is the active
    next step

### After `v0.17.4t-crouch-teacher`

- if compact whole-body teacher guidance produces coordinated crouch + step:
  - stay on the RL path
  - proceed to `v0.17.3b` sim2real hardening, then walking
- if it still fails to produce the required mechanism:
  - B1 actual result:
    - fixed ladder reached `eval_hard = 55.1%`
    - this is slightly better than `v0.17.4t-step` and `v0.17.4t-crouch`, but
      still below the `65%` screening bar
  - if the RL path is continued, only the bounded B2 follow-up remains
  - otherwise proceed to `v0.18` model-based control stack

### On the `v0.18` model-based path

- the deployed artifact changes from an RL policy to an IK + planner stack
- this is a deliberate architecture change, not a failure of the RL approach —
  the RL branch established the benchmark gate and proved what reward/teacher
  tuning can and cannot achieve
- the RL branch remains as a baseline and potential future hybrid layer

### Standing-first validation still applies

- regardless of the control approach (RL or model-based), standing push
  recovery is validated first on the fixed ladder
- walking extends the same stack afterward
- the same eval ladder, same push definitions, and same success metrics apply

### What we learned from the RL branch

Applied results from `v0.17.3a` through `v0.17.4t-step`:
- recipe-fixed pure RL plateaued at `eval_hard ≈ 52%`
- bounded step-target teacher broke through to `eval_hard = 70.6%`
- the teacher improved survival rate but did not produce visible whole-body
  recovery
- the remaining gap is coordination (crouch + step + trunk), not reward
  tuning or step intent

### What is no longer being avoided

The original `v0.17.2` architecture decision avoided model-based control
because it assumed OCS2/Pinocchio/URDF/torque-level WBC. The `v0.18` proposal
is different:
- position-level output (not torque)
- MuJoCo-native kinematics (not Pinocchio/URDF)
- simple planner (capture-point, not full MPC)
- 50 Hz position commands (matches servo interface)

This is a lightweight model-based stack, not the heavyweight OCS2 adoption
that was correctly deferred.

---

## Canonical Plan Status

This document is the active source of truth for training planning on the current
branch:
- standing push recovery is the first acceptance task
- walking is added only after the standing stack is credible
- archived walking-first plans are historical references only

If v0.17 succeeds, it gives the project:
- a deployable standing-recovery policy for current hardware
- a realistic path to conservative walking
- a much clearer answer about whether stronger teachers or heavier controllers
  are actually necessary
