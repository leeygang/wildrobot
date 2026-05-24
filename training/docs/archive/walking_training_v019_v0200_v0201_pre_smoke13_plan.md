# WildRobot Walking Training Plan — Archive (pre-v0.20.1-smoke13)

This archive holds the historical sections of
`training/docs/walking_training.md` that the active doc no longer
carries inline.  Moved 2026-05-24 as part of the smoke13+ refresh:
the verbose `Current status` history that ended at v0.20.0 (pre-PPO
pivot), the per-milestone plan subsections for v0.19.0 through
v0.20.0-D, and the per-smoke milestone plans for v0.20.1-smoke6-prep3
and v0.20.1-smoke7.

The active doc keeps the high-level v0.20.1 framework (smoke contract,
fail-mode tree, naming hierarchy), the appendices, and a refreshed
short current-status summary.  Per-run results are in
`training/CHANGELOG.md` (smoke13+) and `training/CHANGELOG.archive.md`
(smoke12b and earlier).

---

## Region A — Verbose `Current status` history (replaced 2026-05-24)

The block below was originally lines ~426-475 of
`walking_training.md` under the `## Milestones` heading.  It was
written when v0.20.0 was the active stage and grew to a wall-of-text
recap of every v0.19.x failure mode.  The current doc replaces it
with a 5-line summary that points to the changelog for run-level
detail.

### Current status

- `v0.19.0` complete: locomotion-first contract, schema, and runtime/logging
  split are in place.
- `v0.19.1` complete: digital-twin realism profile, SysID capture, and
  sim-vs-real comparison tooling are implemented.
- `v0.19.2` complete: `walking_ref_v1`, bounded sagittal IK, and nominal
  reference smoke tests are implemented.
- `v0.19.3` implemented but first screening run failed:
  - run: `training/wandb/offline-run-20260402_082711-xn1wgkap`
  - failure mode: forward-dive / skate exploit
  - `eval_clean/success_rate = 0%`
  - `term_pitch_frac = 100%`
  - episode length collapsed to about `40` steps
  - achieved forward speed overshot command by about `2x`
  - `m3_swing_foot_tracking` and `m3_foothold_consistency` stayed nearly
    inactive
- `v0.19.3a` failed to clear the same gate:
  - run: `training/wandb/offline-run-20260402_104958-qhb3w0zc`
  - `term_pitch_frac = 100%`
  - `eval_clean/success_rate = 0%`
  - reduced command range and stronger tracking penalties did not create a
    viable stepping prior
- `v0.19.3b` established the root blocker:
  - nominal-only probe with zero residual still pitch-fails early
  - `q_ref` is not dynamically viable under the full v0.19.5 env
- `v0.19.3c` and `v0.19.3d` improved semantics and braking diagnostics, but
  did not materially improve the nominal-only probe gate
- `v0.19.3e` added support-first local clamping for swing-x and pelvis pitch:
  - clamp is active
  - survival still does not improve
  - PPO must remain paused
- `v0.19.4` complete: nominal reference validation and
  walking reference achieved.  Ctrl ordering bug
  found and fixed.
  DCM COM trajectory implemented.  Nominal walking: 461 steps, 15 gait
  cycles, forward speed +0.038 m/s at cmd=0.15.
- `v0.19.5` attempted but handoff was premature (April 13-14):
  - PPO found lean-back exploit (v0.19.5b) and move-less exploit (v0.19.5c)
  - Root cause: nominal only delivered 25% of commanded speed, 30-50% of
    commanded step amplitude — propulsion was mis-assigned to PPO
  - Handoff gate tightened: now requires ≥50% command tracking, multi-seed
    robustness, step-amplitude diagnosis
  - Result: do not continue `v0.19.5x` reward surgery as the mainline path
- Current active stage is now **`v0.20.0`: Prior reference pivot**.
  - a solid prior reference must be chosen and validated before PPO resumes
  - `ZMP` is the recommended first executable prior
  - `ALIP` is the intended upgrade path if `ZMP` proves too conservative
  - `Placo` is a prototyping aid, not the mainline prior source of truth


---

## Region B — Pre-v0.20.1 milestone subsections (replaced 2026-05-24)

The block below was lines ~476-802 of `walking_training.md` and held the milestone-level plan for v0.19.0, v0.19.1, v0.19.2, v0.20.0 Prior Reference Pivot, and v0.20.0-A through v0.20.0-D.  v0.20.1 and later milestones remain inline in the active doc.

### `v0.19.0` Platform Pivot

Software stack:

- define the locomotion-first runtime contract
- define the shared observation, action, and log schema
- add this walking plan as the active locomotion roadmap

Training approach:

- stop using standing-only PPO as the main product path
- define the locomotion task around forward speed and yaw commands

Sim2real:

- create one reproducible calibration checklist
- create one standing and squat replay capture on hardware

Exit gate:

- one documented train/eval/deploy contract exists for locomotion

### `v0.19.1` Digital Twin and SysID

Software stack:

- add actuator realism parameters under version control
- add sim-vs-real replay comparison tools

Training approach:

- no full walking target yet beyond smoke tests
- validate that the simulator reproduces measured actuator behavior

Sim2real:

- collect representative joint step responses
- fit delay, backlash, frictionloss, armature, and effective output limits
- model IMU noise and foot-switch imperfections

Exit gate:

- scripted sim-vs-real traces are close enough to justify policy transfer work

### `v0.19.2` Reference Generator v1

Software stack:

- implement `walking_ref_v1`
- implement `LIPM / DCM / capture-point` stepping logic
- implement simple swing-foot and pelvis target generation
- implement the first leg IK path

Training approach:

- use the reference only for smoke-test rollouts at first
- verify feasibility before training a full policy

Sim2real:

- replay the nominal reference slowly on hardware with safety limits
- verify foot clearance, touchdown geometry, and joint limit margin

Exit gate:

- the reference generator produces coherent stepping in simulation
- the nominal motion is safe enough to serve as a policy prior
- initial smoke command:
  - `uv run python tools/reference_smoke/run_reference_smoke.py --robot-config assets/v2/mujoco_robot_config.json --steps 300 --dt-s 0.02 --forward-speed-mps 0.12`

### `v0.20.0` Prior Reference Pivot

Purpose:

- stop treating reward tuning as the main walking lever
- choose a reference prior family that can own gait semantics
- define hard visual and metric validation before PPO resumes
- fully remove the runtime FSM from the active locomotion path

Status: **active**

Decision:

- start from a `ZMP`-shaped offline command-indexed prior library
- train the first policy in a ToddlerBot-style bounded residual setup
- keep the architecture Cassie-compatible later by preserving a reusable
  reference preview interface
- use `Placo` only for prototyping and visual authoring, not as the mainline
  validation truth

Why:

- `ZMP` is cheapest to build, easiest to debug, and easiest to validate on a
  small position-controlled biped
- `ALIP` has better long-term dynamic coverage, but should upgrade a proven
  prior rather than replace an unvalidated stack
- `Placo` is valuable for authoring and inspection, but on WildRobot it is too
  easy to get kinematically plausible motions that are not dynamically viable
- bounded residual PPO is the safest first policy contract
- Cassie-style direct control remains an upgrade path, not the first step

Execution milestones:

1. `v0.20.0-A`: Prior contract freeze
2. `v0.20.0-B`: `ZMP` prior library generation and viewer
3. `v0.20.0-C`: Reference + IK + short-horizon feasibility validation
4. `v0.20.0-D`: PPO unblocks only after the prior gate clears

These milestone labels are canonical and should match
[reference_design.md](reference_design.md).

SysID note:

- servo-model verification is required before sim-to-real transfer
  (`v0.20.2`), not before ZMP prior generation
- the ZMP library can be generated and validated in simulation using the
  existing MuJoCo actuator model
- hardware SysID data collection should happen in parallel and must gate
  `v0.20.2` deployment, not `v0.20.0` prior work

Exit rule:

- no `v0.20.1` PPO training until the `v0.20.0` prior gate passes

Primary design note:

- [`reference_design.md`](reference_design.md)

### `v0.20.0-A` Prior contract freeze

Expectation:

- this milestone defines a stable reference contract, not walking quality

Quick visual check:

- replaying the same command bin always produces the same reference trajectory
- no target jumps or phase discontinuities are visible in the viewer traces

How this advances the end goal:

- removes interface ambiguity so later milestones measure motion quality, not
  contract mismatch

Tasks:

- define one prior interface:
  - command-indexed library entry
  - preview output
  - nominal `q_ref`
  - foothold targets
  - pelvis / COM targets
  - phase / contact annotations
- deprecate the runtime FSM from the active runtime path
- define one validation schema for reference-only, feasibility, and policy
  probes
- freeze the first reference family as `ZMP`-shaped, with later `ALIP`
  compatibility in the contract

Required schema contents:

- command key:
  - `vx` in the first branch
  - reserved extension fields for `vy` and `yaw_rate`
- timing:
  - `dt`
  - normalized phase
  - step period / cycle length
- nominal targets:
  - `q_ref`
  - optional `dq_ref`
- task-space targets:
  - pelvis pose
  - COM target
  - left/right foot pose targets
- annotations:
  - stance side
  - contact mask / contact state
  - stop / resume markers
- metadata:
  - generator version
  - command-bin bounds
  - validation and feasibility flags

Visual validation:

- prior playback shows alternating step release
- stop command visibly settles into quiet standing
- no visible stepping-in-place when forward command is zero

Metric validation:

- command-conditioned outputs are deterministic and bounded
- preview trajectories are continuous
- no discontinuous target jumps above the configured rate limit
- the same library entry can be consumed by both:
  - bounded residual policy
  - future broader-authority policy

### `v0.20.0-B` `ZMP` prior generator and viewer

Expectation:

- the prior itself should already look like a walk family, before PPO is
  involved

Quick visual check:

- at walk command, feet clearly step out with alternating left/right pattern
- at stop command, posture settles to quiet standing
- resume restarts stepping without a jump

How this advances the end goal:

- proves locomotion semantics are encoded in the prior rather than invented by
  PPO

Tasks:

- implement the first `ZMP`-shaped offline prior library
- use a concrete, reviewable `ZMP` generation pipeline rather than a vague
  “preview-like” approximation
- provide a viewer for:
  - prior-only playback
  - prior + IK playback
  - prior traces over command sweeps

Reference implementation target:

- algorithm family: Kajita et al. `ZMP` preview control over a cart-table /
  `LIPM` model
- public implementation reference: ToddlerBot
  `toddlerbot.algorithms.zmp_planner` and
  `toddlerbot.algorithms.zmp_walk`
- intended WildRobot pipeline:
  - command bin / footstep plan
  - `ZMP` preview-control COM plan
  - swing-foot trajectory generation
  - IK / nominal joint target generation
  - lookup-table export indexed by command

Visual validation:

- at `cmd=0.15`, the feet clearly step out rather than shuffle
- pelvis stays bounded and does not dive forward
- left/right stepping is visibly alternating and repeatable
- stop/resume transitions are visible and non-chaotic

Metric validation:

- commanded foothold mean at `cmd=0.15` is `>= 0.045 m`
- commanded step period remains within `0.45-0.75 s`
- previewed COM and pelvis targets stay bounded and phase-consistent

### `v0.20.0-C` Reference + IK + short-horizon feasibility validation

Expectation:

- prior motion remains trackable after IK and full env dynamics for short
  horizons

Quick visual check:

- startup and one-to-two gait cycles stay coherent in full env replay
- if failure occurs, it looks like bounded balance/tracking limit, not
  incoherent reference behavior

How this advances the end goal:

- confirms the prior is physically usable as a learning target for PPO

Tasks:

- run the prior with IK / nominal targets in full MuJoCo env
- verify one-to-two gait cycles, startup, and stop/resume remain trackable
- do not require long-horizon open-loop walking from the offline prior

Measurement contract for this gate:

- `control steps` means env control-loop ticks (`ctrl_dt`), not MuJoCo
  substeps
- `footfalls` means touchdown events (left/right contact transitions), not
  control-step count
- use both:
  - control steps for short-horizon survival timing
  - touchdown events for gait-cycle progression

Visual validation:

- startup, stop, and one-to-two gait cycles remain coherent in full env replay
- the motion looks trackable rather than instantly impossible
- failures look like bounded tracking or contact limits, not incoherent
  reference design

Metric validation:

- short-horizon env replay survives at least `45` control steps before
  catastrophic pitch termination (equivalent to `>= 0.9 s` at `ctrl_dt=0.02`)
- touchdown progression includes at least one left touchdown and one right
  touchdown in the same replay
- simulated tracking RMSE stays within the actuator-feasibility budget
- touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`

### `v0.20.0-D` PPO unblock decision

Expectation:

- this is a gate decision, not a new behavior milestone

Quick visual check:

- only unblock if prior replay already looks like a real walk family and not a
  balance exploit

How this advances the end goal:

- prevents restarting PPO on an unready prior and repeating `v0.19.5` failure
  modes

PPO is unblocked only if:

- `v0.20.0-A` through `v0.20.0-C` pass
- the offline prior visually looks like a real walk family, not a balance
  exploit
- the remaining issues are plausibly correction-layer problems


---

## Region C — Per-smoke milestone plans (replaced 2026-05-24)

The block below was lines ~1353-1528 of `walking_training.md` and held the launch checklists for v0.20.1-smoke6-prep3 and v0.20.1-smoke7.  Both have long since been superseded by the smoke13/smoke14/smoke15 line; the launch-checklist pattern is now carried per-config in the YAML headers + the CHANGELOG.

### `v0.20.1-smoke6-prep3` Loader fix + actually-test smoke6

Status: **active**

Smoke6 (`offline-run-20260423_213122-v1kclz2w`) was a **null run**.  A
YAML loader bug in
`training/configs/training_config.py:_parse_reward_weights_config`
silently dropped the `lin_vel_z` and `ang_vel_xy` weights set in the
smoke YAML — both fell through to the dataclass default of 0.0, so the
TB phase-signal hypothesis was never tested.  Only `ref_contact_match`
(boolean, w=1.0) was actually enabled; the "stride still fails" outcome
is essentially smoke5 + an inert boolean contact bonus.

Tasks:

1. Fix `_parse_reward_weights_config`: add
   `lin_vel_z=rewards.get("lin_vel_z", 0.0)`,
   `ang_vel_xy=rewards.get("ang_vel_xy", 0.0)`,
   `lin_vel_z_alpha=rewards.get("lin_vel_z_alpha", 200.0)`,
   `ang_vel_xy_alpha=rewards.get("ang_vel_xy_alpha", 0.5)`.
2. Add `tests/test_config_load_smoke6.py`: load the smoke YAML, assert
   every key under `reward_weights:` round-trips to the dataclass field
   with the same value.  Catches this class of bug at config-load time
   instead of at training-time-result interpretation.
3. Re-run smoke6 unchanged.

Decision rule after the re-run:

- if `reward/lin_vel_z` and `reward/ang_vel_xy` are alive at
  training-time and stride moves → smoke6 succeeds late; promote.
- if both terms are alive but stride is still parked → smoke6
  hypothesis falsified; phase signals are not the missing piece;
  proceed to smoke7 (multi-cmd curriculum).
- if either term is dead at training-time err scale → recalibrate α
  per the smoke6-prep2 commitment, single rerun.

### `v0.20.1-smoke7` Promote TB-inspired command/DR pressure

Status: **active, Phase 9D-rescoped** (smoke6-prep3 result falsified
phase-signal hypothesis; five reward-only smokes 3/4/5/6/6-prep3
exhausted reward-only space; April 25 smoke7 is obsolete for the current
21-DOF model; Phase 9A operating point shipped then superseded by
Phase 9D's Froude-similar `cycle_time=0.96 s` + `vx=0.20 m/s` —
CHANGELOG `v0.20.1-phase9D-cycle-time-scaling`).

**Scope revision (2026-04-24):** the original smoke7 plan was "multi-cmd
alone, no DR" with the hypothesis "vx=0.20 hits a cadence ceiling →
forces stride extension."  An open-loop probe at vx ∈ {0.10, 0.15, 0.20,
0.25} via `tools/v0200c_per_frame_probe.py` falsified that mechanism:
WR's prior uses **fixed cycle_time = 0.640 s** across all vx bins
(cadence 1.56 Hz/foot regardless of cmd; only stride per step varies).
PPO's current shuffle at 6.5 Hz/foot already runs 4× the prior's cadence,
and at vx=0.20 needs only 9.5 Hz/foot — well within servo capability.
**No cadence ceiling exists in the WR prior at any vx ∈ [0.10, 0.25].**

TB's prior (`toddlerbot/locomotion/mjx_config.py:98`) is also fixed-cadence
(cycle_time = 0.72 s).  TB's anti-shuffle mechanism is therefore not a
cadence ceiling either — it's the combined effect of multi-cmd
distribution + domain randomization (backlash + friction + kp + IMU
noise make micro-shuffles fragile; pushes are NOT the load-bearing piece
since TB's `add_push` defaults to False for walk).

Smoke7 enables TB-aligned command/DR pressure to test which combination
of mechanisms is load-bearing.  Challenge: this is not yet full
ToddlerBot multi-command reference conditioning, because WR's
`RuntimeReferenceService` still serves one fixed q_ref trajectory.  Treat
this as a fixed-operating-point prior smoke with randomized command
pressure, not proof that the multi-command reference stack is complete.

Scope:

- **Multi-cmd curriculum** (TB anti-shuffle lever #1):
  - `cmd_resample_steps: 150` (3.0 s at ctrl_dt=0.02 — TB resample_time)
  - `cmd_zero_chance: 0.2` (TB default)
  - `cmd_deadzone: 0.05` (TB vx-channel default)
  - vx range: `[0.0, 0.30]` — covers the Phase 9D operating point
    (`0.20`) with generous upper-bracket robustness pressure (1.5×
    operating point) while still deferring negative cmd to v0.20.4.
    Range was historically `[0.0, 0.20]` pre-Phase-9A, then `[0.0,
    0.30]` post-Phase-9A; kept at `[0.0, 0.30]` through Phase 9D so
    smoke results stay directly comparable across the operating-point
    transitions.
- **Domain randomization** (TB anti-shuffle lever #2):
  - `domain_rand_friction_range: [0.4, 1.0]` (TB)
  - `domain_rand_frictionloss_scale_range: [0.8, 1.2]` (TB)
  - `domain_rand_kp_scale_range: [0.9, 1.1]` (TB)
  - `domain_rand_mass_scale_range: [0.9, 1.1]` (WR-specific multiplicative;
    TB uses additive `body_mass_range: [-0.2, 0.2]` kg — different
    semantics, kept WR's existing form)
  - `domain_rand_joint_offset_rad: 0.03` (WR-specific reset-time joint
    offset; TB has no equivalent — TB uses `rand_init_state_indices`
    instead)
- **IMU noise** (sim2real hardening, secondary):
  - `imu_gyro_noise_std: 0.05 rad/s` (conservative vs TB's gyro_std=0.25
    under bias-walk RW)
  - `imu_quat_noise_deg: 2.0` (conservative vs TB's quat_std=0.10 rad ≈
    5.7°)
- **Eval-cmd override** (G4 readout protection):
  - `eval_velocity_cmd: 0.20` — pins eval rollouts at the Phase 9D
    operating point.  Without this, eval would sample uniformly across
    `[0.0, 0.30]` and dilute G4.  The fixed q_ref trajectory is also
    pinned to `loc_ref_offline_command_vx: 0.20`; q_ref/eval-cmd
    mismatch is a null smoke.
- **NOT enabling**: pushes (TB walk default add_push: False), yaw cmd
  (defer to v0.20.4 — WR prior lacks yaw), negative vx (prior lacks
  negative-vx generation).
- Reward family unchanged from smoke6-prep3 (TB-complete and verified
  alive in smoke6-prep3).

Why DR + command pressure jointly (Option C) and not isolated:

- Per the probe results, the cadence-ceiling story is wrong; multi-cmd
  alone has weak a-priori support.
- DR alone would test "is robustness pressure load-bearing for shuffle?"
  cleanly, but the v0.20.1 fail-mode tree has no precedent for which
  TB-inspired knob is load-bearing.
- Per CLAUDE.md "follow ToddlerBot before WR-specific design", TB enables
  both by default — Option C is the closest to TB's actual recipe.
- Trade-off accepted: if smoke7 succeeds, we don't know which mechanism
  was load-bearing without a follow-up ablation (smoke7-ablate-DR or
  smoke7-ablate-multicmd).  That's a known follow-up cost, not a
  blocker.  If smoke7 fails, we have stronger evidence the bottleneck
  is the prior, not the training distribution.

Pre-smoke checks (all completed in smoke7-prep1):

- ✅ DR plumbing audit: `wildrobot_env.py` `_sample_domain_rand_params`
  → `_get_randomized_mjx_model` applied per ctrl step.
- ✅ Multi-cmd resample plumbing audit: `wildrobot_env.py:1644-1664`
  resamples velocity_cmd at `cmd_resample_steps` interval, honors
  zero_chance + deadzone via shared `_sample_velocity_cmd`.
- ✅ YAML loader audit: all DR + cmd_* keys read by
  `_parse_env_config` (training_config.py:480-504).
- ✅ Round-trip test extended to assert smoke7 critical env settings.
- ✅ Zero-residual invariant: 7/7 pass with smoke7 YAML.
- ✅ Eval-cmd override behaviorally verified: `reset_for_eval` pins
  cmd to the configured eval command across all seeds; training reset
  samples within the configured `[min_velocity, max_velocity]` range
  with proper zero_chance/deadzone distribution.
- ✅ Open-loop prior probe at vx ∈ {0.10, 0.15, 0.20, 0.25}: prior
  generates 1120-step trajectories at all bins; no immediate collapse.
- ✅ Phase 9D q_ref/eval alignment: `eval_velocity_cmd` and
  `loc_ref_offline_command_vx` both pin `0.20`, and the env builds an
  explicit one-bin library for that value so it cannot snap to a stale
  default-grid bin.

Decision rule after smoke7:

- **stride at eval (vx=0.20) ≥ 0.050 m** → TB-inspired command/DR hypothesis
  confirmed; promote to v0.20.2 (SysID hardening) and run ablation
  follow-ups (DR-only, multi-cmd-only) to isolate the load-bearing
  mechanism.
- **stride still parked below ~0.030 m** → TB-inspired command/DR pressure
  is not sufficient on WR.  Two candidates:
  - (a) prior is the actual blocker — return to v0.20.0-x prior
    surgery (extend prior to negative vx, re-evaluate prior fidelity
    under the closed-loop dynamics PPO actually faces).
  - (b) smoke compute budget exhausted before convergence — TB trains
    on 2× the episode length (1000 vs 500) and broader command range;
    extend the v0.20.1 smoke compute cap to ~300 iters and rerun before declaring failure.
- **mixed signal at intermediate stride (0.030-0.050)** → smoke7 is on
  the right track but needs more compute or true command-conditioned
  q_ref lookup before declaring the prior blocked.

What we will NOT do in smoke7:

- not enable pushes (TB walk default; defer to v0.20.5).
- not enable yaw cmd (defer to v0.20.4).
- not add stride-amplitude reward terms (TB doesn't have one; five
  smokes ruled out the reward space).
- not widen the residual bound past ±0.25 rad (G5 has ample headroom:
  smoke6-prep3 residuals 0.097 vs 0.20 cap).
- not relax G5.
- not extend the v0.20.1 smoke compute budget yet (judge first at the standard
  150 iters; if intermediate result, then extend).

