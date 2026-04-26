# Reference Architecture Comparison — WildRobot vs ToddlerBot

**Status:** Active comparison note for `v0.20.x` reference-pivot work
**Last updated:** 2026-04-25
**Purpose:** Code-grounded side-by-side of the two projects' walking
reference stacks, end-to-end. Use this to identify which WR design
choices are TB-aligned, which deviate with a load-bearing rationale,
and which deviate without one (the alignment backlog).

All file paths in this doc are repo-relative:
- TB: `projects/toddlerbot/...`
- WR: `projects/wildrobot/...`

---

## Pipeline at a glance

Both projects implement Kajita-style ZMP preview LQR over an LIPM
cart-table model. The algorithm family is the same; the architecture
around it diverges.

```
                  ┌──────────────────────────────────────────────────────┐
                  │                 ToddlerBot pipeline                  │
                  │                                                      │
   command        │  ZMPWalk.build_lookup_table  →  walk_zmp_*.lz4       │  ← built once, persisted
   range          │       (offline, all cmd bins)                        │
                  │                          │                           │
                  │                          ▼                           │
                  │  WalkZMPReference (loads lz4 at env init)            │  ← runtime asset
                  │     get_state_ref(t, command, last_state)            │
                  │                          │                           │
                  │                          ▼                           │
                  │  state_ref dict =                                    │
                  │     • lookup-derived (motor_pos, qpos[7:], body_*,   │  ← fixed-base FK quantities
                  │       site_*, stance_mask)                           │
                  │     • path-integrated runtime (path_pos, path_rot,   │  ← command integration
                  │       lin_vel, ang_vel)                              │
                  │     • computed root pose (qpos[0:7] from the two)    │
                  │                          │                           │
                  │                          ▼                           │
                  │  walk_env / mjx_env reward terms consume state_ref   │
                  └──────────────────────────────────────────────────────┘

                  ┌──────────────────────────────────────────────────────┐
                  │                 WildRobot pipeline                   │
                  │                                                      │
   command        │  ZMPWalkGenerator.generate(vx)                       │  ← per-call, on-demand
   (single vx)    │       runs LQR + IK in-process per command bin       │
                  │                          │                           │
                  │                          ▼                           │
                  │  ReferenceLibrary (in-memory dict of trajectories)   │  ← runtime asset
                  │     get_preview(traj_id, step_index, n_preview)      │
                  │                          │                           │
                  │                          ▼                           │
                  │  preview window =                                    │
                  │     • planner-output (q_ref, pelvis_pos, com_pos,    │  ← LIPM-COM integrated,
                  │       left/right_foot_pos COM-relative,              │    NO mujoco_replay step
                  │       stance_foot_id, contact_mask, phase)           │
                  │                          │                           │
                  │                          ▼                           │
                  │  WildRobotEnv reward terms consume win + finite-diff │
                  │  velocity references derived at runtime              │
                  └──────────────────────────────────────────────────────┘
```

---

## Stage 1 — Footstep planner

| Aspect | ToddlerBot | WildRobot |
|---|---|---|
| File | `toddlerbot/algorithms/zmp_walk.py:214 ZMPWalk.plan` | `control/zmp/zmp_walk.py:244 ZMPWalkGenerator.generate` → `_plan_footsteps` |
| Footstep schedule | Alternating L/R every cycle, stride from `command · total_time / (2·num_cycles - 1)` (`zmp_walk.py:260`) | Alternating L/R every cycle, stride = `vx · cycle_time / 2` |
| Stance ratio | `single_double_ratio = 2.0` (`zmp_walk.py:31`) — single support : double support = 2:1 | `single_double_ratio = 2.0` — same |
| Lateral foot offset | `foot_to_com_y` from `robot.yml` | `default_stance_width_m = hip_lateral_offset_m` from `ZMPWalkConfig` |
| Yaw / lateral support | Full 3-DoF command `(vx, vy, yaw_rate)`; in-place yaw uses circular footstep arrangement (`zmp_walk.py:269-305`) | `vx` only; `vy`/`yaw_rate` reserved schema fields, no generator |
| Cycle 0 handling | Plans 22 s of forward walking, then **truncates the first cycle** (`zmp_walk.py:138-140`) so the stored trajectory is the periodic steady-state portion | Plans 22 s, **keeps cycle 0**; cycle-0 is from-rest (right-foot half-step), explicitly non-periodic with later cycles |

**Architectural divergence:** TB's command space is omnidirectional from
day one; WR is sagittal-only. TB enforces periodicity by truncation; WR
enforces non-periodicity by linear-replay-only contract (consumers
explicitly forbidden to wrap, see `reference_library.py:18-19`).

---

## Stage 2 — ZMP preview LQR (COM trajectory)

| Aspect | ToddlerBot | WildRobot |
|---|---|---|
| Algorithm | Kajita preview control over LIPM cart-table | Same |
| Cost weights | `Q = 1.0 · I`, `R = 0.1 · I` (`zmp_walk.py:34-35`) | `Q = 1.0 · I`, `R = 0.1 · I` (`ZMPWalkConfig.zmp_cost_Q/R`) |
| Solver | `ZMPPlanner.plan(time_steps, desired_zmps, x0, com_z, Qy, R)` | Same — WR's `control/zmp/zmp_planner.py` is a deliberate port |
| Initial state | `x0 = [path_pos[0], path_pos[1], 0, 0]` (rest at origin) | `x0 = [0, 0, 0, 0]` (rest at origin) |
| Forward integration | Forward-Euler closed-loop simulation: `x_traj[idx] = x_traj[idx-1] + xd·dt` with `xd = [x_traj[idx-1, 2:], u_traj[idx-1]]`, `u = get_optim_com_acc(t, x)` (`zmp_walk.py:357-381`) | Same forward-Euler closed-loop integration (`zmp_walk.py:467-477`) — round-9 deliberately mirrored TB's `update_step` |

**Architectural divergence:** None. WR explicitly ported TB's COM
integration scheme in round 9.

---

## Stage 3 — Cartesian foot trajectory

| Aspect | ToddlerBot | WildRobot |
|---|---|---|
| Foot DoF planned | Position **and** orientation `(x, y, z, roll, pitch, yaw)` per swing/stance frame (`zmp_walk.py:406 compute_foot_trajectories`) | Position only `(x, y, z)`; orientation slot stored as zeros (`reference_library.py:86-90 left_foot_rpy`) |
| Swing-z shape | **Triangular** linear-up then linear-down (`zmp_walk.py:483-491`); starts and ends at z=0 by construction | **Half-sine** + **constant DC offset** (`zmp_walk.py:519-525`); first/last frame snap to z = `swing_foot_z_floor_clearance_m` |
| Floor clearance offset | None — TB's swing trajectory naturally lifts to `footstep_height` | `swing_foot_z_floor_clearance_m = 0.025 m` added uniformly during swing to compensate for plantarflexed swing ankle |
| Boundary continuity | Smooth at lift-off / touchdown (z=0) | Discontinuity (snaps from 0 → 0.025 at lift-off, 0.025 → 0 at touchdown) — caught as the P2 swing-step gate failure (1.81× TB normalised) |

**Architectural divergence:**
1. TB plans foot orientation; WR plans only position. Downstream
   consequence: TB's IK can realise the planned foot orientation; WR's
   IK substitutes a heuristic ankle.
2. TB's swing trajectory is C0 at boundaries; WR's has the DC-offset
   discontinuity. The DC offset has a load-bearing rationale (see
   Stage 4); the discontinuity does not.

---

## Stage 4 — Inverse kinematics

| Aspect | ToddlerBot | WildRobot |
|---|---|---|
| File | `toddlerbot/algorithms/zmp_walk.py:601 foot_ik` | `control/zmp/zmp_walk.py:181 _solve_sagittal_ik` |
| Leg DoF | 6: hip pitch / hip roll / **hip yaw** / knee / **ankle roll** / ankle pitch | 4: hip pitch / hip roll / knee / ankle pitch — no hip yaw, no ankle roll |
| Method | Closed-form analytic 6-DoF solver from `(target_foot_pos, target_foot_ori)` per side (`zmp_walk.py:617-665`) | Closed-form analytic 4-DoF sagittal solver (`_solve_sagittal_ik`) + small-angle hip roll IK in WR style |
| Ankle pitch handling | Derived: `ank_pitch = target_pitch + knee_pitch - hip_pitch`. If `target_pitch = 0` (planner default), ankle stays flat throughout swing **and** stance | Heuristic split: stance → flat-foot constraint `ankle = -(hip+knee)` clamped; swing → fixed plantarflex `ankle = -0.15 rad` clamped (`_solve_sagittal_ik:225-232`) |
| Foot orientation tracking | Yes — IK realises planner-set `target_pitch` / `target_roll` | No — orientation is heuristic, not planner-driven |
| COM-relative foot input | `target_foot_adjusted = foot_pos_traj - com_pos_traj + foot_to_com_offset` (`zmp_walk.py:568-579`) | `foot_rel_x = foot_world.x - pelvis_x`, `hip_to_foot_z = -(com_height - pelvis_to_hip - ankle_to_ground - foot_z_above_ground)` (`zmp_walk.py:578-583`) |

**Architectural divergence:** Hardware-driven. WR has 4-DoF legs (no
hip yaw, no ankle roll); TB has 6-DoF. The plantarflex-during-swing
heuristic is the WR-specific consequence: with no ankle roll and a
shorter ankle range, the IK can't both keep the foot flat and bend the
knee enough to lift; choosing ankle = -0.15 rad in swing frees up the
knee budget. This rationale is load-bearing for WR.

The downstream cost is the foot collision box dipping below the foot
body during swing, which the `swing_foot_z_floor_clearance_m = 0.025 m`
offset was added to compensate (see Stage 3). Removing the WR plantar-
flex would eliminate the need for the offset, but requires a more
sophisticated IK strategy (e.g. allow target-pitch input like TB).

---

## Stage 5 — Reference enrichment via `mujoco_replay` (the critical divergence)

| Aspect | ToddlerBot | WildRobot |
|---|---|---|
| Step F exists? | **Yes** — `ZMPWalk.mujoco_replay` (`zmp_walk.py:153-212`) | **No** — planner output stored as-is |
| Sim setup | `MuJoCoSim(robot, fixed_base=True)` — body anchored at `home` keyframe | n/a |
| Per-frame action | `sim.set_joint_angles(joint_pos)`, `sim.forward()`, harvest FK quantities | n/a |
| Harvested fields | `qpos`, `body_pos`, `body_quat`, `body_lin_vel`, `body_ang_vel`, `site_pos`, `site_quat`, plus the planner's `stance_mask_ref` as `contact` | n/a |
| Storage | All harvested arrays go into `Motion` dataclass (`motion_ref.py:24-34`), persisted in lz4 | n/a — `ReferenceTrajectory` stores only planner-output cartesian fields |

**Architectural divergence:** This is the single biggest pipeline-shape
difference. TB's reference asset contains FK-realised body and site
quantities that the env / RSI / debugging tools can read directly.
WR's reference asset contains only planner intent; any FK-realised
quantity must be reconstructed downstream.

**Downstream consequences:**
- TB's RSI (`mjx_env.py:1175-1191`) initialises freejoint qvel from
  stored `body_lin_vel` / `body_ang_vel`. WR can't do this — falls
  back to qvel = 0 at reset.
- TB's reward functions never have to re-run FK. WR's parity tooling
  has to re-FK every frame to compare apples-to-apples (which is what
  `tools/reference_geometry_parity.py:_wr_fk_and_smoothness` now does).
- Reference velocity quality: TB has `body_lin_vel` / `body_ang_vel`
  from MuJoCo's analytical FK derivative (clean). WR computes velocity
  references on-the-fly via finite-diff of stored positions (the G2
  decision in `walking_training.md`) — which is also clean for our
  planner's smooth signals, but doesn't generalise to noisier prior
  generators (ALIP, mocap retarget).

---

## Stage 6 — Reference asset format

| Aspect | ToddlerBot | WildRobot |
|---|---|---|
| Schema | `Motion` dataclass — flat dict of arrays per command bin (`motion_ref.py:24-34`) | `ReferenceTrajectory` dataclass — flat dict of arrays per `vx` bin (`reference_library.py:46-103`) |
| Lifecycle | Pre-baked at build time, loaded once at env init via `joblib.load(walk_zmp_*.lz4)` (`walk_zmp_ref.py:38-47`) | Generated on-demand at env startup via `ZMPWalkGenerator.generate(vx)`; in-memory only; no on-disk persistence |
| Command grid | `vx ∈ [-0.2, 0.4]`, `vy ∈ [-0.2, 0.2]`, `yaw_rate ∈ [-0.8, 0.8]` at `interval = 0.05` (`zmp_walk.py:84-89`) | Single `vx` per call; broader coverage requires multiple `generate()` calls |
| Lookup mechanism | `nearest_command_idx = argmin(||lookup_keys - command||)` (`walk_zmp_ref.py:138-140`); O(n_bins) per query | `library.get_trajectory(vx_bin)`; O(1) per query but generator runs once per new bin |
| Wrap behaviour | `step_idx = (round(t/dt) % episode_len)` — modulo wrap is safe because cycle 0 was truncated (`walk_zmp_ref.py:141`) | Linear replay only; wrap forbidden (`reference_library.py:53-56`); `get_preview` issues a warning past trajectory end (`reference_library.py:362-368`) |

**Architectural divergence:** TB ships a compact, cached, omnidirectional
key-value asset. WR ships a code-that-generates-data interface, single
sagittal command at a time. Functionally equivalent for a single-cmd
smoke; meaningful when smoke7+ introduces multi-cmd resampling.

---

## Stage 7 — Runtime reference service / `get_state_ref`

This is where TB does the bulk of its runtime work. WR's
`runtime_reference_service.py` is intentionally thin.

### TB: `WalkZMPReference.get_state_ref` (`walk_zmp_ref.py:109-250`)

Builds a per-step `state_ref` dict by combining three sources:

1. **Path-state integration** (`integrate_path_state` in `motion_ref.py:200-235`):
   `path_pos = path_pos_prev + path_rot · lin_vel · dt`,
   `path_rot = path_rot_prev · delta_rot(ang_vel · dt)`. `lin_vel` and
   `ang_vel` come straight from the command. **This is where the
   commanded path target lives.**
2. **Lookup query** by `nearest_command_idx` and `step_idx`:
   pulls `qpos`, `body_pos`, `body_quat`, `body_lin_vel`,
   `body_ang_vel`, `site_pos`, `site_quat`, `stance_mask` for the
   current frame.
3. **Runtime root-pose composition**:
   `root_pos = path_rot.apply(default_pos) + path_pos`,
   `root_quat = path_rot · default_rot`,
   `qpos = [root_pos, root_quat, lookup_qpos[7:]]`.

The output `state_ref` is consumed by all reward functions in
`mjx_env.py` (see Stage 8).

### WR: `RuntimeReferenceService.get_preview` (`runtime_reference_service.py`)

Does **only** lookup + slice — pulls the current frame and an optional
1–2 frame future preview from the in-memory trajectory. **No path
integration**, **no command-driven runtime computation**, **no root-
pose composition**. The env wrapper assembles whatever it needs from
the raw window.

**Architectural divergence:** TB's runtime service is a *stateful
integrator + lookup combiner*; WR's is a *stateless slice*. This means
TB's reference target is "where the command says you should be by now",
and WR's reference target is "what the planner said for this phase
of the gait". These are different reference philosophies.

---

## Stage 8 — Env consumption / reward terms

The most important architectural axis. **What does the imitation
reward actually compare against?**

| Reward term | TB source (state_ref field) | WR source (win field) | Same conceptually? |
|---|---|---|---|
| `motor_pos` ↔ `ref_q_track` | `state_ref["motor_pos"]` — IK-output joint angles converted via `robot.joint_to_motor_angles` at build time | `win["q_ref"]` — IK-output q_ref directly | **Yes** (both: planner IK joint targets) |
| `torso_pos_xy` ↔ `ref_torso_pos_xy` | `state_ref["qpos"][:2]` = path-integrated `path_pos`, command-driven (`mjx_env.py:2264-2270`) | `win["pelvis_pos"][:2]` = LQR-integrated COM trajectory from the planner (`wildrobot_env.py:801, 814-818`) | **No** — TB tracks "command-integrated path"; WR tracks "LIPM planner output" |
| `torso_quat` ↔ `ref_body_quat_track` | `state_ref["qpos"][3:7]` = `path_rot · default_rot`, command-driven (`mjx_env.py:2316-2326`) | reference is identity quat (since `pelvis_rpy = 0` in WR planner output); reward = `exp(-α · angle²)` against current root quat (`wildrobot_env.py:769-779`) | Same in spirit (yaw-stationary in both); WR doesn't even read a stored ref quat |
| `lin_vel_xy` ↔ `cmd_forward_velocity_track` | `state_ref["lin_vel"][:2]` = `command[5:7]` directly (`walk_zmp_ref.py:104`) | tracked via `cmd_forward_velocity_track` against `velocity_cmd` | **Yes** — both compare body lin_vel against the command, not against the planner |
| `lin_vel_z` | `state_ref["lin_vel"][2]` = command's z = **0** (constant) (`mjx_env.py:2369-2374`) | finite-diff of `win["pelvis_pos"][2]` — but `pelvis_pos[:, 2] = com_height_m` (constant) → also **0** in practice (`wildrobot_env.py:840`) | **Yes** — both reduce to "body z-velocity should be 0" |
| `ang_vel_xy` | `state_ref["ang_vel"][:2]` = `[0, 0, yaw_rate_command][:2]` = **0** | reference = 0 since `pelvis_rpy = 0` throughout, reward penalises `gyro_rad_s[:2]` (`wildrobot_env.py:855-860`) | **Yes** — both target zero pitch/roll rate |
| `feet_contact` ↔ `ref_contact_match` | `info["state_ref"]["stance_mask"]` from lookup planner stance (`mjx_env.py:2484`) | `win["contact_mask"]` from WR planner stance | **Yes** — both compare physical contact to planner stance |
| `feet_air_time` | `info["feet_air_time"]` derived from physical contacts at runtime, gated on `||cmd|| > 1e-6` (TB) | recently added in WR smoke6/7 with same TB-aligned formula | TB-aligned by recent WR work |

**Architectural divergence in the reward:** Only **`torso_pos_xy`**
materially differs. TB tracks the command-integrated body path; WR
tracks the planner-LIPM body path. The other reward terms turn out to
be equivalent or aligned (often because the planner output is "pelvis
flat, COM at constant z, no yaw rotation", which collapses several TB
fields to constants WR also uses).

---

## What's actually different at the architecture level

After tracing every stage, the architectural differences sort into
three groups:

### Group 1: Hardware-driven (load-bearing, keep WR-specific)

- **Stage 4 IK**: 4-DoF leg vs 6-DoF leg. Drives the analytic-IK form,
  the swing-ankle plantarflex heuristic, and (via Stage 3) the floor-
  clearance DC offset.
- **Robot dimensions**: WR ~1.6–1.8× larger; affects every absolute-
  units metric (already addressed by the size-normalised parity).

### Group 2: Different choices, no clear rationale

- **Stage 3 swing-z shape**: TB triangular C0; WR half-sine + DC
  offset with C0 discontinuity. The DC offset is load-bearing (Group
  1 consequence); the discontinuity is not. **Alignment candidate:
  ride the offset on `sin(π·frac)` to remove the discontinuity.**
- **Stage 5 mujoco_replay**: TB has it; WR doesn't. The G2 finite-diff
  decision avoids the schema growth, but loses RSI-quality velocity
  references and forces every downstream consumer to re-FK if it
  needs body/site quantities. **Alignment candidate: add a fixed-base
  FK pass after IK during library construction; persist body/site
  arrays alongside `q_ref`.**
- **Stage 6 lifecycle**: TB pre-bakes a multi-cmd lz4; WR generates
  on-demand per `vx`. Functionally equivalent today; multi-cmd smoke
  re-runs the planner per cmd switch instead of doing O(1) lookup.
  **Alignment candidate: build a vx-grid library at env init and
  cache it (in-memory or on-disk).**
- **Stage 7 runtime service**: TB does path integration + root-pose
  composition + command-driven velocity. WR does only slice. The
  consequence is the Stage 8 reward divergence around `torso_pos_xy`
  reference. **Alignment candidate: add path-state integration to
  WR's runtime service so the env can compare against a command-
  integrated body target — same concept TB uses.**

### Group 3: Same value, different default — covered in earlier proposal

- `cycle_time` (0.64 vs 0.72)
- `foot_step_height` (0.04 vs 0.05)

These are configuration values, not architecture choices. They can
flip independently.

---

## Reference target philosophy — the meta-architectural choice

The cleanest way to read the divergence is the **reward target**:

- **TB's policy is asked to track** *where the body should be if it had
  perfectly executed the command since reset*, with the planner's IK
  output supplying joint-level shape.
- **WR's policy is asked to track** *what the planner says the body
  will look like at this phase of the gait*, also with planner IK
  output supplying joint-level shape.

The two are equivalent when the planner perfectly tracks the command.
They diverge when LIPM dynamics produce overshoot or oscillation that
the policy is supposed to absorb (TB: yes, the policy absorbs LIPM
deviation; WR: no, the policy is told to copy LIPM deviation).

This is a deliberate-or-accidental design choice, and it's the one
question I'd want a load-bearing answer for before any of the Group 2
alignment moves: **does the WR policy benefit from tracking the
LIPM-deviation reference, or should it track a clean command-
integrated reference like TB?**

If the answer is "track command-integrated", then:
- Stage 7 needs `integrate_path_state`-style runtime composition.
- Stage 8 `ref_torso_pos_xy` should compare against the integrated
  target, not `pelvis_pos`.
- The G2 finite-diff finite-velocity reference becomes simpler: just
  the command's `lin_vel`, plus a constant zero for `lin_vel_z` /
  `ang_vel_xy`.

If the answer is "track LIPM-deviation" (current WR), then no Stage 7
change is needed — but the reasoning should be written down so the
divergence is explicit.

---

## Recommended alignment ordering

In dependency order, smallest first:

1. **Stage 3 swing-z continuity fix.** Pure planner-side smoothness
   change; no schema impact. Closes the P0 / P2 swing-step gates.
2. **Decide the Stage 7 reference-target philosophy** (LIPM-track vs
   command-track). This is a design call, not a code change yet —
   needed before either Stage 5 or Stage 7 alignment can be scoped.
3. **Stage 5 mujoco_replay step.** Schema-growing; persist body /
   site / velocity arrays. Unblocks RSI parity, removes finite-diff
   from runtime, makes WR's library content type match TB's.
4. **Stage 7 runtime composition** (only if step 2 chose "command-
   track"). Adds `integrate_path_state` to the runtime service.
5. **Stage 6 pre-bake.** Caching optimisation; valuable when smoke7+
   resamples cmd often. Cosmetic until then.
6. **Stage 1 yaw / lateral generators.** Already scoped to `v0.20.4`
   in `walking_training.md`; proper TB alignment requires extending
   the planner to handle `vy` and `yaw_rate`.

Group 1 (hardware-driven) and Group 3 (parameter values) are
independent of this ordering and can land any time.
