# Servo Calibration Design (Runtime)

## Context

This repo currently has two runtime “stacks”:

- **Legacy runtime modules** under `runtime/configs/` + `runtime/hardware/`
  - config file: `runtime/configs/wr_runtime_config.json`
  - servo driver: `runtime/hardware/hiwonder_board_controller.py` (board protocol)
  - servo calibration fields today: per-servo `offset` (in servo units)
- **Newer sim2real runtime** under `runtime/wr_runtime/` (used by the Stage 1 `policy_contract` flow docs)

This design doc targets the legacy stack because you explicitly want:
- joint names read from `runtime/configs/wr_runtime_config.json`
- calibration results written back to that same file
- servo communication via the **Hiwonder board controller** (not the actuator convenience wrapper)

WildRobot hardware runtime (`runtime/wr_runtime/`) currently assumes:
- Hiwonder servo position units are in `[0..1000]`
- `500` is the nominal center position
- mapping is linear over `[-120°, +120°]` (240° total range)
- runtime commands and reads joints in radians, using per-joint `joint_offsets_rad` from the runtime JSON config

Today, runtime applies only a **per-joint offset** (in radians). It does **not** support per-joint direction/sign correction, which is needed when a physical servo is installed such that increasing servo units corresponds to negative MuJoCo joint motion (or vice versa).

This doc proposes a servo calibration flow and config format to support:
1) per-joint center offset calibration
2) per-joint direction calibration

The implementation target is a new interactive script `calibrate.py` (exact location/entrypoint specified below).

## Goals

- Calibrate each servo’s **zero/center offset** so that a chosen physical “neutral pose” corresponds to `joint_pos_rad == 0.0` in runtime.
- Calibrate each servo’s **direction sign** so that “positive radians” in runtime matches “positive joint angle” in the MuJoCo model used for training/inference.
- Persist calibration results into the runtime JSON config in a way that is robust, debuggable, and easy to edit by hand.
- Keep the runtime API boundary in radians (do not leak servo units into policy/contract code).

## Non-goals (for initial version)

- Endstop/range calibration (finding actual mechanical min/max).
- Torque/current calibration or PID tuning.
- IMU calibration.
- Automatically inferring MuJoCo “positive direction” physically (we will require human confirmation).

## Terminology

- `units`: servo position command/readback in `[0..1000]`
- `units_center`: nominal center (`500`)
- `range_rad`: full servo travel in radians (`240° = 4.1887902047863905 rad`)
- `s`: direction sign per joint (`+1` or `-1`)
- `offset_units`: offset per joint (servo units), applied at the runtime↔servo conversion boundary

## Current Runtime Behavior (baseline)

### Legacy stack (`runtime/configs` + `runtime/hardware`)

In `runtime/configs/config.py` (`ServoControllerConfig`), mapping is effectively:

- servo units are `[0..1000]` with `500` as center
- runtime composes:
  - a policy/action convention sign (`mirror_sign` from `assets/robot_config.yaml`)
  - a per-robot hardware direction (`direction`, stored in the JSON)
  - a servo center offset (`offset`, stored in the JSON as an integer)

Today it supports `offset`, but does not have an explicit “per-robot direction correction” field in the JSON.

### Why we need an additional direction field

`mirror_sign` is a *model/training convention used when mapping policy actions → joint targets*.
In practice, a physical build can still be flipped due to servo installation/wiring, so we need a per-robot correction (`direction`) that applies when converting **joint targets (radians)** to **servo units**.

## Proposed Config Additions

Use `runtime/configs/wr_runtime_config.json` as the canonical “in-repo” calibration target:
- calibration reads joint names (and servo IDs) from this file
- calibration writes results back to this file (in-place with backups, unless `--output` is provided)

### Config shape (recommended, minimal change)

Extend the existing legacy shape under `Hiwonder_controller.servos.<joint>`:

```json
{
  "Hiwonder_controller": {
    "servos": {
      "left_hip_pitch": { "id": 1, "offset": 0, "direction": 1 }
    }
  }
}
```

Where:
- `offset` is an integer in servo units (default: `0`)
- `direction` is `+1` or `-1` (default: `+1`)

Backwards compatibility:
- If `direction` is missing, assume `+1`.
- Existing configs that only contain `id` and `offset` remain valid.

## Proposed Mapping With Offset + Direction

For each joint:
- `direction` from JSON in `{+1, -1}`
- `offset_units` from JSON (int)

Note:
- `mirror_sign` should **not** be applied when converting an absolute joint target `target_rad` (MuJoCo qpos) into servo units. It only applies when mapping *policy actions* into joint targets.

Command (radians → units, centered at 500):
```
units_cmd = clamp_0_1000(
  units_center + direction * target_rad * (1000 / range_rad) + offset_units
)
```

Readback (units → radians):
```
pos_rad = direction * (units_read - units_center - offset_units) * (range_rad / 1000)
```

This preserves the meaning of `offset_units` as “what runtime adds to units after the centered conversion”, while allowing sign flips.

## Calibration UX Overview

Calibration is interactive because:
- The “correct” neutral pose is a physical choice (a human aligns the limb).
- Correct sign requires interpreting the MuJoCo joint’s positive direction, which depends on the model convention.

The script should guide the user joint-by-joint:
0) optionally move the whole robot to a known-safe “home pose” (MuJoCo keyframe)
1) confirm servo ID / comms
2) determine sign (`+1` or `-1`)
3) determine offset at neutral pose
4) write config (with backups)

Important: calibrate `direction` *before* `offset_units`, because the “direction test” is easier to reason about when the joint is near center.

## Calibration Procedure (per joint)

### Preconditions / safety checklist

- Robot lifted or supported so legs can swing freely.
- Clear access to power switch / e-stop.
- Start with small step sizes (e.g., 2–5 servo units) to avoid slamming into hard stops.
- Set conservative move duration (e.g., 200–500 ms for manual jogging).
- Provide a “panic/unload” key that immediately unloads all servos and exits.

### Optional: go to Home Pose (recommended)

The repo’s sim2real flow recommends using `spec.robot.home_ctrl_rad` as the canonical safe pose (exported from the MuJoCo home keyframe into the policy bundle).

Add a calibrator option to set the robot to that home pose before per-joint calibration:

- CLI: `--go-home` (move all joints once, then proceed)
- CLI: `--bundle <path>` (folder containing `policy_spec.json`)
- CLI: `--scene-xml <path>` (MuJoCo scene XML; e.g. `assets/scene_flat_terrain.xml`)
- CLI: `--keyframes-xml <path>` (direct `keyframes.xml`; e.g. `assets/keyframes.xml`)

Home pose source precedence:
1) If `--bundle` is provided: read `policy_spec.json` and use `spec.robot.home_ctrl_rad` (preferred; does not require MuJoCo installed on the robot).
2) Else if `--scene-xml` is provided and MuJoCo is available: load the scene XML and read the `"home"` keyframe (or keyframe 0) to compute `home_ctrl_rad` in actuator order (matches training/export behavior).
3) Else if `--keyframes-xml` is provided: parse the XML, find `<key name="home" qpos="...">`, and extract the actuated joint slice from `qpos`.
4) Else fallback: `home_ctrl_rad = [0.0] * N` (servo centers).

#### Reading `home_ctrl_rad` from `keyframes.xml` (no MuJoCo dependency)

This repo includes `assets/keyframes.xml` with a comment that documents the `qpos` layout:
- `[0:3]` root position
- `[3:7]` root quaternion
- `[7:15]` actuated joints (8 DOFs) in radians in the expected joint order

For Stage 1 (8 actuators), the calibrator can:
- split the `qpos` string into floats
- take `qpos[7:7+N]` where `N == len(joint_names_from_config)`
- interpret that as `home_ctrl_rad` in the same joint order as the config

If `N != 8`, do not guess; either require MuJoCo (`--scene-xml`) for name-based extraction or require an explicit mapping.

Execution detail:
- Convert `home_ctrl_rad` (absolute joint targets in radians) to servo commands using the mapping above (using `direction` and `offset_units`).
- Use a conservative move time (e.g., 500–1000 ms) and allow user confirmation before moving.

### Step A: Select joint

Inputs:
- config path (reads servo IDs, existing offsets/signs)
- list of joints to calibrate (default: config’s joint order / MJCF actuator order)

For each joint, print:
- joint name
- servo ID
- current `sign` and `offset_unit` (legacy `offset_rad` is accepted but interpreted as the same servo-unit value)

### Step B: Determine direction sign

The script should:
1) move the joint near center (safe) using `units_center + offset_units` (or defaults)
2) prompt the user to run a direction test:
   - apply a small positive delta (e.g., `+0.1 rad` or `+20 units`)
   - apply a small negative delta
3) ask the user an *explicit physical-direction question*, not “MuJoCo-positive”.

#### Why this matters

“MuJoCo-positive direction” is hard to answer on hardware unless you already know the model’s sign convention for that joint. Humans answer much more reliably when the prompt is phrased as a concrete motion: forward/backward, left/right, toes up/down, etc.

#### Recommended prompt format

For each joint, define a short `positive_motion_hint` string (see the table below), then prompt:

> I will command `+delta`. Did the joint move **toward**: `<positive_motion_hint>`?
>
> - `y` = yes (set `direction = +1`)
> - `n` = no  (set `direction = -1`)
> - `r` = repeat with a different delta

#### Default positive-motion hints (must be verified against the MJCF)

These are suggested human-friendly hints in the robot body frame:
- Body frame assumption for wording: **forward/back** relative to robot facing, **left/right** relative to robot, **up/down** relative to gravity.

| Joint name | `positive_motion_hint` (example wording) |
|---|---|
| `left_hip_pitch` | “left leg swings forward” |
| `right_hip_pitch` | “right leg swings forward” |
| `left_knee_pitch` | “left knee bends (foot moves backward toward the body)” |
| `right_knee_pitch` | “right knee bends (foot moves backward toward the body)” |
| `left_ankle_pitch` | “left toes go up (dorsiflex)” |
| `right_ankle_pitch` | “right toes go up (dorsiflex)” |
| `left_hip_roll` | “left leg moves outward (to the left, away from body midline)” |
| `right_hip_roll` | “right leg moves outward (to the right, away from body midline)” |

Important:
- These hints are only labels for the calibration operator. They **must match the MuJoCo model** you trained against.
- If the MJCF’s sign convention differs, update the hint table (or store per-joint hints in config) so the prompt stays unambiguous.

#### How to verify the hint table against MuJoCo (one-time setup)

- In a MuJoCo viewer (or your existing visualization tooling), apply a small +angle to the joint and observe the motion.
- Record the observed “+angle motion” as the `positive_motion_hint` for that joint.
  - After this, the calibrator can ask a purely physical question and still align to the model’s sign.

Notes:
- The script should display the MuJoCo joint axis direction if available (optional enhancement: parse MJCF and print joint axis vector and joint name), but still rely on a human to interpret it physically.
- If the user is unsure, allow repeating the test with larger/smaller deltas.

### Step C: Determine offset at neutral pose

The script should:
1) jog the joint to the user’s chosen “neutral pose” (physical alignment), using keyboard controls:
   - step left/right by configurable `step_units` (e.g., `a`/`d`)
   - optionally bigger steps (`A`/`D`)
2) once the user confirms alignment, read `units_read` from the servo
3) compute offset:

```
offset_units = units_read - units_center
```

Rationale: we want the chosen neutral pose to satisfy `pos_rad == 0` when `units_read == units_center + offset_units`.

### Step D: Persist and verify

Persist:
- update `hiwonder.joint_signs[joint]` and `hiwonder.joint_offsets_rad[joint]` in the JSON config
- write to disk with:
  - an explicit output path option (recommended), OR
  - in-place update with an automatic timestamped backup

Verify (quick checks):
- command `target_rad = 0.0`, read back `pos_rad` and confirm it is “near 0” (within a small tolerance, e.g., `0.02–0.05 rad`)
- optionally command `±0.2 rad` and confirm direction + magnitude look correct

## File Location and Entrypoints

Preferred (packaged) location:
- `runtime/scripts/calibrate.py` (repo-local tooling; does not require the `wr_runtime` package layout)

Alternative (repo-local tooling, not packaged):
- same as above; run via `uv run python runtime/scripts/calibrate.py ...`

Design preference for now: keep it repo-local and directly depend on `runtime.configs` + `runtime.hardware.hiwonder_board_controller`.
We can add a console entrypoint later if desired.

## CLI Sketch

Minimal flags:
- `--config <path>`: input runtime config (default: `runtime/configs/wr_runtime_config.json`)
- `--bundle <path>`: optional policy bundle folder (for `policy_spec.json` / `home_ctrl_rad`)
- `--scene-xml <path>`: optional MuJoCo scene XML (read `"home"` keyframe via MuJoCo)
- `--keyframes-xml <path>`: optional keyframes XML (read `"home"` keyframe by parsing `qpos`)
- `--go-home`: optional; move to exported home pose before calibrating
- `--output <path>`: write updated config here (default: in-place with backup)
- `--joints left_hip_pitch,right_knee_pitch` or `--all`
- `--step-units 5`
- `--delta-rad 0.1` (direction test)
- `--move-ms 300` (jog speed)

Non-interactive mode (optional later):
- `--print-only` (compute but do not write)

## Data Ownership and Integration Points

- Calibration is runtime-specific: it lives in the runtime config and is applied in the hardware adapter layer (`HiwonderBoardActuators`).
- Training and `policy_contract` remain in radians; they should not “know” about servo units, offsets, or wiring direction.

## Validation Checklist (post-calibration)

- `wildrobot-run-policy --config <calibrated_config>` starts without interface errors.
- With a simple “hold neutral pose” action (or a small fixed target pose tool), joints move in the expected direction.
- Logged `signals.joint_pos_rad` near the neutral pose are close to zero for all calibrated joints.

## Future Extensions

- Endstop/range calibration: measure usable min/max units per joint; clamp targets accordingly.
- Per-joint neutral pose presets (e.g., a named “stand pose”) to speed up multi-joint alignment.
- Generate a calibration report artifact (JSON + human-readable summary) for versioning and debugging.
- Optional MJCF parsing to show joint axis and expected positive rotation in a more user-friendly way.

## Open Questions

- What is the canonical “neutral pose” definition for each joint (visual description / photos)?
- Do we want to store calibration per-robot in `~/.wildrobot/config.json`, or keep a separate `~/.wildrobot/calibration.json` included/merged at load time?
- Is servo readback reliable and stable enough to use as the source of truth for offsets, or should we plan for external angle measurement support?
