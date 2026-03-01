# Servo Calibration Design (Runtime)

## Context

WildRobot hardware runtime lives under `runtime/wr_runtime/` and assumes:
- Hiwonder servo position units are in `[0..1000]`
- servo electrical center is `units_center = 500`
- mapping is linear over `[-120°, +120°]` (240° total range)
- runtime commands and reads joints in radians, using a per-joint calibration defined in the runtime JSON config

Calibration is performed with the interactive script:
- `runtime/scripts/calibrate.py`

Calibration data is stored in the runtime JSON config (copy the sample `runtime/configs/wr_runtime_config.json` to your robot, typically `~/.wildrobot/config.json`).

This doc describes the servo calibration model and the operator workflow for per-joint:
1) motor_sign correction
2) center offset in servo units
3) optional joint center angle (`motor_center_mujoco_deg`) used in the rad↔units mapping

## Goals

- Calibrate each servo’s **zero/center offset** so that a chosen physical “neutral pose” corresponds to `joint_pos_rad == 0.0` in runtime.
- Calibrate each servo’s **motor sign** so that “positive radians” in runtime matches “positive joint angle” in the MuJoCo model used for training/inference.
- Persist calibration results into the runtime JSON config in a way that is robust, debuggable, and easy to edit by hand.
- Keep the runtime API boundary in radians (do not leak servo units into policy/contract code).

## Non-goals (for initial version)

- Endstop/range calibration (finding actual mechanical min/max).
- Torque/current calibration or PID tuning.
- IMU calibration.
- Automatically inferring MuJoCo “positive direction” physically (we will require human confirmation).

## Terminology

- `target_rad`: the *MuJoCo joint angle target* in radians.
  - This is the value the runtime ultimately wants the joint to take in the MuJoCo model’s joint coordinate (the same coordinate used for joint `range_min_rad`/`range_max_rad`).
  - In the runtime code paths, `target_rad` is the input to the rad↔units conversion.
  - If you start from a policy action, then conceptually:
    - `corrected_action = action * policy_action_sign`
    - `target_rad = corrected_action * (0.5 * (range_max_rad - range_min_rad)) + (0.5 * (range_min_rad + range_max_rad))`
  - `target_rad` is **not** a servo-unit quantity; it is always in radians.

- `motor_center_mujoco_deg` (config field): MuJoCo joint angle in degrees that corresponds to the *contract-space* center `servo_unit == 500`.
  - In radians, this same “motor center” angle is `deg2rad(motor_center_mujoco_deg)`.
  - Because `offset_unit` relates `servo_unit` and `servo_electrical_unit` as:
    - `servo_electrical_unit = servo_unit + offset_unit`
    - `servo_unit = servo_electrical_unit - offset_unit`
    the statement “`target_rad == deg2rad(motor_center_mujoco_deg)` maps to `servo_unit == 500`” is equivalent to:
    - “`target_rad == deg2rad(motor_center_mujoco_deg)` maps to `servo_electrical_unit == 500 + offset_unit`”.
  - Concrete example:
    - Suppose `motor_center_mujoco_deg = 45`.
    - If `offset_unit = +10`, then commanding `target_rad = deg2rad(45°)` yields:
      - `servo_unit_cmd ≈ 500`
      - `servo_electrical_unit_cmd ≈ 510`.

- `servo_electrical_unit` (a.k.a. `u_elec`): the raw Hiwonder board servo position in `[0..1000]`.
  - This is what you directly read/write over the bus.
- `servo_unit` (a.k.a. `u`): a *conceptual* servo position in `[0..1000]` after applying the per-joint unit offset.
  - This is the “contract-space” unit that is easiest to reason about for calibration math.
- `servo_unit_offset` (config field `offset_unit`): the per-joint integer stored at `servo_controller.servos.<joint>.offset_unit`.
  - Relationship:
    - `servo_unit = servo_electrical_unit - servo_unit_offset`
    - `servo_electrical_unit = servo_unit + servo_unit_offset`
- Example (matches current implementation):
  - If the joint needs “+10 units” added at the electrical layer to make the *contract* center line up, set `servo_unit_offset = +10`.
  - Then `servo_unit == 500` corresponds to `servo_electrical_unit == 510`.
    - Readback: if the board reports `servo_electrical_unit_read = 510`, the runtime interprets `servo_unit_read = 510 - 10 = 500`.
    - Command: if the runtime wants to command `servo_unit_cmd = 500`, it sends `servo_electrical_unit_cmd = 500 + 10 = 510`.
- `servo_unit_center`: the *contract-space* center in units (`servo_unit == 500`).
- `motor_sign` (config field) / `s` (math symbol): per-joint sign in `{+1, -1}` stored at `servo_controller.servos.<joint>.motor_sign`.
  - Meaning: it decides whether “positive MuJoCo/control radians” corresponds to increasing or decreasing servo units.
  - In the runtime conversion (see equations below), it multiplies the angle delta:
    - `servo_electrical_unit_cmd` changes by `motor_sign * (+Δrad) * (1000 / range_rad)`.
  - Intuition:
    - `motor_sign = +1`: increasing `target_rad` increases `servo_electrical_unit_cmd` (and also increases `servo_unit_cmd`).
    - `motor_sign = -1`: increasing `target_rad` decreases `servo_electrical_unit_cmd` (flips the joint).
  - Tiny example (assuming `motor_center_mujoco_deg=0` and `offset_unit=0`):
    - If `motor_sign=+1`, then `target_rad=+0.1` commands a value **greater than** 500.
    - If `motor_sign=-1`, then `target_rad=+0.1` commands a value **less than** 500.
  - Concrete example (hip roll, `+Δrad` test):
    - Suppose the MuJoCo convention for `left_hip_roll` is: `+Δrad` means “left leg moves outward (to the left)”.
    - During calibration, pick a small delta like `delta_rad = 0.10` and command two targets around center:
      - `target_rad_plus  = deg2rad(motor_center_mujoco_deg) + 0.10`
      - `target_rad_minus = deg2rad(motor_center_mujoco_deg) - 0.10`
    - Observe the servo’s *electrical units* while applying those two commands:
      - If `target_rad_plus` yields a **higher** `servo_electrical_unit_cmd` than `target_rad_minus` and the joint moves outward as expected, set `motor_sign = +1`.
      - If `target_rad_plus` yields a **lower** `servo_electrical_unit_cmd` than `target_rad_minus` (and/or the joint moves inward for `+Δrad`), set `motor_sign = -1`.
    - This is exactly what the mapping enforces:
      - `servo_electrical_unit_cmd(target_rad_plus) - servo_electrical_unit_cmd(target_rad_minus)` is proportional to `motor_sign * (target_rad_plus - target_rad_minus)`.
- `policy_action_sign` (policy-contract field): per-joint sign in `{+1, -1}` stored in `assets/v2/robot_config.yaml` and typically exported into `policy_contract/policy_spec.json` under `robot.joints.<joint>.policy_action_sign`.
  - Meaning: it decides whether a *positive policy action* should produce a positive or negative *control* (radians) delta for that joint, so that “same action value on left/right” yields symmetric motion.
  - Where it is applied (conceptual):
    - `corrected_action = action * policy_action_sign`
    - `target_rad = corrected_action * (0.5 * (range_max_rad - range_min_rad)) + (0.5 * (range_min_rad + range_max_rad))`
  - What it is *not*: it is **not** part of servo calibration. Servo calibration only uses `motor_sign`, `offset_unit`, and optionally `motor_center_mujoco_deg`.
  - Relationship to `motor_sign`:
    - `policy_action_sign` is a model/policy convention (action-space → control-space).
    - `motor_sign` is a hardware polarity convention (control-space ↔ servo-units).
    - The physical direction you see for a given `+action` depends on both layers.
  - Concrete example (left/right hip roll symmetry):
    - A common convention is:
      - `left_hip_roll.policy_action_sign = +1`
      - `right_hip_roll.policy_action_sign = -1`
    - Pick concrete numbers for a single timestep:
      - Policy outputs the same action value for both hips: `action = +0.30`.
      - Assume (for illustration) the hip-roll joint range midpoint is `0.00 rad` and the half-range is `0.35 rad`.
    - Then the runtime computes a *target control angle* (this doc’s `target_rad`) per joint:
      - Left: `target_rad = (+0.30 * +1) * 0.35 + 0.00 = +0.105 rad`
      - Right: `target_rad = (+0.30 * -1) * 0.35 + 0.00 = -0.105 rad`
    - Those opposite-signed `target_rad` values are what typically creates symmetric left/right roll motion (e.g., “both legs roll outward”), once each joint’s `motor_sign` is calibrated so that `+target_rad` corresponds to the MuJoCo-positive physical direction for that joint.
- `motor_center_mujoco_deg`: MuJoCo angle in degrees that corresponds to `servo_unit == 500`.
  - If `motor_center_mujoco_deg == 0` (most common), then `servo_unit == 500` corresponds to MuJoCo `0°`.

## Two Different “Ranges” (Servo Range vs Joint Range)

There are **two distinct ranges** involved, and they serve different purposes:

1) Servo range (hardware model / scaling)
- This is the servo’s assumed full travel span: `[-120°, +120°]` (240° total) mapped onto `units ∈ [0..1000]`.
- In this doc and the runtime conversion, this is represented by `range_rad`.
- It defines the slope: `(1000 / range_rad)` units per radian.

2) Joint range (MuJoCo/model limit)
- This is the *allowed joint angle range* for a specific joint in the robot model.
- It is defined in `assets/v2/robot_config.yaml` (and/or can be parsed from MJCF joint `range=` depending on the tool).
- In code (e.g. the calibrator), this typically appears as `min_rad, max_rad` for each joint.

How they interact:
- The **conversion math** uses the *servo range* (`range_rad`) to turn angle deltas into unit deltas.
- The **controller / policy target** should respect the *joint range* (min/max radians) to avoid commanding angles outside what the model (and likely the mechanism) intends.
- Even if the joint range is respected, your current `(offset_unit, motor_center_mujoco_deg, motor_sign)` can still push commands beyond servo unit limits; the final `servo_electrical_unit_cmd` is clamped to `[0..1000]`.
  - This clamp is a safety behavior, but it also means “requested radian targets” may clip and become unreachable at the hardware layer.

Important nuance about “`500` is center”:
- `500` is the center of the **contract-space** unit (`servo_unit`).
- Electrical center depends on offset: `servo_electrical_unit == 500 + servo_unit_offset`.
- In **radians**, `servo_unit == 500` corresponds to `target_rad == deg2rad(motor_center_mujoco_deg)`.
- Therefore, “`500` corresponds to `target_rad == 0`” is only true when `motor_center_mujoco_deg == 0`.
- Separately, the **joint-range midpoint** is `(min_rad + max_rad) / 2`, and it generally does *not* equal `deg2rad(motor_center_mujoco_deg)`.

Concrete equivalences from the command mapping (see below):
- Commanding `target_rad = deg2rad(motor_center_mujoco_deg)` yields `servo_unit_cmd ≈ 500`.
- If `servo_unit_offset == 0`, the same command yields `servo_electrical_unit_cmd ≈ 500`.
- If `motor_center_mujoco_deg == 0`, then commanding `target_rad = 0` yields `servo_unit_cmd ≈ 500`.

## Runtime Mapping Model

The runtime uses the mapping implemented in `runtime/wr_runtime/hardware/actuators.py`.

Per joint, calibration values are:
- `motor_sign ∈ {+1, -1}`
- `offset_unit ∈ ℤ` (a.k.a. `servo_unit_offset`)
- `motor_center_mujoco_deg` (optional; default `0`)

## Config Shape

Use `runtime/configs/wr_runtime_config.json` as the canonical “in-repo” calibration target:
- calibration reads joint names (and servo IDs) from this file
- calibration writes results back to this file (in-place with backups, unless `--output` is provided)

### Config shape (recommended, minimal change)

Under `servo_controller.servos.<joint>`:

```json
{
  "servo_controller": {
    "servos": {
      "left_hip_pitch": { "id": 1, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 }
    }
  }
}
```

Where:
- `offset_unit` is an integer in servo units (default: `0`)
- `motor_sign` is `+1` or `-1` (default: `+1`)
- `motor_center_mujoco_deg` is the MuJoCo angle (deg) that maps to `servo_unit == 500` (default: `0`)

Defaults:
- If `motor_sign` is missing, runtime assumes `+1`.
- If `offset_unit` is missing, runtime assumes `0`.
- If `motor_center_mujoco_deg` is missing, runtime assumes `0`.

## Mapping With Offset + Motor Sign + Center

For each joint:
- `motor_sign` from JSON in `{+1, -1}`
- `offset_unit` (aka `servo_unit_offset`) from JSON (int)

Note:
- `policy_action_sign` is **not** part of the servo calibration mapping. It is only used when mapping policy actions → control targets.

Command (radians → *electrical* units):
```
servo_electrical_unit_cmd = clamp_0_1000(
  units_center
  + servo_unit_offset
  + motor_sign * (target_rad - deg2rad(motor_center_mujoco_deg)) * (1000 / range_rad)
)
```

### Worked Example: Hip Roll (Left + Right)

This example shows the full chain:
`action` → `target_rad` (policy convention) → `servo_electrical_unit_cmd` (hardware command).

Assumptions (taken from typical v2 joint ranges):
- `left_hip_roll` range: `range_min_rad = -1.570796` (-90°), `range_max_rad = +0.174533` (+10°)
- `right_hip_roll` range: `range_min_rad = -0.174533` (-10°), `range_max_rad = +1.570796` (+90°)
- `policy_action_sign(left_hip_roll) = +1`, `policy_action_sign(right_hip_roll) = -1`

Step 1 — action → target_rad

Use:
- `half_range = 0.5 * (range_max_rad - range_min_rad)`
- `mid_range  = 0.5 * (range_min_rad + range_max_rad)`
- `corrected_action = action * policy_action_sign`
- `target_rad = corrected_action * half_range + mid_range`

Pick a concrete policy output: `action = +0.30`.

- Left hip roll:
  - `half_range = 0.5 * (0.174533 - (-1.570796)) = 0.872665`
  - `mid_range  = 0.5 * (-1.570796 + 0.174533) = -0.698132`
  - `corrected_action = +0.30 * (+1) = +0.30`
  - `target_rad = (+0.30)*0.872665 + (-0.698132) = -0.436332` (≈ -25°)

- Right hip roll:
  - `half_range = 0.872665`
  - `mid_range  = 0.5 * (-0.174533 + 1.570796) = +0.698132`
  - `corrected_action = +0.30 * (-1) = -0.30`
  - `target_rad = (-0.30)*0.872665 + (0.698132) = +0.436332` (≈ +25°)

These opposite-signed `target_rad` values are what make “same action on left/right” produce symmetric hip-roll motion.

Step 2 — target_rad → servo_electrical_unit_cmd

Now pick an illustrative hardware calibration (these numbers match the *shape* of runtime configs; your robot’s values may differ):
- `units_center = 500`
- `range_rad = 4.188790` (240°)
- `motor_center_mujoco_deg = 0` (so `deg2rad(...) = 0`)
- `left_hip_roll: offset_unit = +20, motor_sign = -1`
- `right_hip_roll: offset_unit = +21, motor_sign = -1`

Compute the scale once:
- `units_per_rad = 1000 / range_rad ≈ 238.732`

- Left hip roll:
  - `delta_rad = target_rad - 0 = -0.436332`
  - `delta_units = motor_sign * delta_rad * units_per_rad = (-1)*(-0.436332)*238.732 ≈ +104.1`
  - `servo_electrical_unit_cmd ≈ 500 + 20 + 104.1 ≈ 624`

- Right hip roll:
  - `delta_rad = +0.436332`
  - `delta_units = (-1)*(+0.436332)*238.732 ≈ -104.1`
  - `servo_electrical_unit_cmd ≈ 500 + 21 - 104.1 ≈ 417`

Finally, clamp to `[0..1000]` (not needed for these values).

### Worked Example: Shoulder Roll (Non-zero `motor_center_mujoco_deg`)

This example highlights the effect of a non-zero `motor_center_mujoco_deg` in:
`(target_rad - deg2rad(motor_center_mujoco_deg))`.

Assumptions (illustrative):
- `left_shoulder_roll` range: `range_min_rad = -1.221730` (-70°), `range_max_rad = +1.221730` (+70°)
- `policy_action_sign(left_shoulder_roll) = +1`

Step 1 — action → target_rad

Pick `action = +0.25`.

- `half_range = 0.5 * (1.221730 - (-1.221730)) = 1.221730`
- `mid_range  = 0.5 * (-1.221730 + 1.221730) = 0`
- `corrected_action = +0.25 * (+1) = +0.25`
- `target_rad = (+0.25)*1.221730 + 0 = 0.305433` (≈ +17.5°)

Step 2 — target_rad → servo_electrical_unit_cmd

Assume this joint’s servo calibration is (matching the common `left_shoulder_roll` case in `runtime/configs/wr_runtime_config.json`):
- `units_center = 500`
- `range_rad = 4.188790` (240°)
- `motor_center_mujoco_deg = -85` (so `deg2rad(...) ≈ -1.483530`)
- `offset_unit = -30`
- `motor_sign = -1`

Compute the scale:
- `units_per_rad = 1000 / 4.188790 ≈ 238.732`

Now compute the command:
- `delta_rad = target_rad - deg2rad(motor_center_mujoco_deg) = 0.305433 - (-1.483530) = 1.7890`
- `delta_units = motor_sign * delta_rad * units_per_rad = (-1)*1.7890*238.732 ≈ -427.1`
- `servo_electrical_unit_cmd ≈ 500 + (-30) + (-427.1) ≈ 43`

Sanity check (what “center” means):
- If `target_rad == deg2rad(motor_center_mujoco_deg)` (here, `≈ -1.483530`), then `delta_rad = 0` and
  `servo_electrical_unit_cmd == units_center + offset_unit == 500 - 30 == 470`.

Equivalently, in contract-space units (subtracting the offset):
```
servo_unit_cmd = servo_electrical_unit_cmd - servo_unit_offset
             ≈ units_center + motor_sign * (target_rad - deg2rad(motor_center_mujoco_deg)) * (1000 / range_rad)
```

Readback (electrical units → radians):
```
servo_unit_read = servo_electrical_unit_read - servo_unit_offset
pos_rad = deg2rad(motor_center_mujoco_deg) + motor_sign * (servo_unit_read - units_center) * (range_rad / 1000)
```

This preserves the meaning of `offset_unit` as “what runtime adds to electrical units after the centered conversion”, while allowing sign flips and an optional shift of the rad↔units center.

### Computing `servo_unit_offset` in the presence of `motor_center_mujoco_deg`

If you want a particular reference joint angle `target_ref_rad` to match a measured readback `servo_electrical_unit_read`, rearrange the command equation:

$$
servo\_unit\_offset = servo\_electrical\_unit\_read - units\_center - motor\_sign\cdot (target\_ref\_rad - center\_rad)\cdot (1000 / range\_rad)
$$

Two common special cases:
- If `motor_center_mujoco_deg == 0` and you want `target_ref_rad == 0` at the neutral pose, this reduces to `servo_unit_offset = servo_electrical_unit_read - 500`.
- If you choose `motor_center_mujoco_deg` such that `target_ref_rad == deg2rad(motor_center_mujoco_deg)` for your neutral pose, this reduces to `servo_unit_offset = servo_electrical_unit_read - 500`.

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

Important: calibrate `motor_sign` *before* `offset_unit`, because the “motor_sign test” is easier to reason about when the joint is near center.

## How `policy_action_sign` Impacts Calibration (and What It Does *Not* Change)

`policy_action_sign` comes from `assets/v2/robot_config.yaml` (and is typically exported into the policy bundle/spec). It is a **policy/model convention** for mapping *policy actions* into *control targets*.

Conceptually:
- Policy produces `action ∈ [-1, 1]`.
- Runtime computes a control target in radians:
  - `corrected = action * policy_action_sign`
  - `target_rad = corrected * (0.5 * (range_max_rad - range_min_rad)) + (0.5 * (range_min_rad + range_max_rad))`

Key implications:
- `policy_action_sign` **does not appear** in the servo calibration mapping (`motor_sign`, `offset_unit`, `motor_center_mujoco_deg`).
- `policy_action_sign` **does affect your intuition** for what “positive action” does on mirrored joints.

Practical guidance:
- When calibrating `motor_sign`, always interpret prompts in terms of **positive MuJoCo/control radians** (`target_rad` increasing), not in terms of “positive policy action”.
- When calibrating `offset_unit` and `motor_center_mujoco_deg`, you are aligning the servo unit readback to a chosen **reference control angle** (`target_ref_rad`). This choice is independent of `policy_action_sign`.

Example:
- Suppose `right_hip_pitch` has `policy_action_sign = -1`.
  - A *positive* policy action produces a *negative* `target_rad` delta around the joint-range midpoint.
  - But the calibration question “did `+delta_rad` move toward the positive-motion hint?” is still about `+delta_rad` in **control radians**, regardless of `policy_action_sign`.

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
- CLI: `--scene-xml <path>` (MuJoCo scene XML; e.g. `assets/v2/scene_flat_terrain.xml`)
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
- Convert `home_ctrl_rad` (absolute joint targets in radians) to servo commands using the mapping above (using `motor_sign` and `offset_unit`).
- Use a conservative move time (e.g., 500–1000 ms) and allow user confirmation before moving.

### Step A: Select joint

Inputs:
- config path (reads servo IDs, existing offsets/signs)
- list of joints to calibrate (default: config’s joint order / MJCF actuator order)

For each joint, print:
- joint name
- servo ID
- current `sign` and `offset_unit` (legacy `offset_rad` is accepted but interpreted as the same servo-unit value)

### Step B: Determine motor sign

The script should:
1) move the joint near center (safe) using `servo_electrical_unit ≈ 500 + offset_unit` (or defaults)
2) prompt the user to run a motor_sign test:
   - apply a small positive delta (e.g., `+0.1 rad` or `+20 units`)
   - apply a small negative delta
3) ask the user an *explicit physical-direction question*, not “MuJoCo-positive”.

#### Why this matters

“MuJoCo-positive direction” is hard to answer on hardware unless you already know the model’s sign convention for that joint. Humans answer much more reliably when the prompt is phrased as a concrete motion: forward/backward, left/right, toes up/down, etc.

#### Recommended prompt format

For each joint, define a short `positive_motion_hint` string (see the table below), then prompt:

> I will command `+delta`. Did the joint move **toward**: `<positive_motion_hint>`?
>
> - `y` = yes (set `motor_sign = +1`)
> - `n` = no  (set `motor_sign = -1`)
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
2) once the user confirms alignment, read `servo_electrical_unit_read` from the servo
3) compute offset:

```
servo_unit_offset = servo_electrical_unit_read - units_center - motor_sign * (target_ref_rad - deg2rad(motor_center_mujoco_deg)) * (1000 / range_rad)
```

Rationale: we want the chosen reference pose (at `target_ref_rad`) to be consistent with the mapping, so that commanding `target_ref_rad` returns to the same physical alignment.

If you specifically want the neutral pose to correspond to `target_ref_rad == 0`, use `target_ref_rad = 0` in the formula.

### Step D: Persist and verify

Persist:
- update `hiwonder.joint_signs[joint]` and `hiwonder.joint_offsets_rad[joint]` in the JSON config
- write to disk with:
  - an explicit output path option (recommended), OR
  - in-place update with an automatic timestamped backup

Verify (quick checks):
- command `target_rad = 0.0`, read back `pos_rad` and confirm it is “near 0” (within a small tolerance, e.g., `0.02–0.05 rad`)
- optionally command `±0.2 rad` and confirm motor_sign + magnitude look correct

## File Location and Entrypoints

Preferred (packaged) location:
- `runtime/scripts/calibrate.py` (repo-local tooling; does not require the `wr_runtime` package layout)

Alternative (repo-local tooling, not packaged):
- same as above; run via `uv run python runtime/scripts/calibrate.py ...`

Design preference for now: keep it repo-local and directly depend on `configs` + `wr_runtime.hardware.hiwonder_board_controller`.
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
- `--delta-rad 0.1` (motor_sign test)
- `--move-ms 300` (jog speed)

Non-interactive mode (optional later):
- `--print-only` (compute but do not write)

## Data Ownership and Integration Points

- Calibration is runtime-specific: it lives in the runtime config and is applied in the hardware adapter layer (`HiwonderBoardActuators`).
- Training and `policy_contract` remain in radians; they should not “know” about servo units, offsets, or wiring motor_sign.

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
