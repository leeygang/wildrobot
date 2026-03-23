# Joint angle ↔ policy action ↔ servo units

This doc summarizes how WildRobot converts between:
- **MuJoCo joint targets** (`target_rad`, radians)
- **Policy actions** (`action ∈ [-1, 1]`)
- **Servo position units** (Hiwonder board units `0..1000` at the electrical layer, plus a *conceptual/offset-corrected* unit `u = u_elec - offset_unit`)

It is an abstraction of the mapping described in `runtime/docs/servo_calibration_design.md`, with concrete examples for:
`left_shoulder_roll`, `left_shoulder_pitch`, `left_elbow_pitch`.

## 1) Coordinate / quantity definitions

### A. MuJoCo joint coordinate (`target_rad`)
- `target_rad` is the **MuJoCo hinge joint coordinate** in **radians**.
- Positive direction follows MuJoCo's convention: **right-hand rule around the joint axis** defined in the MJCF.
- In `assets/v2/wildrobot.xml`, these hinge joints are typically `axis="0 0 1"`, i.e. rotation around the joint's **local +Z axis**.
- This is the quantity that:
  - the policy contract calls "position target in radians"
  - the runtime actuator layer takes as input (and converts to hardware units)

### B. Policy action coordinate (`action ∈ [-1, 1]`)
- The policy outputs `action` in `[-1, 1]` per joint (in actuator order).
- `policy_action_sign ∈ {+1, -1}` is a **model/policy convention** (from `assets/v2/mujoco_robot_config.json`) used to flip actions so that "same action on symmetric joints" can mean "symmetric motion".
- `policy_action_sign` is applied **only** in the mapping between `action` and `target_rad` (it is *not* a servo calibration parameter).

### C. Servo electrical units (`u_elec ∈ [0, 1000]`)
- `u_elec` is the **raw unit** you read/write on the Hiwonder servo controller.
- Runtime ultimately sends **electrical** units to hardware.

### D. Servo conceptual (offset-corrected) units (`u ∈ [max(0, -offset_unit), 1000-offset_unit]`)
Calibration uses a conceptual "contract-space" unit `u` that is easier to reason about:
- `offset_unit ∈ ℤ` (per-joint) shifts between electrical and conceptual units:
  - `u = u_elec - offset_unit`
  - `u_elec = u + offset_unit`
- By definition, the conceptual center is `u = 500`.

### E. Servo calibration parameters
These parameters live in `runtime/configs/runtime_config_template.json` under `servo_controller.servos.<joint>`:
- `motor_sign ∈ {+1, -1}` (hardware sign correction):
  - `+1`: increasing `target_rad` increases `u` / `u_elec`
  - `-1`: increasing `target_rad` decreases `u` / `u_elec`
- `motor_center_mujoco_deg` (degrees):
  - The MuJoCo angle that corresponds to the **conceptual center** `u = 500`.
  - Let `center_rad = deg2rad(motor_center_mujoco_deg)`.

## 2) Conversion: policy action ↔ MuJoCo target radians

### Current mapping (v0.17.3+): home-centered with universal servo span

The home keyframe (`assets/v2/keyframes.xml`) places all joints at ~0 rad.
The servo physical travel is [-120°, +120°] (240° total), giving a universal
half-span of `servo_span = deg2rad(120) = 2.094 rad`.

Constants:
- `servo_span = 2.094` (same for all joints, derived from servo hardware)

Forward (policy → MuJoCo):
- `action_clipped = clip(action, -1, 1)`
- `corrected = action_clipped * policy_action_sign`
- `target_rad = clip(corrected * servo_span, range_min_rad, range_max_rad)`

Inverse (MuJoCo → policy):
- `corrected = target_rad / servo_span`
- `action = clip(corrected * policy_action_sign, -1, 1)`

Properties:
- `action = 0` → `target_rad = 0` → home (standing) pose for all joints.
- `action = ±1` → `target_rad = ±2.094 rad (±120°)`, then clamped to per-joint
  limits. The full joint range is always reachable because no joint range
  exceeds ±120°.
- For joints whose range is much smaller than [-120°, +120°] (e.g.,
  `left_hip_pitch` range [-5°, +85°]), the policy action saturates (clips)
  well before ±1. The policy learns to stay within the useful region.

Why this mapping:
- The old mapping (`ctrl_center + ctrl_span`) placed action=0 at the midpoint
  of each joint's range, which was a crouched/bent pose — not standing.
- Centering on home=0 means the policy starts from a natural standing pose
  and learns corrections from there.
- A single universal span (the servo limit) avoids per-joint span computation
  and ensures full range access.

### Legacy mapping (pre-v0.17.3): range-center + half-span

Used in v0.17.2 and earlier. Documented here for reference.

Given a joint range in radians `[range_min_rad, range_max_rad]`:
- `ctrl_center = (range_min_rad + range_max_rad) / 2`
- `ctrl_span   = (range_max_rad - range_min_rad) / 2`

Forward (policy → MuJoCo):
- `action_clipped = clip(action, -1, 1)`
- `corrected = action_clipped * policy_action_sign`
- `target_rad = corrected * ctrl_span + ctrl_center`

Inverse (MuJoCo → policy):
- `corrected = (target_rad - ctrl_center) / (ctrl_span + 1e-6)`
- `action = clip(corrected * policy_action_sign, -1, 1)`

Problem: `ctrl_center` ≠ home pose for most joints. Action=0 produced a
crouched pose, forcing the policy to learn a non-zero output just to stand.

### Important note on layer boundaries

- `servo_span` and `policy_action_sign` are used **only** in the action↔target_rad mapping.
- `motor_center_mujoco_deg` and `motor_sign` are used **only** in the target_rad↔servo mapping.
- These are independent layers. Changing the action mapping does not affect
  the runtime servo conversion.

## 3) Conversion: MuJoCo target radians ↔ servo units

The runtime assumes a linear servo model:
- Servo travel: `[-120°, +120°]` (240° total)
- Units: `[0, 1000]`, center at `500`
- `units_per_rad = 1000 / deg2rad(240) ≈ 238.732`

Forward (MuJoCo → hardware command):
- `center_rad = deg2rad(motor_center_mujoco_deg)`
- `u_elec = clamp_0_1000(500 + offset_unit + motor_sign * (target_rad - center_rad) * units_per_rad)`

Inverse (hardware readback → MuJoCo):
- `target_rad = center_rad + motor_sign * ((u_elec - 500 - offset_unit) / units_per_rad)`

Conceptual units (optional, for debugging math):
- `u = u_elec - offset_unit`
- So `u = 500 + motor_sign * (target_rad - center_rad) * units_per_rad`

This layer is unchanged by the v0.17.3 action mapping update.

## 4) How the signs compose (common source of confusion)

There are two independent "sign flips", applied at different layers:
- `policy_action_sign`: flips **action → target_rad** (model/policy convention)
- `motor_sign`: flips **target_rad → servo units** (hardware installation/calibration)

If you want to know whether "+action increases electrical units" for a joint, look at:
- `effective_unit_direction ≈ policy_action_sign * motor_sign`

Since the action mapping no longer uses `ctrl_center`, and the servo mapping
uses a separate `motor_center_mujoco_deg`, the two layers are fully
independent. Reason about each layer separately.

## 5) Concrete numerical examples (forward + inverse)

The numbers below are taken from:
- Joint ranges + `policy_action_sign`: `assets/v2/mujoco_robot_config.json` (unless noted)
- Servo calibration: `runtime/configs/runtime_config_template.json` (unless noted)

Assume the policy outputs the same action for each example:
- `action = +0.25`

All unit outputs are shown as the unclamped float and the commanded integer (runtime rounds after clipping).
Inverse examples start from the integer `u_elec` (what hardware reads back), so they may differ slightly from the original action due to rounding.

### Example A — `left_shoulder_roll`
Config:
- Joint range (deg): `[-175, +5]` and `policy_action_sign = +1` (custom example)
- Servo calibration: `motor_center_mujoco_deg = -85`, `motor_sign = -1`, `offset_unit = 0` (custom example)

Step 1 (action → target):
- `servo_span = 120°`
- `corrected = +0.25 * (+1) = +0.25`
- `target = clip(0.25 * 120°, -175°, +5°) = 30°` (`target_rad ≈ 0.5236`)

Step 2 (target → units):
- `center = -85°`
- `u_elec ≈ 500 + 0 + (-1) * (30° - (-85°)) * (1000/240°)`
- `u_elec ≈ 500 - 479.17 ≈ 20.83` → command `21`

Step 3 (units → action):
- `target ≈ -85° + (-1) * ((21 - 500 - 0) * (240° / 1000)) = -85° + 114.96° = 29.96°`
- `corrected ≈ 29.96° / 120° ≈ 0.2497`
- `action ≈ 0.2497 * (+1) ≈ 0.2497`

### Example B — `left_shoulder_pitch`
Config:
- Joint range (deg): `[-30, +180]` and `policy_action_sign = +1`
- Servo calibration: `motor_center_mujoco_deg = +90`, `motor_sign = -1`, `offset_unit = +14`

Step 1 (action → target):
- `corrected = +0.25`
- `target = clip(0.25 * 120°, -30°, +180°) = 30°` (`target_rad ≈ 0.5236`)

Step 2 (target → units):
- `center = 90°`
- `u_elec ≈ 500 + 14 + (-1) * (30° - 90°) * (1000/240°)`
- `u_elec ≈ 514 + 250 = 764` → command `764`

Step 3 (units → action):
- `target ≈ 90° + (-1) * ((764 - 500 - 14) * (240° / 1000)) = 90° - 60° = 30°`
- `corrected ≈ 30° / 120° ≈ 0.25`
- `action ≈ 0.25 * (+1) ≈ 0.25`

### Example C — `left_elbow_pitch`
Config:
- Joint range (deg): `[0, +90]` and `policy_action_sign = +1`
- Servo calibration: `motor_center_mujoco_deg = 0`, `motor_sign = -1`, `offset_unit = +4`

Step 1 (action → target):
- `target = clip(0.25 * 120°, 0°, +90°) = 30°` (`target_rad ≈ 0.5236`)

Step 2 (target → units):
- `u_elec ≈ 500 + 4 + (-1) * (30° - 0°) * (1000/240°)`
- `u_elec ≈ 504 - 125 = 379` → command `379`

Step 3 (units → action):
- `target ≈ 0° + (-1) * ((379 - 500 - 4) * (240° / 1000)) = 30°`
- `corrected ≈ 30° / 120° ≈ 0.25`
- `action ≈ 0.25 * (+1) ≈ 0.25`

## 6) APIs for doing these conversions

### A. Policy action ↔ MuJoCo radians (training CAL layer)
- `training/cal/cal.py`
  - `policy_action_to_ctrl(action) -> target_rad` — applies `policy_action_sign`, scales by `servo_span`, clips to joint range
  - `ctrl_to_policy_action(ctrl) -> action` — inverse

### B. Policy action ↔ MuJoCo radians (runtime / inference)
- `runtime/configs/config.py` (`ServoControllerConfig.policy_action_to_servo_cmd`)
  - End-to-end: policy action list → `(servo_id, pos)` commands

### C. MuJoCo radians ↔ servo units (runtime actuator-level)
Vectorized (the runtime hardware path):
- `runtime/wr_runtime/hardware/actuators.py`
  - `joint_target_rad_to_servo_pos_elec_units(targets_rad, offsets_unit, motor_signs, centers_rad, servo_model) -> units`
  - `servo_pos_elect_units_to_joint_target_rad(units, offsets_unit, motor_signs, centers_rad, servo_model) -> pos_rad`

Scalar/per-joint (handy for calibration tooling and debugging):
- `runtime/configs/config.py` (`ServoConfig`)
  - `joint_target_rad_to_elect_unit(target_rad) -> int`
  - `servo_elect_units_to_joint_target_rad(units) -> float`

## 7) E2E flow summary

```
Policy NN (tanh)
    │
    ▼
action ∈ [-1, +1]
    │
    │  corrected = action * policy_action_sign
    │  target_rad = clip(corrected * 2.094, range_min, range_max)
    │
    ▼
target_rad (MuJoCo joint angle, radians)
    │
    │  u_elec = clamp(500 + offset + motor_sign * (target_rad - center_rad) * 238.732)
    │
    ▼
u_elec ∈ [0, 1000] (servo command)
```

Two conversions. Two independent sign parameters. No shared center constants
between the layers.
