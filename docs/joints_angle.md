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
- Positive direction follows MuJoCo’s convention: **right-hand rule around the joint axis** defined in the MJCF.
- In `assets/v2/wildrobot.xml`, these hinge joints are typically `axis="0 0 1"`, i.e. rotation around the joint’s **local +Z axis**.
- This is the quantity that:
  - the policy contract calls “position target in radians”
  - the runtime actuator layer takes as input (and converts to hardware units)

### B. Policy action coordinate (`action ∈ [-1, 1]`)
- The policy outputs `action` in `[-1, 1]` per joint (in actuator order).
- `policy_action_sign ∈ {+1, -1}` is a **model/policy convention** (from `assets/v2/robot_config.yaml`) used to flip actions so that “same action on symmetric joints” can mean “symmetric motion”.
- `policy_action_sign` is applied **only** in the mapping between `action` and `target_rad` (it is *not* a servo calibration parameter).

### C. Servo electrical units (`u_elec ∈ [0, 1000]`)
- `u_elec` is the **raw unit** you read/write on the Hiwonder servo controller.
- Runtime ultimately sends **electrical** units to hardware.

### D. Servo conceptual (offset-corrected) units (`u ∈ [max(0, -offset_unit), 1000-offset_unit]`)
Calibration uses a conceptual “contract-space” unit `u` that is easier to reason about:
- `offset_unit ∈ ℤ` (per-joint) shifts between electrical and conceptual units:
  - `u = u_elec - offset_unit`
  - `u_elec = u + offset_unit`
- By definition, the conceptual center is `u = 500`.

### E. Servo calibration parameters
These parameters live in `runtime/configs/wr_runtime_config.json` under `servo_controller.servos.<joint>`:
- `motor_sign ∈ {+1, -1}` (hardware sign correction):
  - `+1`: increasing `target_rad` increases `u` / `u_elec`
  - `-1`: increasing `target_rad` decreases `u` / `u_elec`
- `motor_center_mujoco_deg` (degrees):
  - The MuJoCo angle that corresponds to the **conceptual center** `u = 500`.
  - Let `center_rad = deg2rad(motor_center_mujoco_deg)`.

## 2) Conversion: policy action ↔ MuJoCo target radians

WildRobot uses a “range-center + half-span” mapping (policy contract id `pos_target_rad_v1`).

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

Important nuance:
- `ctrl_center` / `ctrl_span` come from the **joint’s modeled range** (from `assets/v2/robot_config.yaml` and exported into the policy spec).
- `motor_center_mujoco_deg` is a **separate** center used only for the **rad ↔ servo** mapping.

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

## 4) How the signs compose (common source of confusion)

There are two independent “sign flips”, applied at different layers:
- `policy_action_sign`: flips **action → target_rad** (model/policy convention)
- `motor_sign`: flips **target_rad → servo units** (hardware installation/calibration)

If you want to know whether “+action increases electrical units” for a joint, look at:
- `effective_unit_direction ≈ policy_action_sign * motor_sign`

But `motor_center_mujoco_deg` and `ctrl_center` are different centers, so always reason in terms of the full equations when debugging absolute targets.

## 5) Concrete numerical examples (forward + inverse)

The numbers below are taken from:
- Joint ranges + `policy_action_sign`: `assets/v2/robot_config.yaml` (unless noted)
- Servo calibration: `runtime/configs/wr_runtime_config.json` (unless noted)

Assume the policy outputs the same action for each example:
- `action = +0.25`

All unit outputs are shown as the unclamped float and the commanded integer (runtime rounds after clipping).
Inverse examples start from the integer `u_elec` (what hardware reads back), so they may differ slightly from the original action due to rounding.

### Example A — `left_shoulder_roll`
Config:
- Joint range (deg): `[-175, +5]` and `policy_action_sign = +1` (custom example)
- Servo calibration: `motor_center_mujoco_deg = -85`, `motor_sign = -1`, `offset_unit = 0` (custom example)

Interpretation (physical framing, if you use it this way):
- With `range_max = +5°`, MuJoCo `0°` sits `5°` away from that “midline/inward limit” toward the left.

Step 1 (action → target):
- `ctrl_center = (-175 + 5)/2 = -85°`
- `ctrl_span   = (5 - (-175))/2 = 90°`
- `corrected = +0.25 * (+1) = +0.25`
- `target = -85° + 0.25 * 90° = -62.5°` (`target_rad ≈ -1.0908`)

Step 2 (target → units):
- `center = -85°`
- `u_elec ≈ 500 + 0 + (-1) * (-62.5° - (-85°)) * (1000/240°)`
- `u_elec ≈ 406.25` → command `406`

Sanity check:
- When `target == -85°`, `u_elec == 500 + offset_unit == 500`.

Step 3 (units → action):
- `target ≈ -85° + (-1) * ((406 - 500 - 0) * (240° / 1000)) = -62.44°`
- `corrected ≈ (target - ctrl_center) / ctrl_span = (-62.44° - (-85°)) / 90° ≈ 0.2507`
- `action ≈ corrected * policy_action_sign = 0.2507 * (+1) ≈ 0.2507`

### Example B — `left_shoulder_pitch`
Config:
- Joint range (deg): `[-30, +180]` and `policy_action_sign = +1`
- Servo calibration: `motor_center_mujoco_deg = +90`, `motor_sign = -1`, `offset_unit = +14`

Step 1 (action → target):
- `ctrl_center = (-30 + 180)/2 = 75°`
- `ctrl_span   = (180 - (-30))/2 = 105°`
- `corrected = +0.25`
- `target = 75° + 0.25 * 105° = 101.25°` (`target_rad ≈ 1.7671`)

Step 2 (target → units):
- `center = 90°`
- `u_elec ≈ 500 + 14 + (-1) * (101.25° - 90°) * (1000/240°)`
- `u_elec ≈ 467.13` → command `467`

Step 3 (units → action):
- `target ≈ 90° + (-1) * ((467 - 500 - 14) * (240° / 1000)) = 101.28°`
- `corrected ≈ (101.28° - 75°) / 105° ≈ 0.2503`
- `action ≈ 0.2503 * (+1) ≈ 0.2503`

### Example C — `left_elbow_pitch`
Config:
- Joint range (deg): `[0, +90]` and `policy_action_sign = +1`
- Servo calibration: `motor_center_mujoco_deg = 0`, `motor_sign = -1`, `offset_unit = +4`

Step 1 (action → target):
- `ctrl_center = 45°`
- `ctrl_span   = 45°`
- `target = 45° + 0.25 * 45° = 56.25°` (`target_rad ≈ 0.9817`)

Step 2 (target → units):
- `u_elec ≈ 500 + 4 + (-1) * (56.25° - 0°) * (1000/240°)`
- `u_elec ≈ 269.63` → command `270`

Step 3 (units → action):
- `target ≈ 0° + (-1) * ((270 - 500 - 4) * (240° / 1000)) = 56.16°`
- `corrected ≈ (56.16° - 45°) / 45° ≈ 0.2480`
- `action ≈ 0.2480 * (+1) ≈ 0.2480`

## 6) APIs for doing these conversions

### A. Policy action ↔ MuJoCo radians (contract-level)
- `policy_contract/numpy/calib.py`
  - `action_to_joint_target_rad(spec=..., action=...) -> joint_target_rad`
  - `joint_target_rad_to_action(spec=..., joint_target_rad=...) -> action`
- `policy_contract/jax/calib.py` provides the same mapping for JAX arrays.

Notes:
- These functions use per-joint `range_min_rad`, `range_max_rad`, and `policy_action_sign` from the `PolicySpec`.
- They implement the equations in section 2.

### B. MuJoCo radians ↔ servo units (runtime actuator-level)
Vectorized (the runtime hardware path):
- `runtime/wr_runtime/hardware/actuators.py`
  - `joint_target_rad_to_servo_pos_elec_units(targets_rad, offsets_unit, motor_signs, centers_rad, servo_model) -> units`
  - `servo_pos_elect_units_to_joint_target_rad(units, offsets_unit, motor_signs, centers_rad, servo_model) -> pos_rad`

Scalar/per-joint (handy for calibration tooling and debugging):
- `runtime/configs/config.py` (`ServoConfig`)
  - `rad_to_units(target_rad) -> int`
  - `units_to_rad(units) -> float`

End-to-end helper (policy action list → `(servo_id, pos)` commands):
- `runtime/configs/config.py` (`ServoControllerConfig.policy_action_to_servo_cmd`)
