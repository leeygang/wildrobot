# Joint Angles And Servo Units

WildRobot runtime now keeps the control path simple:

```text
policy action
  -> joint target radians
  -> Hiwonder servo electrical units
```

For the current `wr_obs_v8_cmd3d` walking runtime, policy actions are residuals
around the exported home pose:

```python
target_q = clip(home_q_rad + clip(action, -1, 1) * residual_scale, joint_min, joint_max)
```

The generic legacy action mapping still exists in `policy_contract/*/calib.py`
for old bundles and tests, but new/current robot configs and exported policy
specs do not emit a per-joint action-sign field.

## Joint Radians To Servo Units

Runtime hardware calibration is independent of policy action semantics.  Each
servo config maps a MuJoCo/control joint angle in radians to Hiwonder electrical
units:

```python
center_rad = deg2rad(motor_center_mujoco_deg)
u_elec = clamp_0_1000(
    500 + offset_unit + motor_sign * (target_rad - center_rad) * units_per_rad
)
```

Inverse readback:

```python
target_rad = center_rad + motor_sign * ((u_elec - 500 - offset_unit) / units_per_rad)
```

Where:

- `target_rad` is the MuJoCo/control joint coordinate in radians.
- `u_elec` is the raw Hiwonder servo unit in `[0, 1000]`.
- `offset_unit` shifts the physical neutral around servo center `500`.
- `motor_sign` is hardware direction: `+1` means increasing joint radians
  increases servo units, `-1` means increasing joint radians decreases servo
  units.
- `motor_center_mujoco_deg` is the joint angle that corresponds to conceptual
  servo center `500` before `offset_unit`.

## Calibration Rule

When calibrating hardware, reason only in joint radians/degrees and servo units.
Do not reason in policy action space.

For a joint:

- Command raw units near center, for example `500 -> 530`.
- Observe whether the physical joint moves in the positive MuJoCo/control
  direction printed by the calibrator.
- If increasing units increases joint radians, use `motor_sign = +1`.
- If increasing units decreases joint radians, use `motor_sign = -1`.

After direction is correct, calibrate `offset_unit` so the chosen physical
neutral pose reads as the intended joint angle, usually `0 rad`.
