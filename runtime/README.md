# WildRobot Runtime

Lightweight hardware runtime for running WildRobot ONNX policies on Raspberry Ubuntu.

Implementation lives in `runtime/wr_runtime/` (this is what the installed CLIs use).

Breaking change: the legacy modules under `runtime/control`, `runtime/inference`, `runtime/utils`, and `runtime/validation`
were removed. Use the console scripts (`wildrobot-run-policy`, `wildrobot-validate-bundle`, ...) or import from `wr_runtime.*`.

## Latest contract (v0.21.0, `wr_obs_v8_cmd3d`)

The current walking policy is a **home-base residual** policy (smoke9): the
network outputs a residual that is scaled per-joint and added to the home pose,
NOT a direct midpoint-range target.  The runtime composes the control target as

```
filtered = postprocess(raw_action)                    # action_filter_alpha=0 -> identity
applied   = previous filtered action                  # action_delay_steps=1
delta     = clip(applied, -1, 1) * residual_scale_per_joint
target_q  = clip(home_q_rad + delta, joint_min, joint_max)
```

These quantities (residual base/scale/per-joint, action delay/filter, the 3D
`(vx, vy, wz)` command, and the gait phase clock) are NOT in `policy_spec.json`.
They are frozen at export time into **`runtime_policy_config.json`** inside the
bundle (see `training/exports/runtime_metadata.py`).  The runner refuses to fall
back to the generic `pos_target_rad_v1` midpoint mapping for a v8 bundle, and
fails loudly for unsupported obs layouts or residual bases.

`action_scale_rad` in the hardware `wildrobot_config.json` is **ignored** for the
v8 contract — per-joint residual scales come from `runtime_policy_config.json`.

## Install (on robot)

From the `wildrobot/` repo root:

```bash
cd runtime

# Create/update the runtime virtualenv from uv.lock
uv sync
```

Alternative (explicit editable install):
```bash
cd runtime
uv venv
uv pip install -e .
```

Why editable (`-e`) matters: it installs the `wildrobot-*` console scripts while still running
your live checked-out source. If you `git pull` frequently on the device, you usually do *not*
need to reinstall (unless dependencies changed — then rerun `uv sync`).

## Configuration

The runtime reads a JSON config. By default it looks at `~/wildrobot_config.json`.

You can start from the sample at `runtime/configs/runtime_config_template.json` (or `runtime/configs/runtime_config_v2.json`) and copy it to `~/wildrobot_config.json` (or pass `--runtime-config`).

Minimal example:

```json
{
  "mjcf_path": "../assets/v2/wildrobot.xml",
  "policy_onnx_path": "../policies/wildrobot_policy/policy.onnx",
  "realism_profile_path": "../assets/v2/realism_profile_v0.19.1.json",

  "control_hz": 50,
  "action_scale_rad": 0.35,
  "velocity_cmd": 0.0,
  "yaw_rate_cmd": 0.0,

  "servo_controller": {
    "type": "hiwonder",
    "port": "/dev/ttyUSB0",
    "baudrate": 9600,
    "servos": {
      "left_hip_pitch":  { "id": 1, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "left_hip_roll":   { "id": 2, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "left_knee_pitch": { "id": 3, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "left_ankle_pitch": { "id": 4, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "right_hip_pitch": { "id": 5, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "right_hip_roll":  { "id": 6, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "right_knee_pitch":{ "id": 7, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 },
      "right_ankle_pitch": { "id": 8, "offset_unit": 0, "motor_sign": 1, "motor_center_mujoco_deg": 0 }
    }
  },

  "bno085": {
    "i2c_address": "0x4B",
    "upside_down": false,
    "axis_map": ["+X", "+Y", "+Z"]
  },

  "foot_switches": {
    "left_toe": "D5",
    "left_heel": "D6",
    "right_toe": "D13",
    "right_heel": "D19"
  }
}
```

Notes:
- `mjcf_path` is used to discover *actuator/joint order* (the policy action order must match this).
- `policy_onnx_path` should point to `policy.onnx` inside a bundle folder that also contains `policy_spec.json`.
- **Servo IDs do not come from MJCF**. Servo IDs are physical IDs stored on the servos / controller and should live in your runtime config (`servo_controller.servos.<joint>.id`).
- `servo_controller.servos.<joint>.offset_unit` is a per-joint calibration offset in **servo units** around the electrical center (500). Values can be positive or negative. Use the calibration script to write these.
- `servo_controller.servos.<joint>.motor_sign` is a per-joint sign (`+1.0` or `-1.0`) to correct mechanical reversals; if a joint moves the wrong way, flip its sign.
- `servo_controller.servos.<joint>.motor_center_mujoco_deg` (optional, default 0) shifts which MuJoCo angle maps to servo center (500). Most joints can keep this at 0.
- `foot_switches` uses Adafruit Blinka `board` pin names (e.g. `D5`).
- `realism_profile_path` points to versioned digital-twin realism parameters used by SysID/sim2real tooling and validated at runtime startup.

## Run

`wildrobot-run-policy` loads a bundle (`policy.onnx` + `policy_spec.json` +
`runtime_policy_config.json`) and runs the deterministic control loop at
`control_hz`.

Flags:
- `--bundle PATH` (required): bundle directory.
- `--runtime-config PATH`: hardware servo IDs/calibration (required unless `--dry-run`).
- `--dry-run`: run with mock IO (no servos/IMU/footswitches) — for smoke tests
  and safe validation on a dev machine. Never sleeps.
- `--max-steps N` (default 500), `--log-steps N` (default 20, 0=off).
- `--velocity-cmd vx` or `--velocity-cmd vx,vy,wz` (default: bundle
  `default_velocity_cmd`, e.g. `0.13,0,0` for smoke9 straight walk).
- `--no-realtime`: don't sleep to maintain `control_hz` (hardware only).

Dry run (no hardware) — exercises the full read → obs → predict → compose →
write loop:

```bash
cd runtime
uv run wildrobot-run-policy --bundle ../policies/wildrobot_policy_bundle \
  --dry-run --max-steps 5 --velocity-cmd 0.13,0.0,0.0
```

On the robot, with calibrated servo IDs/offsets/directions in a local config
(recommended):

```bash
cd runtime
uv run wildrobot-run-policy --bundle ../policies/wildrobot_policy_bundle \
  --runtime-config ~/.wildrobot/config.json --velocity-cmd 0.13,0,0
```

## Run a bundle from `training/checkpoints/` (on the WildRobot device)

Assumptions:
- You have the `wildrobot/` repo synced on the WildRobot device.
- The bundle directory exists under `training/checkpoints/` (exported via `training/exports/export_policy_bundle_cli.py`).

First, export a bundle from a checkpoint (on a dev machine with the training
env; this builds the gait phase table into `runtime_policy_config.json`):

```bash
uv run python training/exports/export_policy_bundle_cli.py \
  --checkpoint training/checkpoints/<run>/checkpoint_NNNN.pkl \
  --training-config training/checkpoints/<run>/training_config.yaml \
  --bundle-path /tmp/wr_bundle
```

From the repo root on the device:
```bash
cd ~/wildrobot/runtime

# (Recommended) sanity-check Hiwonder serial comms before starting control loop
uv run python scripts/test_hiwonder_board.py --port /dev/ttyUSB0 --baudrate 9600 --servo-ids 1,2,3

# Validate the bundle is self-consistent (policy_spec + ONNX dims + actuator order).
# Works on non-Linux dev machines too (rpi-gpio is gated to Linux).
uv run wildrobot-validate-bundle --bundle /tmp/wr_bundle

# Dry run first (no hardware) to confirm the loop builds obs + composes targets:
uv run wildrobot-run-policy --bundle /tmp/wr_bundle --dry-run --max-steps 5

# Run the control loop on hardware (Ctrl+C to stop; runtime disables actuators on exit)
uv run wildrobot-run-policy --bundle /tmp/wr_bundle --runtime-config ~/.wildrobot/config.json
```

Notes:
- The bundle contains its own `wildrobot_config.json` (often a template) and `wildrobot.xml` (actuator order snapshot).
- If this device/robot has different servo IDs, offsets, IMU axis map, or footswitch pins, prefer passing `--runtime-config ~/.wildrobot/config.json` (or your calibrated JSON) rather than editing the bundle.

The loop runs for `--max-steps` control steps and then exits, disabling
actuators via `robot_io.close()`.  Ctrl+C also stops it and unloads servos.

If you see `Servo position response missing or incomplete`:
- Confirm the board is powered and servos have external power.
- Confirm `servo_controller.port` and `servo_controller.baudrate` in `wildrobot_config.json` match your board (9600 is common).
- Try smaller `--servo-ids` lists in `scripts/test_hiwonder_board.py` (some links are unreliable with large multi-servo reads).

## Bundle utilities

Validate a policy bundle against MJCF + ONNX:
```bash
cd runtime
uv run wildrobot-validate-bundle --bundle ../policies/wildrobot_policy_bundle
```

Inspect a signals log (quick health summary):
```bash
uv run wildrobot-inspect-log --input signals_log.npz
```

> Note: `wildrobot-replay-policy` was removed with the v1/v2 walking-ref stack
> and has not been reintroduced.
