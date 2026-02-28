# WildRobot Runtime

Lightweight hardware runtime for running WildRobot ONNX policies on Raspberry Ubuntu.

Implementation lives in `runtime/wr_runtime/` (this is what the installed CLIs use).

Breaking change: the legacy modules under `runtime/control`, `runtime/inference`, `runtime/utils`, and `runtime/validation`
were removed. Use the console scripts (`wildrobot-run-policy`, `wildrobot-validate-bundle`, ...) or import from `wr_runtime.*`.

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

You can start from the sample at `runtime/configs/wr_runtime_config.json` and copy it to `~/wildrobot_config.json` (or pass `--config`).

Minimal example:

```json
{
  "mjcf_path": "../assets/v1/wildrobot.xml",
  "policy_onnx_path": "../policies/wildrobot_policy/policy.onnx",

  "control_hz": 50,
  "action_scale_rad": 0.35,
  "velocity_cmd": 0.0,

  "servo_controller": {
    "type": "hiwonder",
    "port": "/dev/ttyUSB0",
    "baudrate": 9600,
    "servos": {
      "left_hip_pitch":  { "id": 1, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "left_hip_roll":   { "id": 2, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "left_knee_pitch": { "id": 3, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "left_ankle_pitch": { "id": 4, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "right_hip_pitch": { "id": 5, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "right_hip_roll":  { "id": 6, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "right_knee_pitch":{ "id": 7, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 },
      "right_ankle_pitch": { "id": 8, "offset_unit": 0, "direction": 1, "center_deg_offset": 0 }
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
- `servo_controller.servos.<joint>.center_deg_offset` defines the MuJoCo angle (deg) that maps to servo center (500). Use $+90$ or $-90$ for shoulder pitch/roll as needed.
- `servo_controller.servos.<joint>.offset_unit` is a per-joint calibration offset in **servo units** around the electrical center (500). Values can be positive or negative. Use the calibration script to write these.
- `servo_controller.servos.<joint>.direction` is a per-joint sign (`+1.0` or `-1.0`) to correct mechanical reversals; if a joint moves the wrong way, flip its sign. Direction is applied after subtracting the center.
- `foot_switches` uses Adafruit Blinka `board` pin names (e.g. `D5`).

## Run

```bash
wildrobot-run-policy --bundle ../policies/wildrobot_policy_bundle
```

## Run a bundle from `training/checkpoints/` (on the WildRobot device)

Assumptions:
- You have the `wildrobot/` repo synced on the WildRobot device.
- The bundle directory exists under `training/checkpoints/` (exported via `training/exports/export_policy_bundle_cli.py`).

Example bundle (already in this repo layout):
- `training/checkpoints/standing_push_v0.13.4_ckpt20/`

From the repo root on the device:
```bash
cd ~/wildrobot

# Run using the runtime uv environment (no manual venv activation needed)
cd runtime

# Validate the bundle is self-consistent (policy_spec + ONNX dims + actuator order)
uv run wildrobot-validate-bundle --bundle ../training/checkpoints/standing_push_v0.13.4_ckpt20

# Run the control loop (Ctrl+C to stop; runtime disables actuators on exit)
uv run wildrobot-run-policy --bundle ../training/checkpoints/standing_push_v0.13.4_ckpt20
```

Notes:
- The bundle contains its own `wildrobot_config.json` (hardware settings) and `wildrobot.xml` (actuator order snapshot).
- If this device/robot has different servo IDs, offsets, IMU axis map, or footswitch pins, edit:
  `training/checkpoints/standing_push_v0.13.4_ckpt20/wildrobot_config.json` before running.

Stop with Ctrl+C (runtime will unload servos).

Optional: capture a replay log for offline policy replay:
```bash
wildrobot-run-policy --bundle ../policies/wildrobot_policy_bundle --log-path signals_log.npz --log-steps 2000
```

## Bundle utilities

Validate a policy bundle against MJCF + ONNX:
```bash
wildrobot-validate-bundle --bundle ../policies/wildrobot_policy_bundle
```

Replay a bundle on logged signals (npz → obs/actions):
```bash
wildrobot-replay-policy --bundle ../policies/wildrobot_policy_bundle --input signals_log.npz --output replay_output.npz
```

Inspect a signals log (quick health summary):
```bash
wildrobot-inspect-log --input signals_log.npz
```
