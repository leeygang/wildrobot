# WildRobot Runtime

Lightweight hardware runtime for running WildRobot ONNX policies on Raspberry Ubuntu.

## Install (on robot)

From the `wildrobot/` repo root:

```bash
cd runtime
pip install -e .
```

## Configuration

The runtime reads a JSON config. By default it looks at `~/wildrobot_config.json`.

You can start from the sample at `runtime/configs/wr_runtime_config.json` and copy it to `~/wildrobot_config.json` (or pass `--config`).

Minimal example:

```json
{
  "mjcf_path": "../assets/wildrobot.xml",
  "policy_onnx_path": "../policies/wildrobot_policy/policy.onnx",

  "control_hz": 50,
  "action_scale_rad": 0.35,
  "velocity_cmd": 0.0,

  "servo_controller": {
    "type": "hiwonder",
    "port": "/dev/ttyUSB0",
    "baudrate": 9600,
    "servos": {
      "left_hip_pitch":  { "id": 1, "offset_unit": 0, "direction": 1 },
      "left_hip_roll":   { "id": 2, "offset_unit": 0, "direction": 1 },
      "left_knee_pitch": { "id": 3, "offset_unit": 0, "direction": 1 },
      "left_ankle_pitch": { "id": 4, "offset_unit": 0, "direction": 1 },
      "right_hip_pitch": { "id": 5, "offset_unit": 0, "direction": 1 },
      "right_hip_roll":  { "id": 6, "offset_unit": 0, "direction": 1 },
      "right_knee_pitch":{ "id": 7, "offset_unit": 0, "direction": 1 },
      "right_ankle_pitch": { "id": 8, "offset_unit": 0, "direction": 1 }
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
- `servo_controller.servos.<joint>.offset_unit` is a per-joint calibration offset in **servo units** relative to center (500). Values can be positive or negative. Use the calibration script to write these.
- `servo_controller.servos.<joint>.direction` is a per-joint sign (`+1.0` or `-1.0`) to correct mechanical reversals; if a joint moves the wrong way, flip its sign.
- `foot_switches` uses Adafruit Blinka `board` pin names (e.g. `D5`).

## Run

```bash
wildrobot-run-policy --bundle ../policies/wildrobot_policy_bundle
```

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
