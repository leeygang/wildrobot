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

You can start from the example file at `runtime/configs/wildrobot_config.example.json` and copy it to `~/wildrobot_config.json` (or pass `--config`).

Minimal example:

```json
{
  "mjcf_path": "../assets/wildrobot.xml",
  "policy_onnx_path": "../policies/wildrobot_policy/policy.onnx",

  "control_hz": 50,
  "action_scale_rad": 0.35,
  "velocity_cmd": 0.0,

  "hiwonder": {
    "port": "/dev/ttyUSB0",
    "baudrate": 9600,
    "servo_ids": {
      "left_hip_pitch": 1,
      "left_hip_roll": 2,
      "left_knee_pitch": 3,
      "left_ankle_pitch": 4,
      "right_hip_pitch": 5,
      "right_hip_roll": 6,
      "right_knee_pitch": 7,
      "right_ankle_pitch": 8
    },
    "joint_offsets_rad": {
      "left_hip_pitch": 0.0,
      "left_hip_roll": 0.0,
      "left_knee_pitch": 0.0,
      "left_ankle_pitch": 0.0,
      "right_hip_pitch": 0.0,
      "right_hip_roll": 0.0,
      "right_knee_pitch": 0.0,
      "right_ankle_pitch": 0.0
    },
    "joint_directions": {
      "left_hip_pitch": 1.0,
      "left_hip_roll": 1.0,
      "left_knee_pitch": 1.0,
      "left_ankle_pitch": 1.0,
      "right_hip_pitch": 1.0,
      "right_hip_roll": 1.0,
      "right_knee_pitch": 1.0,
      "right_ankle_pitch": 1.0
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
- **Servo IDs do not come from MJCF**. Servo IDs are physical IDs stored on the servos / controller and should live in your runtime config (`hiwonder.servo_ids`).
- `hiwonder.joint_offsets_rad` is a per-joint calibration offset (radians) applied at the hardware boundary (before rad→servo conversion).
- `hiwonder.joint_directions` is a per-joint sign (`+1.0` or `-1.0`) to correct mechanical reversals; if a joint moves the wrong way, flip its sign.
- `foot_switches` uses Adafruit Blinka `board` pin names (e.g. `D5`).

## Run

```bash
wildrobot-run-policy --config ~/wildrobot_config.json
```

Stop with Ctrl+C (runtime will unload servos).

Optional: capture a replay log for offline policy replay:
```bash
wildrobot-run-policy --config ~/wildrobot_config.json --log-path signals_log.npz --log-steps 2000
```

## Bundle utilities

Validate a policy bundle against MJCF + ONNX:
```bash
wildrobot-validate-bundle --config ~/wildrobot_config.json
```

Replay a bundle on logged signals (npz → obs/actions):
```bash
wildrobot-replay-policy --bundle ../policies/wildrobot_policy_bundle --input signals_log.npz --output replay_output.npz
```
