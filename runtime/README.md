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
  "policy_onnx_path": "../policies/wildrobot_policy.onnx",

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
    }
  },

  "bno085": {
    "i2c_address": "0x4A",
    "upside_down": false
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
- **Servo IDs do not come from MJCF**. Servo IDs are physical IDs stored on the servos / controller and should live in your runtime config (`hiwonder.servo_ids`).
- `foot_switches` uses Adafruit Blinka `board` pin names (e.g. `D5`).

## Run

```bash
wildrobot-run-policy --config ~/wildrobot_config.json
```

Stop with Ctrl+C (runtime will unload servos).
