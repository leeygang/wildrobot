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

Legacy `action_scale_rad` in hardware configuration is **ignored** for the
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

`runtime/configs/hardware_config.json` is the hardware source of truth. Policy
bundles do not contain hardware configuration. `--hardware-config` remains an
explicit override, and `--runtime-config` remains its legacy alias.

Minimal example:

```json
{
  "robot_config_path": "./mujoco_robot_config.json",
  "servo_controller": {
    "type": "hiwonder_ttl_bus",
    "baudrate": 115200,
    "boards": [
      { "name": "left_leg_board", "port": "/dev/serial/by-id/usb-board-left", "servo_ids": [1, 2, 3, 4] },
      { "name": "right_leg_board", "port": "/dev/serial/by-id/usb-board-right", "servo_ids": [5, 6, 7, 8] },
      { "name": "upper_body_board", "port": "/dev/serial/by-id/usb-board-upper", "servo_ids": [40] }
    ],
    "servos": {
      "left_hip_pitch":  { "id": 1, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "left_hip_roll":   { "id": 2, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "left_knee_pitch": { "id": 3, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "left_ankle_pitch": { "id": 4, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "right_hip_pitch": { "id": 5, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "right_hip_roll":  { "id": 6, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "right_knee_pitch":{ "id": 7, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "right_ankle_pitch": { "id": 8, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 },
      "waist_yaw": { "id": 40, "servo_offset_unit": 0, "motor_unit_direction": 1, "joint_angle_at_servo_center_deg": 0 }
    }
  },

  "bno085": {
    "transport": "spi",
    "i2c_address": "0x4B",
    "upside_down": false,
    "spi_baudrate": 50000,
    "spi_read_skip_bytes": 2,
    "spi_cs_pin": "D8",
    "spi_int_pin": "D17",
    "spi_reset_pin": "D27",
    "axis_map": ["+X", "+Y", "+Z"]
  },

  "foot_switches": {
    "enabled": true,
    "left_toe": "D5",
    "left_heel": "D6",
    "right_toe": "D13",
    "right_heel": "D19"
  }
}
```

Notes:
- New deployment bundles keep `wildrobot.xml` and `mujoco_robot_config.json` at
  the root; `bundle_manifest.json` selects each policy ONNX/spec pair.
- Policy/model paths do not belong in the canonical `hardware_config.json`.
- **Servo IDs do not come from MJCF**. Servo IDs are physical IDs stored on the servos / controller and should live in your runtime config (`servo_controller.servos.<joint>.id`).
- `servo_controller.boards` assigns each globally unique servo ID to exactly one named USB TTL board. The required roles are `left_leg_board`, `right_leg_board`, and `upper_body_board`; calibration derives their ID sets from joint names. Run `uv run python runtime/scripts/calibrate.py --config runtime/configs/hardware_config.json --calibrate-servo-board` to detect and write the real ports and assignments. The legacy single-board `servo_controller.port` field remains supported.
- `servo_read_schedule.max_cache_age_s` defines feedback freshness limits. Each board worker continuously round-robins all servos assigned to that board; read-group lists are no longer configured.
- `servo_controller.servos.<joint>.servo_offset_unit` is a per-joint calibration offset in **servo units** around the electrical center (500). Values can be positive or negative. Use the calibration script to write these.
- `servo_controller.servos.<joint>.motor_unit_direction` is a per-joint sign (`+1.0` or `-1.0`) to correct mechanical reversals; if a joint moves the wrong way, flip its sign.
- `servo_controller.servos.<joint>.joint_angle_at_servo_center_deg` (optional, default 0) selects the MuJoCo angle represented by conceptual servo center unit 500. The electrical command at that angle is `500 + servo_offset_unit`. Most joints can keep this at 0. The former `joint_angle_at_zero_unit_deg` key remains load-compatible but is serialized using the new name.
- In the current v2 CAD, the left/right shoulder-roll electrical midpoints are
  MuJoCo `-80` / `+80` degrees respectively. Calibration prints these values;
  do not align raw unit 500 to MuJoCo zero for those two joints.
- Policy and hardware configurations use the same native 17-actuator order. Wrist
  servos are not part of locomotion configuration, feedback, or writes.
- Policy runtime uses the raw Hiwonder/HTD TTL bus through the USB debug board. The old Hiwonder LSC controller-board path is legacy diagnostics only.
- `foot_switches` uses Adafruit Blinka `board` pin names (e.g. `D5`).
  Set `foot_switches.enabled` to `false` when the switches are not installed.
  Contact-free standing policies remain usable; runtime reports the signal as
  unavailable instead of treating placeholder zeros as measured open contacts.

## Run

`wildrobot-run-policy` loads a deployment bundle containing standing and
walking policies. Normal execution stabilizes with the standing policy before
resetting and starting the walking policy.

Flags:
- `--bundle PATH` (required): deployment or legacy policy bundle.
- `--hardware-config PATH`: explicitly override
  `runtime/configs/hardware_config.json`. `--runtime-config` is a legacy alias.
- `--dry-run`: run with mock IO (no servos/IMU/footswitches) — for smoke tests
  and safe validation on a dev machine. Never sleeps.
- `--max-steps N` (default 500), `--log-steps N` (default 20, 0=off).
- `--fall-tilt-deg DEG`: stop control and unload servos when body tilt exceeds
  the limit (default 45 degrees).
- Every run tees stdout/stderr to `_run_policy_logs/` at the repository root.
  `--log PATH` overrides the automatic path; `--log-only PATH` also suppresses
  console output.
- `--velocity-cmd vx` or `--velocity-cmd vx,vy,wz` (default: bundle
  `default_velocity_cmd`, e.g. `0.13,0,0` for smoke9 straight walk).
- `--no-realtime`: don't sleep to maintain `control_hz` (hardware only).

Dry run (no hardware) — exercises the full read → obs → predict → compose →
write loop:

```bash
cd runtime
uv run wildrobot-run-policy --bundle bundles/deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90 \
  --dry-run --max-steps 5 --velocity-cmd 0.13,0.0,0.0
```

On the robot, using the calibrated canonical hardware config:

```bash
cd runtime
uv run wildrobot-run-policy \
  --bundle bundles/deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90 \
  --velocity-cmd 0.13,0,0 --confirm-before-walk
```

## Run a bundle from `training/checkpoints/` (on the WildRobot device)

Assumptions:
- You have the `wildrobot/` repo synced on the WildRobot device.
- The bundle directory exists under `training/checkpoints/` (exported via `training/exports/export_policy_bundle_cli.py`).

Export one deployment bundle from the selected standing and walking
checkpoints. Hardware configuration is intentionally excluded:

```bash
uv run python training/exports/export_policy_bundle_cli.py \
  --walk-checkpoint training/checkpoints/<walking-run>/checkpoint_NNNN.pkl \
  --training-config training/checkpoints/<walking-run>/training_config.yaml \
  --standing-checkpoint training/checkpoints/<standing-run>/checkpoint_NNNN.pkl \
  --standing-training-config training/checkpoints/<standing-run>/training_config.yaml \
  --bundle-path /tmp/wr_bundle
```

From the repo root on the device:
```bash
cd ~/wildrobot/runtime

# (Recommended) sanity-check raw TTL bus reads before starting control loop
uv run python scripts/probe_hiwonder_ttl_timing.py \
  --port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --baudrate 115200 --servo-ids 1,2,3 --mode read --cycles 20

# Validate the bundle is self-consistent (policy_spec + ONNX dims + actuator order).
# Works on non-Linux dev machines too (rpi-gpio is gated to Linux).
uv run wildrobot-validate-bundle --bundle /tmp/wr_bundle

# Dry run first (no hardware) to confirm the loop builds obs + composes targets:
uv run wildrobot-run-policy --bundle /tmp/wr_bundle --dry-run --max-steps 5

# Run the control loop on hardware (Ctrl+C to stop; runtime disables actuators on exit)
uv run wildrobot-run-policy --bundle /tmp/wr_bundle --confirm-before-walk

# Calibrate the canonical hardware configuration using walking or standing order.
uv run python scripts/calibrate.py --bundle /tmp/wr_bundle \
  --config configs/hardware_config.json --policy-role walking --calibrate
```

Notes:
- `wildrobot.xml` and `mujoco_robot_config.json` are shared policy-model
  snapshots. Each `policies/<role>` directory owns its ONNX, `policy_spec.json`,
  and `runtime_policy_config.json`.
- Servo IDs, offsets, board ports, IMU mapping, and GPIO live only in
  `runtime/configs/hardware_config.json`.

The loop runs for `--max-steps` control steps and then exits, disabling
actuators via `robot_io.close()`.  Ctrl+C also stops it and unloads servos.
To copy the newest hardware policy log from `wrdev.local` into the local
repository, run `./scripts/scp_from_remote.sh --latest-policy-log` from the
repository root.

### Stable-only mode

`--stable-only` (alias `--stable_only`) runs a native 17D standing bundle. The
mode always uses a zero velocity command and runs until Ctrl+C. `--max-steps`
applies only to walking and dry-run execution. Historical deployment bundles
whose walking half is still 21D are archival; point directly at a 17D standing
policy directory for standing-only operation.

```bash
cd runtime
uv run wildrobot-run-policy \
  --bundle bundles/<native-17d-standing-policy> \
  --stable-only \
  --log-steps 20
```

Runtime startup must report 17 policy and hardware actuators. A combined
standing/walking deployment is valid only after the walking policy is migrated
and exported as `(937,17)`. No checked-in historical bundle satisfies the full
native contract.

For the paired hardware diagnostic, run the home hold and standing policy from
the same foot placement with:

```bash
cd runtime
uv run python scripts/run_paired_policy_trial.py \
  --session-log ../_run_policy_logs/paired_session.log
```

The runner prompts once before each of three trials. `READY` starts a single
runtime process that blends from measured joint positions to home for two
seconds, holds home for the remainder of `--home-seconds`, resets policy state,
and automatically activates the standing policy without restarting the IMU or
unloading servos. Each trial writes one combined log under `_run_policy_logs`;
`--session-log` additionally captures the wrapper output in one shareable file.
A policy-runtime import preflight runs before any servo is loaded. The runner
rejects startup or motion above 15 degrees and stops policy control after 60
seconds or at 20 degrees of tilt. This continuous preparation-to-policy handoff
matches ToddlerBot's initial-stand control-loop pattern.

To characterize natural-placement home states without running the policy:

```bash
cd runtime
uv run python scripts/run_paired_policy_trial.py --home-characterization
```

This mode defaults to ten operator-confirmed 60-second trials. Reposition the
robot normally before each `READY`; exact foot placement is not expected to
match. Each home log contains 50 Hz runtime-frame IMU, home target, cached joint
readback/error, servo-cache age, and four footswitch observations. Footswitches
are recorded but never gate a trial. Servos unload after every trial, the
15-degree cutoff is retained, and a
`vXXX_ckptYYY_home_characterization_summary_*.log` JSON summary reports pitch,
tilt, drift slope, joint error, sample rate, and contact ratios across trials.

If you see servo cache initialization or position-read failures:
- Confirm the USB TTL debug board is powered and servos have external power.
- Rerun `calibrate.py --calibrate-servo-board` and confirm every board and servo ID is listed once.
- Confirm `servo_controller.boards[*].port` and `servo_controller.baudrate` in `hardware_config.json` match the debug boards (`115200`).
- Try smaller `--servo-ids` lists in `scripts/probe_hiwonder_ttl_timing.py` to isolate bus or servo issues.

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
