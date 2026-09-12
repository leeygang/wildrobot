# tools/sysid

`v0.19.1` adds a usable capture path for representative joint SysID traces.

## HTD-45H automatic known-load capture

Use `runtime/scripts/capture_servo_sysid.py` for real HTD-45H
characterization. It runs a single isolated fixture servo through symmetric 2, 5,
and 8 degree steps followed by 0.1-2 Hz decaying chirps at the deployment rate
of 50 Hz. The profile follows ToddlerBot's multi-amplitude chirp method, with a
smaller initial range and bandwidth because the first WildRobot measurement is
on an uncharacterized high-torque servo and lever fixture.

The validated BAM fixture has a fixed top/servo body and a 0.612761 kg moving
arm/weight subtree. Its zero pose hangs nearly vertically, so the default
profile reaches about 0.10 Nm rather than the maximum horizontal gravity load.

The physical fixture must have:

- only the test servo connected to the selected TTL adapter;
- a rigidly mounted servo housing matching `assets/bam/robot.xml`;
- verified collision-free travel across the complete commanded range;
- a catcher that supports the lever and mass whenever torque is disabled; and
- a reachable power cutoff. Never use a hand as the stop or catcher.

Mount the lever so its hanging zero pose matches MJCF `pitch=0`. The first run
uses the exported fixture mass and inertia directly. The fixture servo is
addressed in raw centered coordinates: electrical unit 500 is 0 degrees and
increasing electrical units are positive.

First assign the isolated fixture servo its reserved ID. This command does not
move or unload the servo, but broadcast discovery requires that every other
servo be physically disconnected from the selected TTL bus:

```bash
uv run python runtime/scripts/set_sysid_servo_id.py \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --servo-id 100
```

The tool scans all servo addresses, prints the current ID, aborts if more than
one distinct ID responds, requires explicit confirmation, writes the requested
ID, and verifies both the old and new addresses. Two servos already sharing the
same ID cannot be distinguished by the HTD addressed protocol, so the fixture
servo should still be the only servo physically connected. With that servo
connected, validate the capture configuration:

```bash
uv run python runtime/scripts/capture_servo_sysid.py \
  --servo-id 100 \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --center-deg 0 \
  --fixture-mjcf assets/bam/robot.xml \
  --fixture-joint pitch \
  --fixture-direction 1 \
  --fixture-qpos-offset-deg 0 \
  --write-deadband-units 0 \
  --cooldown-target-c 35 \
  --servo-label htd45h-sysid-id100 \
  --fixture-label bam-v1 \
  --dry-run
```

Use `--fixture-direction -1` instead if increasing raw servo angle moves
opposite to positive MJCF `pitch`. Remove `--dry-run` only
after checking the printed root, moving body, moving mass, inertia, torque,
port, and duration. The script then:

1. prints a cancellable startup delay before opening the bus or applying torque;
2. reads the unpowered starting position and torque state, unloads the servo,
   and waits until its temperature reaches `--cooldown-target-c`;
3. records every cooldown temperature, voltage, and actual wait interval;
4. primes the target at the current position, enables torque, and verifies it;
5. moves to center at a bounded speed, reads back the servo's accepted target,
   position, torque state, voltage, and temperature, and retries automatically
   up to `--center-max-attempts` when the center tolerance is not met; a retry
   re-primes and reloads a servo that disabled torque, then uses a longer move;
6. runs the full profile while checking encoder reads, tracking error, and loop
   timing;
7. returns to center, reads voltage and temperature again, moves to the
   configured gravity-neutral unload pose, verifies it, and disables torque;
8. stops and unloads the servo in a `finally` path on completion, error, or
   Ctrl-C.

Use `--write-deadband-units 0` for the actuator-model identification captures;
this preserves every quantized command. Use a separate run with the deployment
value `3` to measure the combined actuator plus runtime command-suppression
behavior. Each load center remains a separate capture so the operator can
verify the clear travel range and catcher before higher-torque tests. The
campaign runner below invokes those captures sequentially without removing
their cooldown, diagnostics, or safety checks.

Each condition is one invocation so the fixture cannot be reconfigured while
the servo is energized. Review the hanging-zero capture before running
off-vertical centers such as 30 or 45 degrees. Those positions provide greater
gravity load without adding weight. Manual mass/radius arguments remain
available for fixtures without a validated MJCF, but they cannot be combined
with `--fixture-mjcf`.

The output pair is written under `runtime/calibration/servo_sysid/` by default:

- `.npz`: the established `command_rad`, `measured_position_rad`,
  `measured_velocity_rad_s`, and `timestamps_s` arrays, plus raw servo units,
  the smooth `requested_command_rad`, segment labels, command/read timing, and
  `fixture_qpos_rad`, MuJoCo-derived holding torque and moving-load inertia.
  `command_rad` is the quantized target actually transmitted after the
  selected write deadband. Schema v4 also records scheduled time, scheduler
  sleep, actual serial-write completion, command age at each read, and cooldown
  temperature/voltage history;
- `.json`: fixture geometry and inertia, servo/bus identity, pre/post voltage
  and temperature, initial/center/final positions, torque-load state, all
  operator/cooldown/preparation/profile/return waits, step 10/50/90% response,
  steady gain, command-update cadence, outcome, and per-chirp correlation lag.

The reported chirp `delay_s` is explicitly a cross-correlation lag that includes
servo response dynamics; it is not a pure serial or actuator transport delay.

### Standard follow-up campaign

After validating one zero-load, 0.1-2 Hz baseline capture, run the remaining
bandwidth, signed-load, held-out validation, and repeatability conditions with
one command:

```bash
uv run python runtime/scripts/run_servo_sysid_campaign.py \
  --servo-id 100 \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --fixture-mjcf assets/bam/robot.xml \
  --fixture-joint pitch \
  --fixture-direction 1 \
  --fixture-qpos-offset-deg 0
```

The runner executes six captures in order: a 2-degree 0.1-10 Hz bandwidth
trace at zero, fit traces at +30 and -30 degrees, held-out validation traces at
+45 and -45 degrees, and a final zero-load repeat. All actuator-model captures
use zero command-write deadband. Each child capture reads and records the servo
EEPROM angle and voltage limits, temperature limit, position/motor mode, alarm
configuration, and accepted move target. It requires no typed confirmation:
after a cancellable startup delay, it waits unloaded for the servo to cool to
35 C, runs the profile, returns to the verified zero-degree gravity-neutral
pose, and disables torque. A failure or operator abort stops the campaign
immediately; already completed capture pairs remain under the campaign
directory printed at startup.

After a corrected setup or transient preparation failure, resume in a new
campaign directory without repeating completed conditions:

```bash
uv run python runtime/scripts/run_servo_sysid_campaign.py \
  --servo-id 100 \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --start-at B1_fit_plus30
```

Use `--dry-run` to validate all six profiles and fixture loads without opening
the serial port.

The HTD protocol exposes position, supply voltage, and temperature, but not
motor current or torque. Torque-related parameters therefore require the
known lever mass, radius, and inertia and must be fitted across multiple load
conditions; one unloaded trace is not sufficient to identify torque limits.
The reported fixture inertia excludes the MJCF joint's placeholder `armature`
value so that the later fit does not count an assumed actuator inertia as a
measured mechanical load.

References:

- ToddlerBot paper, system-identification method: https://arxiv.org/abs/2502.00893
- ToddlerBot implementation: `~/projects/toddlerbot/toddlerbot/policies/sysID.py`
  and `~/projects/toddlerbot/toddlerbot/tools/run_sysID.py`
- HTD/Hiwonder protocol examples: `docs/HTD-45H Serial Bus Servo/`

## Capture commands

Step response:

```bash
uv run python tools/sysid/run_capture.py \
  --mode step \
  --joint-name left_knee_pitch \
  --runtime-config runtime/configs/hardware_config.json \
  --output-dir runtime/logs/sysid
```

Hold response:

```bash
uv run python tools/sysid/run_capture.py \
  --mode hold \
  --joint-name left_knee_pitch \
  --hold-rad 0.15 \
  --output-dir runtime/logs/sysid
```

Chirp response:

```bash
uv run python tools/sysid/run_capture.py \
  --mode chirp \
  --joint-name left_knee_pitch \
  --amplitude-rad 0.2 \
  --chirp-start-hz 0.2 \
  --chirp-end-hz 3.0 \
  --output-dir runtime/logs/sysid
```

By default, this commands real hardware through the configured Hiwonder servo board and records measured readback.

Offline/synthetic fallback is still available for CI/dev:

```bash
uv run python tools/sysid/run_capture.py \
  --capture-source synthetic \
  --mode step \
  --joint-name left_knee_pitch \
  --output-dir runtime/logs/sysid
```

## Outputs

Each run writes:

- `<prefix>_<mode>_<joint>.npz` with arrays:
  - `command_rad`
  - `measured_position_rad`
  - `measured_velocity_rad_s`
  - `timestamps_s`
- `<prefix>_<mode>_<joint>.json` manifest with metadata:
  - joint name
  - mode
  - sample rate
  - runtime config path
  - asset/runtime context

## Notes

- Milestone `v0.19.1` now includes a real measurement loop (`--capture-source hardware`, default) plus structured exports.
- Full parameter fitting/optimization loops remain intentionally deferred to `v0.19.2+`.
