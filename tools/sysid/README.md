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

First assign the isolated fixture servo its reserved ID. Without `--set-unit`,
this command does not move or unload the servo. Broadcast discovery requires
that every other servo be physically disconnected from the selected TTL bus:

```bash
uv run python runtime/scripts/set_sysid_servo_id.py \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --servo-id 100
```

The tool scans all servo addresses, prints the current ID, aborts if more than
one distinct ID responds, requires explicit confirmation, writes the requested
ID, and verifies both the old and new addresses. Two servos already sharing the
same ID cannot be distinguished by the HTD addressed protocol, so the fixture
servo should still be the only servo physically connected.

To position a bare replacement servo at its raw electrical center before
installing the horn, keep the shaft mechanically clear and add `--set-unit
500`:

```bash
uv run python runtime/scripts/set_sysid_servo_id.py \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --servo-id 100 \
  --set-unit 500
```

`--set-unit` accepts an inclusive raw position from 0 through 1000. The tool
prints the current encoder unit, primes that position before enabling torque,
moves at 20 degrees/s, and verifies the final position within five raw units.
It then holds the target with torque enabled until the operator types `y` to
unload. Ctrl-C or another exception also executes the torque-off cleanup. Do
not use an endpoint target with an installed linkage unless its full travel has
been mechanically verified.

With that servo connected, validate the capture configuration:

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
4. primes the target at the current position, enables torque, verifies it, and
   monitors a loaded hold before issuing any movement;
5. moves to center at a bounded speed while continuously sampling position,
   torque-enable state, voltage, and temperature; it records whether torque was
   lost during the pre-move hold, center movement, or center settle and retries
   automatically up to `--center-max-attempts`; a retry re-primes and reloads a
   servo that disabled torque, then uses a longer move;
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
  temperature/voltage history. `preparation_*` arrays preserve the monitored
  pre-move, start-pose normalization, center-move, center-settle, and
  unload-pose states even when capture fails before the profile starts;
- `.json`: fixture geometry and inertia, servo/bus identity, pre/post voltage
  and temperature, initial/center/final positions, torque-load state, all
  operator/cooldown/preparation/profile/return waits, step 10/50/90% response,
  steady gain, command-update cadence, outcome, and per-chirp correlation lag.

The reported chirp `delay_s` is explicitly a cross-correlation lag that includes
servo response dynamics; it is not a pure serial or actuator transport delay.

### Torque-loss preparation diagnostic

If the servo unexpectedly reports `loaded=false`, do not repeat the full
campaign. Power-cycle the fixture, leave the load catcher in place, and run one
slow preparation-only movement:

```bash
uv run python runtime/scripts/capture_servo_sysid.py \
  --servo-id 100 \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --center-deg 30 \
  --fixture-mjcf assets/bam/robot.xml \
  --fixture-joint pitch \
  --fixture-direction 1 \
  --fixture-qpos-offset-deg 0 \
  --prepare-speed-deg-s 5 \
  --prepare-monitor-hz 25 \
  --pre-move-hold-s 2 \
  --center-max-attempts 1 \
  --prepare-only \
  --servo-label htd45h-sysid-id100 \
  --fixture-label bam-v1
```

This diagnostic first holds the servo at its measured starting position for two
seconds, then moves toward +30 degrees at 5 degrees/s, observes the one-second
center settle, and returns to the gravity-neutral zero pose. It never runs the
step/chirp profile. The JSON reports `first_unload`; the NPZ keeps the complete
25 Hz preparation trace and modeled holding torque. A failure in
`post_load_hold` indicates torque cannot remain enabled before motion;
`center_move` indicates a motion/load transient or obstruction; and
`center_settle` indicates that the final static load cannot be held.

The default voltage safety floor is 9.6 V, matching the HTD-45H vendor working
range. If any preparation sample crosses that floor while torque remains
enabled, the script skips the profile, returns to the neutral unload pose, and
marks the capture failed. Console error and exception messages are yellow on a
color-capable terminal; JSON error strings remain plain text.

The HTD protocol does not report motor current and does not expose a latched
active fault code. `alarm_mask=7` only means all three LED alarm sources are
configured. A brief power-rail collapse may therefore require an oscilloscope
or current-logging supply at the servo connector even when the sampled voltage
looks normal afterward.

### Separated load/speed characterization

Before running chirps, characterize the protection boundary with one factor at
a time:

```bash
uv run python runtime/scripts/run_servo_sysid_campaign.py \
  --plan limits \
  --servo-id 100 \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --fixture-mjcf assets/bam/robot.xml \
  --fixture-joint pitch \
  --fixture-direction 1 \
  --fixture-qpos-offset-deg 0
```

The limits plan runs preparation-only moves, never the step/chirp profile:

1. load sweep: 10, 20, and 30 degrees, all at 5 degrees/s;
2. speed sweep: the identical 0-to-10-degree path at 20, 50, and 100
   degrees/s; and
3. interaction check: the identical 0-to-30-degree path at 10, 15, and 20
   degrees/s.

Before every condition, the capture normalizes the fixture to the configured
zero-degree unload pose at 5 degrees/s and verifies it. The outbound leg then
uses only that condition's test speed, while every return to zero remains fixed
at 5 degrees/s. This prevents an arbitrary unpowered starting angle or a fast
return from contaminating the requested comparison.

The load cases hold each endpoint for three seconds. The validated BAM model
predicts approximately 0.12, 0.24, and 0.35 Nm of static gravity torque at 10,
20, and 30 degrees. Keeping speed fixed in the first sweep isolates the load
trend; keeping the endpoint and path fixed in the second isolates the
speed/current-transient trend. The third sweep then measures their interaction
at the previously demonstrated 30-degree load.

Every condition uses one test attempt and 50 Hz preparation telemetry. The
JSON records the initial and normalized positions, normalization timing,
outbound test speed, and independent return speed. A voltage or torque-disable
event stops the campaign immediately so a reset servo is never mistaken for a
valid slower retry. Power-cycle the fixture and resume at a named condition
only after inspecting the failed JSON/NPZ pair, for example:

```bash
uv run python runtime/scripts/run_servo_sysid_campaign.py \
  --plan limits \
  --servo-id 100 \
  --board-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127022-if00 \
  --start-at S1_speed_10deg_20dps
```

This preserves ToddlerBot's later multi-amplitude/frequency SysID structure,
but adds an explicit load axis because WildRobot uses a different, larger
actuator/fixture system and the measured HTD/TTL path has already crossed its
9.6 V operating floor during motion. Use an oscilloscope or supply-current
logger for sub-20 ms transients; the 50 Hz protocol telemetry cannot prove
that faster events did not occur.

### Standard follow-up campaign

After the separated limits plan passes, run the remaining bandwidth,
signed-load, held-out validation, and repeatability conditions with one
command:

```bash
uv run python runtime/scripts/run_servo_sysid_campaign.py \
  --plan sysid \
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
use zero command-write deadband. Each child capture normalizes to zero before
its measured move and returns to zero at 5 degrees/s. It also reads and records
the servo EEPROM angle and voltage limits, temperature limit, position/motor
mode, alarm configuration, and accepted move target. It requires no typed
confirmation: after a cancellable startup delay, it waits unloaded for the
servo to cool to 35 C, runs the profile, returns to the verified zero-degree
gravity-neutral pose, and disables torque. A failure or operator abort stops
the campaign immediately; already completed capture pairs remain under the
campaign directory printed at startup.

After a corrected setup or transient preparation failure, resume in a new
campaign directory without repeating completed conditions:

```bash
uv run python runtime/scripts/run_servo_sysid_campaign.py \
  --plan sysid \
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
