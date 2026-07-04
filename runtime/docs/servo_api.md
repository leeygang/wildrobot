# Servo API Design

This document proposes the WildRobot servo API layers for the HTD/Hiwonder
servo runtime. It is a design review document, not an implementation plan that
has already been accepted.

## Summary
The proposed layers are:

```text
policy/runtime RobotIO
  -> calibrated actuator API in radians
     -> servo IO worker/cache/scheduler
        -> raw servo command API
           -> serial transport
```

The raw servo command API should wrap the `55 55` HTD/Hiwonder bus-servo
protocol. It should not know about policy actuator names, joint limits,
calibration offsets, cache age, or read schedules.

## Evidence

The HTD examples use a request/response protocol:

```text
[0x55][0x55][servo_id][length][command][params...][checksum]
```

Relevant command IDs from the HTD examples:

```text
MOVE_TIME_WRITE       = 1
MOVE_TIME_READ        = 2
MOVE_TIME_WAIT_WRITE  = 7
MOVE_TIME_WAIT_READ   = 8
MOVE_START            = 11
MOVE_STOP             = 12
ID_WRITE              = 13
ID_READ               = 14
POS_READ              = 28
```

The vendor read path sends a read command, switches the half-duplex bus to read
mode, waits for a response, and parses the returned packet. There is no evidence
of unsolicited streaming telemetry or protocol-level async reads.

Measured raw TTL bus timing at 115200 baud:

```text
1 servo read:
  avg 3.640 ms

1 servo read+write:
  avg 4.537 ms

3 servos read+write:
  avg 7.971 ms
  p95 8.094 ms
```

Reads dominate. Writes are comparatively cheap.

## ToddlerBot Comparison

ToddlerBot uses Dynamixel servos and the Dynamixel SDK group APIs:

```text
GroupSyncRead
GroupBulkRead
GroupSyncWrite
```

ToddlerBot still serializes access to the bus with a mutex, but its protocol and
baudrate make full feedback much cheaper than WildRobot's HTD bus.

Public references:

- ROBOTIS Dynamixel Protocol 2.0 and Dynamixel SDK group read/write APIs.
- ROS control / ros2_control hardware-interface pattern: controller logic is
  separated from hardware read/write details.

WildRobot should follow the same architectural separation, but cannot copy the
Dynamixel full-state sync-read design directly because HTD raw servo reads are
single-servo request/response transactions.

## Goals

- Keep the policy-facing interface in joint radians.
- Keep raw protocol details out of policy/runtime code.
- Ensure exactly one owner touches the serial bus.
- Support a background worker that prioritizes writes and fills a servo state
  cache from staggered reads.
- Make timing, cache age, read failures, and write delays measurable.

## Non-Goals

- Do not change the ONNX policy shape.
- Do not introduce a new training interface in this API change.
- Do not assume HTD supports protocol-level async reads.
- Do not hide stale feedback; expose cache age and failure counters.

## Layer 1: Serial Transport

The transport owns the serial port and handles bytes only.

Responsibilities:

- Open and close the serial port.
- Configure baudrate, timeout, parity, stop bits.
- Write raw bytes.
- Read raw bytes until a packet, timeout, or quiet period.
- Provide metrics for byte counts and blocking time.

It should not know command IDs or packet semantics.

Configuration ownership:

```python
@dataclass(frozen=True)
class SerialTransportConfig:
    port: str
    baudrate: int = 115200
    byte_timeout_s: float = 0.001
    write_timeout_s: float = 0.002
    bytesize: int = serial.EIGHTBITS
    parity: str = serial.PARITY_NONE
    stopbits: int = serial.STOPBITS_ONE
    reset_buffers_on_open: bool = True
    tx_enable_pin: int | None = None
    rx_enable_pin: int | None = None
```

Suggested API:

```python
class SerialTransport:
    def __init__(self, config: SerialTransportConfig) -> None: ...

    @property
    def port(self) -> str: ...

    @property
    def baudrate(self) -> int: ...

    def open(self) -> None: ...
    def write(self, packet: bytes) -> None: ...
    def read_available(self, *, deadline_s: float, quiet_s: float = 0.0005) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def reset_output_buffer(self) -> None: ...
    def close(self) -> None: ...
```

For a raw TTL debug board connected by USB, pyserial controls the USB serial
device. For a GPIO half-duplex UART path, the transport may also need direction
control, but the upper layers should not change.

Important separation:

```text
byte_timeout_s:
  short pyserial read timeout used while collecting bytes

RawServoBus response_timeout_s:
  logical command timeout used while waiting for one servo response
```

Example construction:

```python
transport = SerialTransport(
    SerialTransportConfig(
        port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_...",
        baudrate=115200,
        byte_timeout_s=0.001,
        write_timeout_s=0.002,
    )
)
```

## Layer 2: Raw Servo Command API

This layer wraps the raw HTD/Hiwonder `55 55` bus-servo command protocol.

Responsibilities:

- Build packets.
- Validate checksums.
- Parse responses.
- Provide typed command methods.
- Own request/response timeouts for a single transaction.

It should not know actuator names, joint signs, home poses, or policy timing.

Configuration ownership:

```python
@dataclass(frozen=True)
class RawServoBusConfig:
    response_timeout_s: float = 0.006
    quiet_s: float = 0.0005
    broadcast_id: int = 0xFE
    min_servo_id: int = 0
    max_servo_id: int = 253
    min_position: int = 0
    max_position: int = 1000
```

Suggested API:

```python
class RawServoBus:
    def __init__(self, transport: SerialTransport, config: RawServoBusConfig) -> None: ...

    def read_id(self, target_id: int | None = None) -> int | None: ...
    def write_id(self, old_id: int, new_id: int) -> None: ...

    def move_time_write(self, servo_id: int, position: int, time_ms: int) -> None: ...
    def move_time_wait_write(self, servo_id: int, position: int, time_ms: int) -> None: ...
    def move_start(self, servo_id: int) -> None: ...
    def move_stop(self, servo_id: int) -> None: ...

    def read_position(self, servo_id: int) -> int | None: ...
    def read_positions(self, servo_ids: Sequence[int]) -> dict[int, int]: ...

    def unload(self, servo_id: int) -> None: ...
```

`read_positions()` is a convenience loop over single-servo `POS_READ` unless a
future controller path provides a real group read. It should return partial
success as a map rather than failing the whole group.

Packet helpers should be pure and unit-testable:

```python
def build_packet(servo_id: int, command: int, params: Sequence[int]) -> bytes: ...
def parse_packets(data: bytes) -> list[RawServoPacket]: ...
def checksum(packet_without_checksum: bytes) -> int: ...
```

Checksum rule:

```text
checksum = bitwise_not(sum(bytes from servo_id through params)) & 0xFF
```

Example construction:

```python
raw_bus = RawServoBus(
    transport=transport,
    config=RawServoBusConfig(response_timeout_s=0.006),
)
```

## Layer 3: Servo IO Worker

This layer owns runtime bus scheduling.

Responsibilities:

- Be the only thread that calls `RawServoBus`.
- Accept latest 21D target commands from the policy thread.
- Prioritize writes over reads.
- Poll one scheduled servo at a time when no write is pending.
- Update full joint-state cache.
- Publish a non-blocking cached servo state for the policy thread.

Important behavior:

```text
policy thread:
  servo_state = servo_io.get_cached_servo_state()
  action = policy(servo_state, imu, footswitches)
  servo_io.submit_targets(action)

servo worker thread:
  if latest target pending:
    write latest target
  else:
    read next scheduled servo
    update that joint cache
```

Reads still block writes while a read transaction is in flight. For raw HTD
servo reads, one servo `POS_READ` should be the atomic IO unit. Read "groups"
should be scheduling concepts, not indivisible bus transactions.

Example:

```text
read schedule:
  left_leg group -> enqueue servo IDs [1,2,3,4,9]
  right_leg group -> enqueue servo IDs [5,6,7,8,10]
  torso_arms group -> enqueue servo IDs [...]

worker read step:
  sid = read_scheduler.next_servo_id()
  pos = raw_bus.read_position(sid)
  cache.update(sid, pos)
```

This bounds write delay to one servo read or one servo read timeout instead of a
whole leg or torso group read. Based on measured raw TTL timing, this is the
simplest way to keep worst-case write latency small.

Target queue semantics should be "latest wins":

```text
queue depth: 1 target vector
if a newer target arrives before the worker writes the previous target:
  replace the older target
```

This prevents stale commands from being replayed after the loop is already late.

Configuration ownership:

```python
@dataclass(frozen=True)
class ServoReadGroup:
    name: str
    servo_ids: tuple[int, ...]
    retry_cache_age_s: float | None = None
    max_cache_age_s: float | None = None

@dataclass(frozen=True)
class ServoIOWorkerConfig:
    servo_ids: tuple[int, ...] = ()
    read_groups: tuple[ServoReadGroup, ...] = ()
    read_group_schedule: tuple[str, ...] = ()
    max_write_attempts: int = 2
    max_read_attempts: int = 2
    retry_cache_age_s: float = 0.08
    max_cache_age_s: float = 0.25
    idle_sleep_s: float = 0.0005
    stale_log_period_s: float = 1.0
```

Suggested API:

```python
class ServoIOWorker:
    def __init__(
        self,
        raw_bus: RawServoBus,
        config: ServoIOWorkerConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self, *, timeout_s: float = 1.0) -> None: ...

    def submit_targets_units(
        self,
        positions_by_servo_id: dict[int, int],
        *,
        move_time_ms: int,
    ) -> None: ...

    def get_cached_servo_state(self) -> CachedServoState: ...
    def get_metrics(self) -> ServoIOMetrics: ...
    def close(self) -> None: ...
```

`submit_targets_units()` accepts servo electrical units, not radians. Joint
calibration belongs in the calibrated actuator layer above the worker.

Example construction:

```python
servo_io = ServoIOWorker(
    raw_bus=raw_bus,
    config=ServoIOWorkerConfig(
        servo_ids=tuple(servo_ids.values()),
        read_groups=read_groups,
        read_group_schedule=("left_leg", "right_leg", "left_leg", "right_leg", "torso_arms"),
    ),
)
```

## Retry Policy

Retries must not turn a transient bad packet into a missed control deadline.

### Writes

Writes are safety-critical because they drive the robot. They should have higher
priority than reads, but still use latest-target semantics.

Suggested first policy:

```text
write transaction:
  max attempts: 2 total (initial attempt + 1 retry)
  retry sleep: 0 s
  attempt latest target once
  if pyserial write/flush raises immediately and deadline budget remains:
    retry immediately using the newest pending target
  if retry does not fit within the write deadline:
    log/print write error
    record write failure and let safety policy decide abort
```

Do not queue multiple old target vectors for retry. If a new target arrives,
replace the old pending target. A retry should never sleep before trying again;
sleeping spends control-loop budget without improving an HTD write that has no
acknowledgement packet.

If the serial write partially succeeds, the runtime usually cannot know. Treat
the transaction as failed only when pyserial raises. Do not retry based on servo
position mismatch in the same control step; use later position feedback and
safety limits to detect execution problems.

Recommended metrics:

```text
write_attempt_count
write_retry_count
write_failure_count
write_target_replaced_count
write_latency_ms avg/p50/p95/max
write_deadline_miss_count
```

Open point: raw HTD `MOVE_TIME_WRITE` has no response packet, so successful
`write()` only means the command was placed on the serial bus. It does not prove
the servo executed it. Execution is verified indirectly through later position
reads and cache age/error metrics.

### Reads

Reads are feedback updates. A failed read should not block the next write for a
long retry loop.

Suggested first policy:

```text
read transaction:
  read one servo with a bounded timeout
  if timeout/checksum/parse failure:
    mark that servo read failed
    leave cached position unchanged
    if cache age exceeds retry threshold:
      put that servo back into the read queue with retry priority
    if this servo has exceeded max read retry attempts:
      log/print read error
    move on to the next worker iteration, checking writes first
```

No immediate retry inside the same read transaction. The retry is queued as
another one-servo read item, so every retry still goes through the normal worker
priority check:

```text
worker loop:
  if latest target pending:
    write latest target
  else:
    read one servo from retry queue or normal schedule
```

This keeps write latency bounded by one read timeout while still allowing urgent
feedback recovery.

Suggested first thresholds:

```text
global retry cache age:           0.08 s
global max cache age:             0.25 s
per-group override:               ServoReadGroup retry/max age fields
max read attempts:                2 total per stale event
read timeout:                     RawServoBusConfig.response_timeout_s
```

The worker keeps at most one pending retry item per servo. That prevents one
broken servo from starving the rest of the read schedule. If the same servo
keeps failing, cache age and read failure counters should eventually trigger a
warning or abort instead of creating an unbounded retry loop.

Recommended metrics:

```text
read_attempt_count
read_success_count
read_timeout_count
read_checksum_error_count
read_parse_error_count
read_fail_count_by_servo
cache_age_ms_by_servo
cache_age_ms_by_group
```

### Error Reporting

The worker should emit explicit errors for final failures, not only metrics.
Use the runtime logger when available and print to console for hardware runs.
Repeated errors should be rate-limited, but the first final failure must be
visible.

Write error after max attempts:

```text
ERROR servo write failed after 2 attempts:
  target_age_s=...
  write_latency_s=...
  deadline_s=...
  exception=...
```

Read error after max attempts:

```text
ERROR servo read failed after max retry attempts:
  servo_id=...
  joint_name=...
  attempts=...
  cache_age_s=...
  last_error_type=timeout|checksum|parse|serial
  last_error=...
```

Cache expiration error:

```text
ERROR servo cache expired:
  servo_id=...
  joint_name=...
  group=...
  cache_age_s=...
  max_cache_age_s=...
  read_fail_count=...
  last_read_error=...
```

Cache expiration is separate from read failure. A read may fail without being
immediately unsafe if the cached value is still young. Once the configured cache
age expires, the runtime must log/print an error and follow the safety policy
for that joint.

## Layer 4: Calibrated Actuator API

This is the runtime-facing actuator layer.

Responsibilities:

- Map policy actuator names to servo IDs.
- Convert radians to servo electrical units.
- Convert servo electrical units to radians.
- Apply servo direction signs, offsets, centers, and limits.
- Provide policy-order full vectors.

Suggested API:

```python
class HiwonderCachedActuators:
    def __init__(
        self,
        actuator_names: Sequence[str],
        servo_ids: dict[str, int],
        default_move_time_ms: int | None,
        joint_servo_offset_units: dict[str, int],
        servo_io: ServoIOWorker,
        joint_motor_unit_directions: dict[str, float] | None = None,
        joint_angle_at_zero_unit_deg: dict[str, float] | None = None,
        servo_model: ServoModel | None = None,
    ) -> None: ...

    def set_targets_rad(self, targets_rad: np.ndarray, *, move_time_ms: int | None = None) -> None: ...
    def get_positions_rad(self) -> np.ndarray | None: ...
    def estimate_velocities_rad_s(self, dt: float) -> np.ndarray: ...
    def disable(self) -> None: ...
    def close(self) -> None: ...
```

For the background-worker design, `get_positions_rad()` should return cached
state and should not perform serial IO directly.

Example construction:

```python
actuators = HiwonderCachedActuators(
    actuator_names=policy_actuator_names,
    servo_ids=servo_ids,
    servo_io=servo_io,
    default_move_time_ms=20,
    joint_servo_offset_units=offsets,
    joint_motor_unit_directions=motor_signs,
    joint_angle_at_zero_unit_deg=joint_zero_deg,
)
```

## Cached Servo State Data Model

The worker should publish one immutable cached servo state. This is not a
serial-bus read; it returns the latest cached positions, velocities, ages, and
diagnostics populated by the worker thread.

```python
@dataclass(frozen=True)
class CachedServoState:
    servo_ids: tuple[int, ...]
    position_units: np.ndarray
    velocity_units_s: np.ndarray
    position_age_s: np.ndarray
    read_fail_count: np.ndarray
    last_update_time_s: np.ndarray
    last_read_group: str | None
    last_error: str | None
```

Velocity must be computed from actual per-joint update times:

```text
velocity_units_s[j] = (
  position_units[j] - prev_position_units[j]
) / (update_time[j] - prev_update_time[j])
```

It should not assume every joint updates at the 20 ms control period.

## Ownership Rules

- Only the servo worker touches `RawServoBus`.
- Only `RawServoBus` touches `SerialTransport`.
- Policy/runtime code never writes or reads serial bytes.
- Setup/debug scripts may use `RawServoBus` directly, but not while runtime is
  running.

This avoids corrupting the half-duplex bus with overlapping read/write
transactions.

## Safety Rules

Startup:

- Perform an initial full read before enabling walking.
- Fail startup if any required policy joint is missing.
- Publish initial cache ages as zero only after real reads.

Runtime aborts:

- Leg joint cache age exceeds the configured limit.
- Write latency or write miss streak exceeds the configured limit.
- Serial worker dies.
- Repeated read failures leave critical joints stale.
- Any required joint cache expires past its configured max age; log/print the
  cache expiration error before aborting.

Runtime warnings:

- Partial read success.
- Increasing cache age below abort limit.
- Read p95 approaching the available loop budget.
- Read retry queued because cache age crossed the retry threshold.
- Write retry attempted after an immediate serial write/flush exception.

## Metrics

Minimum metrics to print in policy summary:

```text
servo_worker_hz
servo_write_count
servo_write_latency_ms avg/p50/p95/max
servo_read_count
servo_read_latency_ms avg/p50/p95/max
servo_read_success_rate
servo_cache_age_ms max by group
servo_target_replaced_count
servo_write_deadline_miss_count
```

These metrics decide whether raw TTL at 115200 is enough or whether the hardware
path still needs to change.

## Open Design Questions

1. Should writes use individual raw servo commands or a controller-level
   multi-servo move path if available?
2. Walking runtime uses only the raw TTL debug-board path; the old LSC
   controller path is legacy diagnostics only.
3. What are the first cache-age limits for legs, arms, and wrists?
4. Should wrist joints be write-only/cached-lower-rate if the bus is still tight?
5. Do we need direct current/load/voltage reads for safety, or only position?

## Proposed First Implementation

Start with the simplest path that tests the design:

Implemented first slice:

1. Pure packet helpers and `RawServoBus`.
2. Tests for packet build/parse/checksum using known hardware examples.
3. `ServoIOWorker` with latest-wins writes and one-servo-at-a-time reads.
4. `HiwonderCachedActuators` adapter that converts worker servo units to
   calibrated runtime radians.

Still open:

1. Update timing/setup scripts to reuse `RawServoBus` instead of duplicate packet
   helpers.
2. Wire the cached actuator path behind an explicit runtime config flag.
3. Print servo IO summary metrics at policy exit.

This keeps the runtime behavior reviewable and avoids mixing protocol work with
policy-control changes in one step.
