# Servo API Design

This document proposes the WildRobot servo API layers for the HTD/Hiwonder
servo runtime. It is a design review document, not an implementation plan that
has already been accepted.

## Summary

Yes, it makes sense to add a raw servo command layer.

The current runtime has two effective layers:

```text
policy/runtime layer
  -> joint-space Actuators API
     -> serial controller implementation
```

That is too compressed for the next runtime design. The raw HTD command protocol
should be separated from scheduling, cache policy, calibration, and policy IO.

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

Sketch:

```python
class SerialTransport:
    def write(self, packet: bytes) -> None: ...
    def read_until(self, deadline_s: float) -> bytes: ...
    def close(self) -> None: ...
```

For a raw TTL debug board connected by USB, pyserial controls the USB serial
device. For a GPIO half-duplex UART path, the transport may also need direction
control, but the upper layers should not change.

## Layer 2: Raw Servo Command API

This layer wraps the raw HTD/Hiwonder `55 55` bus-servo command protocol.

Responsibilities:

- Build packets.
- Validate checksums.
- Parse responses.
- Provide typed command methods.
- Own request/response timeouts for a single transaction.

It should not know actuator names, joint signs, home poses, or policy timing.

Suggested API:

```python
class RawServoBus:
    def ping_or_read_id(self, target_id: int = 0xFE) -> int | None: ...
    def read_id(self, target_id: int = 0xFE) -> int | None: ...
    def write_id(self, old_id: int, new_id: int) -> bool: ...

    def read_position(self, servo_id: int) -> int | None: ...
    def read_positions(self, servo_ids: Sequence[int]) -> dict[int, int]: ...

    def move_time_write(self, servo_id: int, position: int, time_ms: int) -> None: ...
    def move_time_wait_write(self, servo_id: int, position: int, time_ms: int) -> None: ...
    def move_start(self, servo_id: int) -> None: ...
    def move_stop(self, servo_id: int) -> None: ...

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

## Layer 3: Servo IO Worker

This layer owns runtime bus scheduling.

Responsibilities:

- Be the only thread that calls `RawServoBus`.
- Accept latest 21D target commands from the policy thread.
- Prioritize writes over reads.
- Poll scheduled servo read groups when no write is pending.
- Update full joint-state cache.
- Publish a non-blocking snapshot for the policy thread.

Important behavior:

```text
policy thread:
  snapshot = servo_io.snapshot()
  action = policy(snapshot, imu, footswitches)
  servo_io.submit_targets(action)

servo worker thread:
  if latest target pending:
    write latest target
  else:
    read next scheduled group
    update cache
```

Reads still block writes while a read transaction is in flight. The worker
controls this by keeping read groups small and timeouts bounded.

Target queue semantics should be "latest wins":

```text
queue depth: 1 target vector
if a newer target arrives before the worker writes the previous target:
  replace the older target
```

This prevents stale commands from being replayed after the loop is already late.

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
class Actuators:
    def set_targets_rad(self, targets_rad: np.ndarray, *, move_time_ms: int | None = None) -> None: ...
    def get_positions_rad(self) -> np.ndarray: ...
    def get_velocities_rad_s(self) -> np.ndarray: ...
    def get_state_snapshot(self) -> ServoStateSnapshot: ...
    def disable(self) -> None: ...
    def close(self) -> None: ...
```

For the background-worker design, `get_positions_rad()` should return cached
state and should not perform serial IO directly.

## Snapshot Data Model

The worker should publish one immutable snapshot:

```python
@dataclass(frozen=True)
class ServoStateSnapshot:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    position_age_s: np.ndarray
    last_update_time_s: np.ndarray
    read_fail_count: np.ndarray
    write_latency_s: float | None
    read_latency_s: float | None
    last_read_group: str | None
    last_error: str | None
```

Velocity must be computed from actual per-joint update times:

```text
vel[j] = (pos[j] - prev_pos[j]) / (update_time[j] - prev_update_time[j])
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

Runtime warnings:

- Partial read success.
- Increasing cache age below abort limit.
- Read p95 approaching the available loop budget.

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
2. Should the runtime support both the old LSC controller and raw TTL debug-board
   path behind one interface, or should walking only use the raw TTL path?
3. What are the first cache-age limits for legs, arms, and wrists?
4. Should wrist joints be write-only/cached-lower-rate if the bus is still tight?
5. Do we need direct current/load/voltage reads for safety, or only position?

## Proposed First Implementation

Start with the simplest path that tests the design:

1. Implement pure packet helpers and `RawServoBus`.
2. Add tests for packet build/parse/checksum using known examples.
3. Implement a timing probe that uses `RawServoBus` instead of hand-built packet
   code.
4. Implement `ServoIOWorker` with a small fixed read schedule.
5. Wire `HiwonderBoardActuators` to cached snapshots behind an explicit config
   flag.
6. Print servo IO summary metrics at policy exit.

This keeps the runtime behavior reviewable and avoids mixing protocol work with
policy-control changes in one step.
