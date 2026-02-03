# Sim2Real Policy Contract (Training ↔ Runtime)

Status: Draft (for review)  
Scope: Stage 1 (PPO-only walking, task rewards, no AMP) with a forward-compatible design for later stages.

## 1) Problem Statement

Sim2real failures in legged RL are often *interface* failures, not “RL failures”:
- Observation semantics drift (frames, units, normalization, history terms)
- Action semantics drift (normalized vs physical, joint range mapping, mirroring, filtering)
- Asset/joint order drift (actuator ordering between MJCF, training, runtime)
- Hidden runtime changes (different filters, clamps, delays) invalidate a trained policy

In this repo, the policy-facing contract now lives in `policy_contract/`, and both training + runtime import it. The remaining risk is any ad-hoc runtime logic outside the contract (filters, clamps, timing) that can still drift.

Long-term, keep training with a MuJoCo-specific **sim adapter** (not a second copy of the contract) and progressively shrink sim-only helpers (legacy CAL usage) to reward/termination/metrics only.

## 2) Goals / Non‑Goals

### Goals
- **Single source of truth** for observation layout and action mapping used by both training and hardware runtime.
- **Versioned, enforceable policy interface** (“policy contract”) that fails fast on mismatch.
- **Policy bundle artifact**: ONNX + spec + robot metadata so runtime can validate without importing training code.
- **Minimal runtime dependencies** (Raspberry-friendly): NumPy + ONNX Runtime; no JAX/Flax.
- **Deterministic parity tests**: contract tests with golden vectors to prevent regressions.

### Non‑Goals (Stage 1)
- Full state estimation (base linvel estimator) on hardware.
- Complex safety (MPC fallback, recovery controllers) beyond basic clamps and e-stop.
- Latency/real-time OS guarantees (we’ll still measure and budget timing).

## 3) Current Repo Reality (what we must match)

### Training semantics (canonical)
- Actor observations are built via `policy_contract` (shared JAX/NumPy) from a `Signals` struct.
- Joint positions/velocities in observations are **normalized**:
  - `joint_pos = cal.get_joint_positions(..., normalize=True)` → ~[-1, 1] based on joint ranges
  - `joint_vel = cal.get_joint_velocities(..., normalize=True)` → clipped/normalized by velocity limits
- Action semantics:
  - Policy output is normalized `action ∈ [-1, 1]`
- Environment converts with `cal.action_to_ctrl(action)`:
    - clips action
    - applies `mirror_sign` symmetry correction
    - maps to **physical control target angles (radians)** using joint centers/spans
- Optional action filtering:
  - `filtered_action = alpha * prev_action + (1 - alpha) * raw_action`
- Linear velocity (linvel) is treated as **privileged-only** (reward/metrics/critic). It is not part of the actor observation contract.
- IMU sensors in sim (available in MJCF):
  - `chest_imu_quat` (MuJoCo `framequat`) and `chest_imu_gyro` (MuJoCo `gyro`) exist in `assets/v1/wildrobot.xml` (or the selected variant MJCF).
  - **Current training code** uses `chest_imu_quat` for gravity, and uses the gyro sensor path for actor-facing angular velocity.
  - **Migration goal** is to use the gyro sensor path for actor-facing angular velocity to match hardware semantics (BNO085 gyro) and reduce sim↔real drift.

### Runtime today (risk)
Runtime entrypoint (e.g. `runtime/wr_runtime/control/run_policy.py`) currently:
- builds observations using **physical** joint pos/vel and body-frame gyro
- interprets actions as `targets = action_scale_rad * action` (delta around zero pose)

These are not the same semantics the policy was trained on, even if `obs_dim/action_dim` match.

## 4) Proposed Architecture (industry pattern)

### Layering (keep “learning” separate from “running”)

1) **Robot Spec / Policy Contract** (shared, versioned)
   - what the policy *expects* (obs/action schema + normalization + frames)
2) **Adapters**
   - `SimIO` (MuJoCo/JAX) and `HardwareIO` (IMU/servos) both implement the same high-level IO interface
3) **Policy Inference Backend**
   - ONNX Runtime (and later optional TorchScript/TensorRT)
4) **Safety + Supervisory Control**
   - clamps, rate limits, e-stop, watchdog, logging

Training owns learning. Runtime owns hardware integration. The contract is the bridge.

### Architecture diagram (data flow)

```
                          TRAINING (Sim) / EXPORT
┌───────────────────────────────────────────────────────────────────────┐
│  MuJoCo/JAX Env                                                      │
│  ┌───────────────┐   obs (contract)   ┌───────────────────────────┐  │
│  │     SimIO     │ ─────────────────▶ │  RL Policy (PPO network)   │  │
│  │ (mjx + sensors│                    │  (trained in playground)    │  │
│  └───────────────┘   action [-1,1]    └───────────────────────────┘  │
│             ▲                 │                     │                 │
│             │                 ▼                     ▼                 │
│        ctrl (rad)     Policy Contract (shared)   Export Pipeline       │
│             │     (frames + calib + layout)    (ONNX + policy_spec)    │
└─────────────┼───────────────────────────────────────────┬─────────────┘
              │                                           │
              │                           Policy Bundle Artifact
              │                   (policy.onnx + policy_spec.json + ...)
              │                                           │
              ▼                                           ▼

                             RUNTIME (Hardware)
┌───────────────────────────────────────────────────────────────────────┐
│  Sensors + Actuators                                                  │
│  ┌───────────────┐  raw sensors   ┌───────────────────────────────┐  │
│  │   HardwareIO   │ ────────────▶ │   PolicyRunner / ControlLoop   │  │
│  │ (IMU/servos/   │               │                               │  │
│  │  foot switches)│ ◀──────────── │  1) build obs (contract)      │  │
│  └───────────────┘   ctrl (rad)   │  2) ONNX inference            │  │
│                                   │  3) safety filter/arbitration │  │
│                                   │  4) action→ctrl (contract)    │  │
│                                   └───────────────┬───────────────┘  │
│                                                   │                  │
│                                   optional fallback/recovery          │
│                              ┌───────────────────────────────────┐   │
│                              │ Safety Supervisor + MPC/Recovery   │   │
│                              │ (triggered on tilt/limits/timeouts)│   │
│                              └───────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

### Call graphs (who calls what)

The core idea is: **IO layers** produce/consume raw physical signals; **`policy_contract`** owns the policy-facing semantics.

#### Training (MuJoCo/JAX) call graph

```
training/train.py
  └─ training/envs/wildrobot_env.py
     ├─ step/reset/reward/termination (env logic)
     ├─ SimIO (MuJoCo/MJX → raw signals)
     │   └─ training/sim_adapter/mjx_signals.py
     │        └─ read_signals(mjx.Data, mjx.Model, ...) -> Signals
     ├─ policy_contract (raw signals → obs; action → ctrl)
     │   ├─ policy_contract/jax/obs.py
     │   │    └─ build_observation(signal_packet, prev_action, velocity_cmd, ...) -> obs
     │   └─ policy_contract/jax/calib.py
     │        └─ action_to_ctrl(spec, action_normalized) -> ctrl_targets_rad
     └─ mjx: apply ctrl_targets_rad and step physics
```

#### Runtime (hardware) call graph

```
runtime/wr_runtime/control/run_policy.py
  ├─ HardwareRobotIO (hardware IO → Signals; ctrl → hardware commands)
  │   └─ runtime/wr_runtime/hardware/robot_io.py
  │        ├─ read() -> Signals (quat_xyzw, gyro_rad_s, joint_pos/vel, foot switches, ...)
  │        └─ write_ctrl(ctrl_targets_rad) -> sends to actuators (drivers)
  ├─ policy_contract (raw signals → obs; action → ctrl)
  │   ├─ policy_contract/numpy/obs.py
  │   │    └─ build_observation(...) -> obs
  │   └─ policy_contract/numpy/calib.py
  │        └─ action_to_ctrl(spec, action) -> ctrl_targets_rad
  └─ runtime/wr_runtime/inference/onnx_policy.py
       └─ predict(obs) -> action_normalized
```

IMU data path (runtime): `BNO085IMU` (body-frame gyro + quat with `upside_down`/`axis_map` applied) → `HardwareRobotIO.read()` → `policy_contract.numpy.obs.build_observation(...)`. Configure via `bno085` in the runtime JSON (`i2c_address`, `upside_down`, `axis_map`, `suppress_debug`, `i2c_frequency_hz`, `init_retries`). A pre-loop sanity check can verify `|quat| ≈ 1`, `gravity_local.z ≈ -1`, and small gyro before any motors move.

## 5) Policy Contract: What it Contains

Think of this as the “ABI” of the policy.

### 5.0 Frames and conventions (must be consistent everywhere)

This contract assumes MuJoCo-style axes:
- `+X` forward
- `+Y` left (so “right” is `-Y`)
- `+Z` up

We use these frames:
- **world (`W`)**: inertial frame, fixed axes (`+Z` up)
- **local/body (`B`)**: body-attached frame (IMU/base frame); rotates with full orientation (yaw + pitch + roll)
- **heading_local (`H`)**: yaw-only frame; rotates with yaw, but stays level:
  - `H.z` is aligned with world up (`W.z`)
  - `H.x` is the robot’s heading direction projected onto the ground plane
  - `H.y` completes a right-handed basis in the ground plane

“Remove yaw” means: rotate a world-frame vector by the inverse yaw rotation:
- `v_H = Rz(-yaw) * v_W`

Notes:
- `gravity_local` is expressed in the local/body frame to capture roll/pitch tilt.
- `angvel_heading_local` is expressed in heading-local to be yaw-invariant and not tilt with the body.
- Your IMU quaternion convention (body→world vs world→body) must be specified in code once (e.g., in `policy_contract/*/frames.py`) and then reused everywhere.

### 5.1 Contract versioning
- `contract_name`: `"wildrobot_policy"`
- `contract_version`: semantic version (e.g., `"1.0.0"`)
- `git_commit` / `training_run_id` (optional)
- `created_at` timestamp

Policy contract version bumps:
- **PATCH**: metadata only, no semantic change
- **MINOR**: backward-compatible additive fields
- **MAJOR**: any semantic change (layout reorder, new normalization, new frames, etc.)

### 5.2 Robot specification (identity + ordering)
Source of truth is the variant `robot_config.yaml` (e.g., `assets/v1/robot_config.yaml`).
Contract records:
- `actuator_names` in **policy action order**
- per-actuator:
  - `range_min/max` (rad)
  - `mirror_sign`
  - `max_velocity` (rad/s)
- `observation_breakdown` (named counts) and computed `obs_dim`

Runtime must validate:
- MJCF actuator order matches `actuator_names`
- ONNX IO dims match `obs_dim/action_dim`

### 5.3 Observation specification (schema + semantics)
Each observation component must declare:
- `name` (stable)
- `size`
- `dtype` (float32)
- `frame` (`local`, `heading_local`, `world`) where applicable
- `units` (`rad`, `rad_s`, `m_s`, `normalized`)
- `normalization` method where applicable

For Stage 1, the canonical layout is:
- gravity_local (3) (unit-ish vector)
- angvel_heading_local (3)
- joint_pos_normalized (N)
- joint_vel_normalized (N)
- foot_switches (4)
- prev_action (N)
- velocity_cmd (1)
- padding (1)

### 5.4 Action specification (schema + mapping)
The policy output is:
- `action_normalized`: `[-1, 1]` per actuator, order=`actuator_names`

The contract must define the mapping to **control targets**:
- `ctrl_type = position_target_rad`
- `action_to_ctrl`:
  - `action_clipped`
  - `corrected = action_clipped * mirror_sign`
  - `ctrl = corrected * ctrl_span + ctrl_center`

Optional runtime shaping (must be declared if used):
- action low-pass: alpha
- action rate limiting
- ctrl clamping to joint range

**Policy on `alpha` (recommended for this repo):**
- Treat `alpha` as a *training-selected hyperparameter* that is **exported** into the bundle (`policy_spec.json`) and honored by runtime.
- Runtime should not override `alpha` ad-hoc; changing it changes inference-time semantics and invalidates parity.
- Practically: training configs can sweep/tune `env.action_filter_alpha`; the exporter writes the chosen value into `spec.action.postprocess_params.alpha`.

### 5.5 `policy_spec.json` interface (concrete schema)

`policy_spec.json` is the machine-readable contract shipped with the model bundle. It must be sufficient for runtime to:
- validate compatibility (name/version/dims/order)
- build observations deterministically
- interpret model outputs and map to control targets deterministically

#### Required top-level fields (v1)

- `contract_name: str` (e.g., `"wildrobot_policy"`)
- `contract_version: str` (semver, e.g., `"1.0.0"`)
- `spec_version: int` (schema version, start with `1`)
- `model: {format, input_name, output_name, obs_dim, action_dim, dtype}`
- `robot: {robot_name, actuator_names, joints}`
- `observation: {dtype, layout}`
- `action: {dtype, bounds, mapping}`

Where:
- `robot.joints` maps each actuator/joint name to the fields needed for normalization and action→ctrl mapping:
  - `range_min_rad`, `range_max_rad`, `mirror_sign`, `max_velocity_rad_s`
- `observation.layout` is an ordered list of fields; the concatenation order is the canonical obs order.
- `action.mapping` declares the *meaning* of policy outputs and how to map to control targets.

#### Example `policy_spec.json` (Stage 1, PPO walking)

This is an example shape; values shown are illustrative.

```json
{
  "contract_name": "wildrobot_policy",
  "contract_version": "1.0.0",
  "spec_version": 1,

  "model": {
    "format": "onnx",
    "input_name": "observation",
    "output_name": "action",
    "dtype": "float32",
    "obs_dim": 39,
    "action_dim": 8
  },

  "robot": {
    "robot_name": "WildRobotDev",
    "actuator_names": [
      "left_hip_pitch",
      "left_hip_roll",
      "left_knee_pitch",
      "left_ankle_pitch",
      "right_hip_pitch",
      "right_hip_roll",
      "right_knee_pitch",
      "right_ankle_pitch"
    ],
    "joints": {
      "left_hip_pitch": {
        "range_min_rad": -0.087268,
        "range_max_rad": 1.570794,
        "mirror_sign": 1.0,
        "max_velocity_rad_s": 10.0
      },
      "right_hip_pitch": {
        "range_min_rad": -1.570797,
        "range_max_rad": 0.087266,
        "mirror_sign": -1.0,
        "max_velocity_rad_s": 10.0
      }
    }
  },

  "observation": {
    "dtype": "float32",
    "layout_id": "wr_obs_v1",
    "layout": [
      {"name": "gravity_local", "size": 3, "frame": "local", "units": "unit_vector"},
      {"name": "angvel_heading_local", "size": 3, "frame": "heading_local", "units": "rad_s"},
      {"name": "joint_pos_normalized", "size": 8, "units": "normalized_-1_1"},
      {"name": "joint_vel_normalized", "size": 8, "units": "normalized_-1_1"},
      {"name": "foot_switches", "size": 4, "units": "bool_as_float"},
      {"name": "prev_action", "size": 8, "units": "normalized_-1_1"},
      {"name": "velocity_cmd", "size": 1, "units": "m_s"},
      {"name": "padding", "size": 1, "units": "unused"}
    ]
  },

  "action": {
    "dtype": "float32",
    "bounds": {"min": -1.0, "max": 1.0},
    "postprocess_id": "lowpass_v1",
    "postprocess_params": {"alpha": 0.7},
    "mapping_id": "pos_target_rad_v1"
  },

  "provenance": {
    "created_at": "2026-01-04T00:00:00Z",
    "git_commit": "<optional>",
    "training_config": "<optional>",
    "source_checkpoint": {
      "format": "pkl",
      "path": "<optional>"
    }
  }
}
```

Notes:
- `model.format` refers to the **deployed inference artifact** inside the bundle (Stage 1 uses ONNX). Training can use a different format (e.g., a Brax PPO `.pkl` checkpoint); that belongs under `provenance.source_checkpoint`.
- Only a subset of joints is shown under `robot.joints` above; in practice it must include **all** `actuator_names`.
- Keep “behavior knobs” minimal and enumerated: `observation.layout_id`, `action.postprocess_id`, `action.mapping_id`.
- `action.mapping_id = "pos_target_rad_v1"` means: clip → mirror_sign → scale to joint center/span in radians.
- For `wr_obs_v1`, `prev_action` is required by definition and stored in `PolicyState` (initialized to zeros unless the contract version introduces another init mode).

#### Deterministic mapping definitions (norms)

To remove ambiguity, the spec implies these reference formulas:

- `center = (range_min_rad + range_max_rad) / 2`
- `span   = (range_max_rad - range_min_rad) / 2`
- `action_clipped = clip(action, -1, 1)`
- `corrected = action_clipped * mirror_sign`
- `ctrl_target_rad = corrected * span + center`
- `joint_pos_normalized = (joint_pos_rad - center) / (span + 1e-6)`
- `joint_vel_normalized = clip(joint_vel_rad_s / max_velocity_rad_s, -1, 1)`

### 5.6 Python-facing spec interface (how code consumes it)

The shared package (`policy_contract/spec.py`) should expose a small, stable API:
- `PolicySpec.from_json(path_or_str) -> PolicySpec`
- `PolicySpec.validate() -> None` (raises on missing/invalid fields)
- `PolicySpec.obs_dim` / `PolicySpec.action_dim` (derived from layout/dims)
- `PolicySpec.actuator_names` (canonical order)

Runtime should load a **bundle** (folder or zip) that contains at least:
- `policy.onnx`
- `policy_spec.json`
- optional `robot_config.yaml` snapshot + `checksums.json`

Recommended loader shape:
- `PolicyBundle.load(bundle_path) -> (spec: PolicySpec, model_path: Path, extra_paths: dict[str, Path])`

## 5.7 `policy_contract` code interface (runtime + training)

This section defines the **code-level API** that both training (JAX) and runtime (NumPy) use.

Design goals:
- Keep the surface area small and stable.
- Make state explicit (`prev_action`, filters).
- Avoid “random config”: behavior is implemented in code and selected via enum-like spec fields.
- Make it hard to accidentally violate the contract (validate early, validate often).

### 5.7.1 Core datatypes

- `PolicySpec` (loaded from `policy_spec.json`)
  - contract metadata + robot/joint specs + obs/action schema
- `PolicyState`
  - stores state required by the contract (Stage 1: `prev_action`)
- `Signals` (raw physical signals)
  - the minimum set of sensor/estimator values needed to construct the observation

Suggested shapes:
- `actuator_names` defines canonical order for all per-joint vectors.
- All arrays are float32; actions are 1D `(action_dim,)`, observations are 1D `(obs_dim,)`.

### 5.7.2 PolicyState API

`PolicyState` is owned by the runner (training loop or runtime control loop).

```python
class PolicyState:
    prev_action: Array  # shape (action_dim,), dtype float32

    @classmethod
    def init(cls, spec: PolicySpec, *, mode: str = "zeros") -> "PolicyState":
        ...
```

Initialization modes are deliberately limited:
- `"zeros"` (safe default)
- `"from_current_pose"` (optional; requires a `ctrl_to_action` helper and current joint positions)

### 5.7.3 Observation builder API

```python
def build_observation(
    *,
    spec: PolicySpec,
    state: PolicyState,
    signals: Signals,
    velocity_cmd: float,
) -> Array:
    """Return obs float32, shape (spec.model.obs_dim,)."""
```

Behavior is determined by spec enums, not ad-hoc runtime logic:
- If `observation.layout_id == "wr_obs_v1"`, the builder requires `state.prev_action` and includes it in the `prev_action` slice.

Backends:
- Training uses `policy_contract/jax/obs.py:build_observation`
- Runtime uses `policy_contract/numpy/obs.py:build_observation`

### 5.7.4 Action postprocess + mapping API

```python
def postprocess_action(
    *,
    spec: PolicySpec,
    state: PolicyState,
    action_raw: Array,
) -> Array:
    """Return filtered action float32, shape (action_dim,); updates state as needed."""


def action_to_ctrl(
    *,
    spec: PolicySpec,
    action: Array,
) -> Array:
    """Return ctrl_targets_rad float32, shape (action_dim,)."""
```

Notes:
- `postprocess_action` is where declared filters live (e.g., `lowpass_v1` with `alpha`).
- `action_to_ctrl` implements `pos_target_rad_v1` (and future mappings).
- For Stage 1, `postprocess_action` should update `state.prev_action = action_filtered` after filtering.

### 5.7.5 Validation API (fail fast)

```python
def validate_spec(spec: PolicySpec) -> None:
    ...


def validate_runtime_compat(
    *,
    spec: PolicySpec,
    mjcf_actuator_names: list[str],
    onnx_input_dim: int,
    onnx_output_dim: int,
) -> None:
    ...
```

Minimum required checks:
- `spec.robot.actuator_names == mjcf_actuator_names`
- `spec.model.obs_dim == onnx_input_dim`
- `spec.model.action_dim == onnx_output_dim`
- supported enums/ids for `observation.layout_id`, `action.postprocess_id`, `action.mapping_id`

### 5.7.6 Signals interface (IO boundary)

The IO layer (`SimIO` / `HardwareIO`) should produce a `Signals` object with raw physical units.

Stage 1 recommended minimal signals:
- `quat_xyzw` (4,) float32
- `gyro_rad_s` (3,) float32
- `joint_pos_rad` (N,) float32
- `joint_vel_rad_s` (N,) float32 (or zeros if not available)
- `foot_switches` (4,) float32/bool

Linear velocity estimates can exist for logging or privileged training signals, but they are intentionally **outside** the actor policy contract.

## 5.8 SimIO / HardwareIO API (IO boundary)

`SimIO` and `HardwareIO` are responsible for interfacing with the outside world (simulator or robot hardware).
They do **not** implement policy semantics (layout, normalization, filters, action→ctrl). They only:
- produce raw physical signals (`Signals`)
- apply control targets (`ctrl_targets_rad`) produced by `policy_contract`

### 5.8.1 Minimal interface

```python
class RobotIO(Protocol):
    actuator_names: list[str]  # canonical order for all per-joint vectors
    control_dt: float          # desired control period (seconds)

    def read(self) -> Signals:
        """Return latest raw signals in physical units."""

    def write_ctrl(self, ctrl_targets_rad: ArrayLike) -> None:
        """Apply control targets (radians) in actuator_names order."""

    def close(self) -> None:
        """Release resources; safe to call multiple times."""
```

### 5.8.2 `Signals` payload (Stage 1 minimal)

```python
@dataclass(frozen=True)
class Signals:
    timestamp_s: float
    quat_xyzw: ArrayLike        # (4,)
    gyro_rad_s: ArrayLike       # (3,)
    joint_pos_rad: ArrayLike    # (N,)
    joint_vel_rad_s: ArrayLike  # (N,) (zeros if unavailable)
    foot_switches: ArrayLike    # (4,) bool/float
```

Notes:
- `Signals` is intentionally raw/physical. `policy_contract` converts it to normalized observation features.
  - Linear velocity is privileged-only, so `Signals` does not need to carry linvel for Stage 1.

### 5.8.3 SimIO vs HardwareIO differences

- `SimIO` (training)
  - reads from MuJoCo/MJX state (e.g., `mjx.Data`)
  - applies `ctrl_targets_rad` back into sim control (functional update in JAX)
- `HardwareIO` (runtime)
  - reads from physical sensors (IMU, servo readback, GPIO switches)
  - converts `ctrl_targets_rad` into device commands (e.g., servo units) and applies calibration

### 5.8.4 HardwareIO internal components

`HardwareIO` is composed of internal driver components. The control loop only interacts with
the unified `RobotIO` interface (`read()` / `write_ctrl()`), while hardware-specific details
are encapsulated in internal components:

```
┌─────────────────────────────────────────────────────────────────┐
│  Control Loop (run_policy.py)                                   │
│    signals = robot_io.read()        # All sensors               │
│    obs = build_observation(signals) # policy_contract           │
│    action = policy.predict(obs)                                 │
│    ctrl = action_to_ctrl(action)    # policy_contract           │
│    robot_io.write_ctrl(ctrl)        # All actuators             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  HardwareRobotIO (implements RobotIO)                           │
│    - Composes: Actuators, IMU, FootSwitches                     │
│    - read() aggregates all sensor data into Signals             │
│    - write_ctrl() delegates to Actuators                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│ Actuators       │ │ IMU         │ │ FootSwitches    │
│ (Protocol)      │ │ (BNO085)    │ │ (GPIO)          │
└─────────────────┘ └─────────────┘ └─────────────────┘
```

#### Actuators interface (internal to HardwareIO)

The `Actuators` protocol abstracts hardware-specific actuator implementations (Hiwonder servos,
Dynamixel, etc.) from the `HardwareRobotIO` layer:

```python
class Actuators(Protocol):
    """Internal actuator interface used by HardwareRobotIO.

    All positions/velocities are in radians in actuator_names order.
    Hardware-specific details (servo units, IDs, protocols) are hidden.
    """

    @property
    def actuator_names(self) -> list[str]:
        """Canonical joint names in policy order."""
        ...

    @property
    def num_actuators(self) -> int:
        """Number of actuators."""
        ...

    def set_targets_rad(self, targets: ArrayLike) -> None:
        """Command position targets in radians.

        Args:
            targets: Shape (num_actuators,), in actuator_names order.
        """
        ...

    def get_positions_rad(self) -> Optional[np.ndarray]:
        """Read current positions in radians.

        Returns:
            Shape (num_actuators,) float32, or None if read fails.
        """
        ...

    def estimate_velocities_rad_s(self, dt: float) -> np.ndarray:
        """Estimate velocities via finite-difference.

        Args:
            dt: Time since last position reading (seconds).

        Returns:
            Shape (num_actuators,) float32. Zeros if unavailable.
        """
        ...

    def disable(self) -> None:
        """Disable torque / unload all actuators (safe state)."""
        ...

    def close(self) -> None:
        """Release hardware resources."""
        ...
```

#### Timing semantics (Hiwonder `time_ms`)

Some actuator transports (including the Hiwonder board protocol) require a per-command move duration (`time_ms`).
To keep this explicit and avoid hidden timing drift, pick one of the following patterns:

- **Preferred (explicit per-call):** extend the interface to accept a move duration:
  - `set_targets_rad(targets, *, move_time_ms: int) -> None`
  - `run_policy.py` passes `int(control_dt * 1000)` (or another explicitly chosen value).
- **Acceptable (implicit in actuator instance):** `Actuators` is constructed with `control_dt` and always uses `int(control_dt * 1000)` when
  sending commands.
  - If you choose this, treat `control_dt` as part of runtime compatibility validation (startup validator should fail if bundle/run config
    expects a different control rate).

#### Reliability: retry + fail-fast

Servo bus communication is the most failure-prone part of the runtime stack. Best practice is:

- **Retries live in the hardware layer** (inside the concrete `Actuators` implementation, not in `HardwareRobotIO`), so calibration tools,
  validators, and the policy loop all share the same reliability behavior.
- **Read retry**: `get_positions_rad()` should retry **2–3 times** on transient failures (timeout/CRC/serial exceptions) with a short backoff
  (e.g., 1–5 ms). If all retries fail, return `None` (or raise a typed exception) so the control loop can trip safety.
- **Write retry**: `set_targets_rad(...)` should retry **2–3 times** if the transport reports a failure (serial write exception, explicit NACK
  if available). If the protocol is fire-and-forget (no ACK), retries only apply to detected transport errors; add a separate **readback
  watchdog** (below).
- **Safety reaction**: the control loop should treat repeated read/write failures as a hard fault and immediately:
  - stop issuing new commands,
  - `disable()` actuators (unload torque),
  - and optionally trigger an e-stop latch / require operator intervention to resume.

This keeps `Signals` semantics clean (no “silent zeros”) and makes comms failures visible and actionable.

#### Readback failures (avoid “silent zeros”)

Do **not** mask actuator readback failures by substituting zeros into `Signals.joint_pos_rad`/`Signals.joint_vel_rad_s`.
That can cause the policy to receive nonsensical observations while the robot is still being commanded.

Recommended patterns:
- `get_positions_rad()` returns `None` on failure; `HardwareRobotIO.read()` either:
  - raises (letting the control loop trip safety), or
  - returns last-known positions and increments an internal failure counter, and the control loop trips safety after a small budget of
    consecutive failures.

If you want to keep `Signals` minimal, keep validity/failure counters *out of the contract* and implement them in runtime safety logic.

#### Conversion responsibility

The `Actuators` implementation is responsible for:
- **Unit conversion**: radians ↔ device units (e.g., servo units [0-1000])
- **Calibration**: applying per-joint `direction` (+1/-1) and `offset` (in device units)
- **Joint ordering**: mapping `actuator_names` to hardware IDs
- **Velocity estimation**: finite-difference from position readings (if hardware lacks velocity feedback)

**Design decision**: Conversion logic lives in `ServoConfig` (Option 1 - co-located with calibration data).

Rationale:
- `ServoConfig` already holds `offset` and `direction` needed for conversion
- Calibration tools (`calibrate.py`) can use the same methods directly
- Simple and avoids extra abstraction layers
- If multi-hardware support is needed later, can migrate to a utility class

```python
# ServoConfig owns conversion (single source of truth)
@dataclass
class ServoConfig:
    id: int
    offset: int       # in servo units
    direction: int    # +1 or -1
  center_deg: float # MuJoCo angle (deg) that maps to servo center (500)

    # Servo model parameters (should be config-driven if hardware varies)
    # Defaults match Hiwonder HTD-45H: units in [0..1000], center at 500, ~240° total travel.
    UNITS_MIN: int = 0
    UNITS_MAX: int = 1000
    UNITS_CENTER: int = 500
    UNITS_PER_RAD: float = 238.73  # 1000 / (240° in rad)

    def rad_to_units(self, target_rad: float) -> int:
        """Convert MuJoCo radians to servo units with calibration."""
      center_rad = math.radians(self.center_deg)
      units = self.UNITS_CENTER + self.offset + self.direction * ((target_rad - center_rad) * self.UNITS_PER_RAD)
        return clamp(units, self.UNITS_MIN, self.UNITS_MAX)

    def units_to_rad(self, units: int) -> float:
        """Convert servo units to MuJoCo radians with calibration."""
      center_rad = math.radians(self.center_deg)
      return center_rad + self.direction * (units - self.UNITS_CENTER - self.offset) / self.UNITS_PER_RAD

# HiwonderActuators delegates to ServoConfig (no conversion logic here)
class HiwonderActuators(Actuators):
    def set_targets_rad(self, targets: np.ndarray) -> None:
        for i, name in enumerate(self._names):
            cfg = self._servo_configs[name]
            units = cfg.rad_to_units(targets[i])  # delegate
            commands.append((cfg.id, units))
        self._controller.move_servos(commands, self._move_ms)
```

This ensures conversion logic is shared between:
- Runtime actuator implementation (`HiwonderActuators`)
- Calibration tooling (`calibrate.py`)
- Any visualization/debugging tools

#### Concrete implementations

| Implementation | Hardware | Notes |
|----------------|----------|-------|
| `HiwonderActuators` | Hiwonder HTD-45H bus servos | Uses `HiwonderBoardController` for serial protocol |
| `DynamixelActuators` | Dynamixel servos | Future: Protocol v2 |
| `SimActuators` | MuJoCo (testing) | For hardware-in-the-loop testing without real servos |

#### Naming/structure note (Hiwonder actuator implementation)

This doc uses `HiwonderActuators` / `HiwonderBoardActuators` as placeholders for the concrete implementation that talks to the
Hiwonder servo controller board. The important design point is:
- `HardwareRobotIO` depends on the `Actuators` protocol (not a specific class name),
- the Hiwonder implementation owns the retry policy, timing (`time_ms`), and unit conversion.

If the repo uses both a legacy stack (`runtime/configs` + `runtime/hardware`) and a newer package (`runtime/wr_runtime`), prefer to have a
single canonical implementation under `runtime/wr_runtime/hardware/` and keep any legacy modules as thin wrappers.

### 5.8.5 IMU integration (end-to-end plan)

This section describes the recommended end-to-end path to bring a physical IMU online in runtime while keeping observation semantics
aligned with training and enforced by `policy_contract`.

#### Contract expectation (what the policy sees)

The actor observation contract assumes the IMU signals are expressed in the **robot body frame**:
- `Signals.quat_xyzw`: quaternion in `xyzw` order, representing the robot body attitude in a convention compatible with
  `policy_contract.*.frames.gravity_local_from_quat(quat_xyzw)` (upright → gravity_local ≈ `[0, 0, -1]`).
- `Signals.gyro_rad_s`: angular velocity `[wx, wy, wz]` in rad/s about body X/Y/Z.
  - yaw-left should yield `wz > 0`, pitch-up should yield `wy > 0`, roll-right should yield `wx > 0`.

`policy_contract` (NumPy/JAX) is responsible for converting these raw signals into the model observation features
(e.g., `gravity_local`, `angvel_heading_local`, normalization, layout packing).

#### Where calibration lives (and why)

Runtime should treat IMU mounting corrections as **hardware/driver calibration**, not policy logic:
- `bno085.upside_down` and `bno085.axis_map` belong in runtime config and are applied in the IMU driver.
- The driver returns body-frame IMU outputs (`quat_xyzw_body`, `gyro_body`) so every consumer (policy loop, calibration tools, loggers)
  sees consistent body-frame semantics.
- `HardwareRobotIO.read()` should remain an aggregator: it packages already-calibrated device outputs into `Signals`.

This mirrors the actuator design: servo offsets/directions are applied in the actuator layer, not in the policy contract.

#### Recommended bring-up sequence

1) **Electrical + OS bring-up**
   - Ensure I2C is enabled and the device address is visible (`i2cdetect`).
   - Use `runtime/docs/bno08x_rpi_install_and_test.md` to install dependencies and validate basic reads.

2) **Set runtime config basics**
   - In `runtime/configs/wr_runtime_config.json`, set:
     - `bno085.i2c_address`
     - `bno085.upside_down` (initial guess is fine; calibration can update it)

3) **Calibrate mount inversion and axis signs**
   - Run the interactive tool: `runtime/scripts/calibrate.py --calibrate-imu`.
   - Prefer the “sign-only” flow if you believe the board axes are aligned with the robot body axes:
     - Calibrate `body_z` (yaw-left), `body_y` (pitch-up), `body_x` (roll-right).
   - The tool enforces two gates:
     - **motion magnitude**: rotation angle must exceed a minimum threshold (prevents accidental “no-motion” acceptance)
     - **axis alignment**: dominant axis must align with the expected axis (guards against silent X/Y swaps due to messy motion or rotated mounting)

4) **Sanity checks (must pass before motors)**
   - Upright and still:
     - `gravity_local_from_quat(quat_xyzw)` should be close to `[0, 0, -1]`.
     - `gyro_rad_s` should be near zero (small noise OK).
   - Small manual motions:
     - yaw-left → `gyro_z` positive and dominant
     - pitch-up → `gyro_y` positive and dominant
     - roll-right → `gyro_x` positive and dominant

5) **Policy loop integration**
   - `HardwareRobotIO.read()` returns `Signals` containing the calibrated IMU outputs and joint/foot signals.
   - `run_policy.py` builds observations via:
     - `obs = policy_contract.numpy.obs.build_observation(spec, state, signals, velocity_cmd)`
   - Runtime then runs inference and applies action→ctrl mapping, without any IMU-specific special cases.

#### IMU integration architecture (data flow)

```
                        ┌─────────────────────────────────────────────┐
                        │          runtime/configs/*.json              │
                        │  bno085: { i2c_address, upside_down, axis_map }
                        └───────────────────────┬─────────────────────┘
                                                │
                                                ▼
┌───────────────────────┐   raw board-frame   ┌───────────────────────┐
│   Physical BNO08x      │  quat/gyro over I2C │   BNO085 IMU Driver    │
│  (mounted on robot)    ├────────────────────►│ runtime/.../bno085.py  │
└───────────────────────┘                      │  - retries/suppress    │
                                               │  - applies upside_down │
                                               │  - applies axis_map    │
                                               │  => body-frame IMU     │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   HardwareRobotIO      │
                                               │ runtime/.../robot_io.py │
                                               │  - read(): aggregates  │
                                               │    IMU + joints + feet │
                                               │  => policy_contract.Signals
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   policy_contract      │
                                               │ numpy/obs.py           │
                                               │  - gravity_local       │
                                               │  - angvel_heading_local│
                                               │  - normalize + pack    │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   Inference + Control  │
                                               │ run_policy.py          │
                                               │  obs -> onnx -> action │
                                               │  action -> ctrl -> HW  │
                                               └───────────────────────┘
```

Key boundary:
- **IMU driver boundary** is where board-frame → body-frame conversion happens.
- `policy_contract` only consumes **body-frame** `Signals` and never needs to know mounting details.

#### Validation gates (recommended)

Add (or keep) lightweight “bring-up” checks that fail fast:
- **IMU contract gate**: in runtime startup, validate that `Signals.quat_xyzw` has unit norm (within tolerance) and produces reasonable
  `gravity_local` when upright.
- **Sign gate**: optional runtime “diagnostic mode” that prints dominant gyro axis/sign during manual yaw/pitch/roll motions.

These gates prevent “silent drift” where the policy sees a plausible-looking observation vector with incorrect semantics.

## 6) Shared Implementation Package (what both sides import)

Create a small Python package intended to be imported by both training and runtime:

Proposed module layout:
- `policy_contract/spec.py`
  - `PolicySpec`, `RobotSpec`, `ObsSpec`, `ActionSpec`
  - JSON schema + load/save + validation
- `policy_contract/spec_builder.py`
  - canonical `build_policy_spec(...)` used by training + export + eval to prevent spec drift
- `policy_contract/numpy/obs.py`
  - `build_observation(...)` implementing the actor observation contract
  - frame transforms helpers (quat→gravity, heading_local transforms)
- `policy_contract/numpy/calib.py`
  - “CAL-lite”: reads joint ranges + mirror_sign + velocity limits from `robot_config.yaml`
  - `normalize_joint_pos/vel`, `action_to_ctrl`
- `policy_contract/jax/obs.py`
  - JAX-native implementation used by training (same semantics as NumPy builder)
- `policy_contract/jax/calib.py`
  - JAX-native action→ctrl mapping (can delegate to training CAL initially)
- `policy_contract/tests/` (fast, deterministic)
  - parity tests that compare `policy_contract/jax/*` vs `policy_contract/numpy/*`

JAX-only implementations can initially coexist in training, but the **long-term direction** is:
- `policy_contract/*` owns the semantics and implementations
- training provides a `sim_adapter/*` that extracts state from MuJoCo/MJX and calls the contract functions

### 6.1 Proposed repository folder structure

This section describes the **ultimate** (“clean”) folder structure that maps to the design.

We can migrate to it incrementally (see §11), but it helps to agree on the destination first.

```
wildrobot/
├─ assets/
│  ├─ wildrobot.xml
│  └─ robot_config.yaml
├─ training/
│  ├─ envs/
│  │  └─ wildrobot_env.py                 # env (reward/termination/metrics) calls policy_contract for actor obs/actions
│  ├─ sim_adapter/
│  │  ├─ mjx_signals.py                   # MJX (training) → Signals for policy_contract
│  │  └─ mujoco_signals.py                # native MuJoCo (visualization) → Signals for policy_contract
│  ├─ export/
│  │  ├─ export_onnx.py                   # checkpoint → deterministic onnx
│  │  ├─ export_policy_bundle.py          # bundle writer (onnx + policy_spec.json + checksums)
│  │  └─ export_policy_bundle_cli.py      # CLI entrypoint
│  ├─ configs/
│  ├─ train.py
│  └─ visualize_policy.py
├─ runtime/
│  ├─ pyproject.toml                       # packaging + console scripts
│  ├─ README.md
│  ├─ configs/                             # runtime JSON configs
│  │  ├─ config.py                         # WrRuntimeConfig + ServoConfig (with rad_to_units)
│  │  └─ wr_runtime_config.json            # per-robot calibration (servo IDs, offsets, direction)
│  ├─ scripts/
│  │  └─ calibrate.py                      # interactive servo calibration tool
│  └─ wr_runtime/                          # ONLY importable runtime package (deploys to robot)
│     ├─ __init__.py
│     ├─ config.py                         # legacy config loader (delegates to runtime/configs)
│     ├─ control/
│     │  ├─ __init__.py
│     │  └─ run_policy.py                  # CLI entry (control loop using HardwareRobotIO)
│     ├─ inference/
│     │  ├─ __init__.py
│     │  └─ onnx_policy.py                 # ONNX backend
│     ├─ hardware/                         # sensor/actuator drivers
│     │  ├─ __init__.py
│     │  ├─ robot_io.py                    # HardwareRobotIO: implements RobotIO protocol
│     │  ├─ actuators.py                   # Actuators protocol + HiwonderActuators implementation
│     │  ├─ hiwonder_board_controller.py   # low-level servo bus serial protocol
│     │  ├─ bno085.py                      # BNO085 IMU driver (I2C)
│     │  └─ foot_switches.py               # GPIO foot switch driver
│     ├─ io/                               # (optional) deprecated stubs; prefer `hardware/robot_io.py`
│     │  └─ __init__.py
│     ├─ validation/
│     │  ├─ __init__.py
│     │  └─ startup_validator.py           # validates bundle + robot_config + mjcf order
│     └─ utils/
│        ├─ __init__.py
│        └─ mjcf.py
└─ policy_contract/                        # NEW shared package (spec + dual backends)
   ├─ __init__.py
   ├─ spec.py                              # PolicySpec + JSON schema + validators (no heavy deps)
   ├─ numpy/
   │  ├─ __init__.py
   │  ├─ frames.py                         # quat/heading-local helpers (pure numpy)
   │  ├─ obs.py                            # build_observation(...) (numpy)
   │  └─ calib.py                          # action→ctrl + normalization (numpy)
   ├─ jax/
   │  ├─ __init__.py
   │  ├─ obs.py                            # build_observation(...) (jax)
   │  └─ calib.py                          # action→ctrl + normalization (jax)
   └─ tests/
      ├─ test_obs_parity.py                # jax obs matches numpy obs
      └─ test_action_mapping_parity.py     # jax action→ctrl matches numpy action→ctrl
```

Notes:
- `policy_contract/spec.py` must remain dependency-light. Runtime must only import `policy_contract/numpy/*` + `policy_contract/spec.py`.
- `runtime/` is the distribution/project root; `runtime/wr_runtime/` is the Python package you import and ship to the robot.
- Runtime stays “thin”: it owns hardware I/O, timing, safety, logging. It does not own policy semantics.
- `policy_contract/*` stays “authoritative” for semantics; training and runtime are adapters constrained by parity tests.

#### Mapping from today → ultimate (incremental, low-risk)

Today, your training stack lives under `training/`. The clean end state above uses `training/` for readability, but we should migrate without breaking imports/tests/scripts:

- **Phase 0 (no behavior change):** keep Python import package name `training` as-is.
- **Phase 1 (structure alignment):** introduce a top-level `training/` *folder* that contains wrappers/docs/export scripts, but internally calls into `training.*`.
- **Phase 2 (optional full rename):** rename the Python package to `training` only once:
  - all internal imports are updated,
  - all tests pass,
  - and external users (runtime/tools) no longer import `training`.

If Phase 2 is desired, the common safe pattern is a temporary compatibility shim:
- keep a `training/` package that re-exports `training/` for one release and warns on import.

### 6.2 File/class structure (interfaces and responsibilities)

This section defines the concrete “seams” between training, export, and runtime.

#### Shared package (`policy_contract/`)

- `PolicySpec` (in `policy_contract/spec.py`)
  - Fields: `contract_name`, `contract_version`, `robot`, `observation`, `action`, `runtime_assumptions`
  - Methods: `from_json()`, `to_json()`, `validate()`
- `RobotSpec`
  - `actuator_names: list[str]`
  - `joint_ranges_rad: dict[str, tuple[float, float]]`
  - `mirror_sign: dict[str, float]`
  - `max_velocity_rad_s: dict[str, float]`
- `ObsSpec`
  - `layout: list[ObsFieldSpec]` where each field has `name,size,units,frame,normalization`
  - `obs_dim: int` (derived) and `breakdown` (optional)
- `ActionSpec`
  - `action_dim: int`
  - `mapping: Literal["action_to_ctrl_v1"]` (or similar)
  - any declared filters (alpha, rate limits) that must be reproduced in runtime
- `policy_contract/numpy/calib.py` (NumPy backend)
  - Constructed from `RobotSpec` (or directly from `robot_config.yaml`)
  - `normalize_joint_pos_rad(qpos_rad) -> np.ndarray`
  - `normalize_joint_vel_rad_s(qvel_rad_s) -> np.ndarray`
  - `action_to_ctrl(spec, action) -> ctrl_targets_rad`
- `policy_contract/numpy/obs.py` (NumPy backend)
  - `build_observation(...) -> obs (float32, shape (obs_dim,))`
  - should be a 1:1 semantic mirror of the JAX builder used in training (`policy_contract/jax/obs.py`)
- `policy_contract/jax/*` (JAX backend)
  - used by training and parity tests; should not be imported by runtime

### 6.3 What lives in `calib.py` (and what does not)

`policy_contract/*/calib.py` is the **minimal, portable “control semantics”** module. It should contain only:

**Required (Stage 1)**
- `action_to_ctrl(spec, action) -> ctrl_targets_rad`
  - implements `action.mapping_id == "pos_target_rad_v1"`
  - uses `robot.joints[*].range_min/max` and `mirror_sign`
- `normalize_joint_pos(spec, joint_pos_rad) -> joint_pos_normalized`
  - uses joint range centers/spans (same formula as training)
- `normalize_joint_vel(spec, joint_vel_rad_s) -> joint_vel_normalized`
  - uses `max_velocity_rad_s` and clips to `[-1, 1]`

**Optional (as contract grows)**
- `ctrl_to_action(spec, ctrl_targets_rad) -> action_normalized`
  - useful for initializing `prev_action` from current pose (`prev_action_init="from_pose"`)

**Explicit non-goals (do NOT put here)**
- hardware calibration/offsets in device units (belongs in `HardwareIO` / drivers)
- MuJoCo indexing (qpos/qvel addresses) or MJX data access (belongs in `training/sim_adapter`)
- safety logic (clamps, watchdogs) beyond the declared policy postprocess (belongs in runtime)

This keeps `calib.py` stable, testable, and shareable across sim, runtime, and visualization.

#### Runtime (`runtime/wr_runtime/`)

- `HardwareRobotIO` (in `runtime/wr_runtime/hardware/robot_io.py`)
  - implements the shared `RobotIO` interface (`read()` / `write_ctrl()` / `close()`)
  - composes device drivers (IMU/actuators/foot switches) and returns contract-friendly `Signals`
- `Signals` (in `policy_contract/numpy/signals.py`)
  - `quat_xyzw`, `gyro_rad_s`, `joint_pos_rad`, `joint_vel_rad_s`, `foot_switches`, `timestamp_s`
- `run_policy.py` (in `runtime/wr_runtime/control/run_policy.py`)
  - CLI control loop: `Signals` → `policy_contract.numpy.obs.build_observation` → ONNX inference → `policy_contract.calib.action_to_ctrl`
  - fail-fast validation via `runtime/wr_runtime/validation/startup_validator.py`
- `OnnxPolicy` (in `runtime/wr_runtime/inference/onnx_policy.py`)
  - minimal ONNX runtime wrapper (input/output name + predict)
- `validate_runtime_interface` (in `runtime/wr_runtime/validation/startup_validator.py`)
  - checks policy bundle dims + MJCF actuator order + runtime config completeness

Note: A dedicated `SafetySupervisor` layer is recommended long-term (clamps, watchdogs, tilt thresholds, e-stop handling), but the current
runtime loop already fail-fast disables actuators on exceptions. Treat safety as an incremental layer on top of the contract pipeline.

#### Training export (`training/exports/export_policy_bundle.py`)

- `export_policy_bundle(checkpoint_path, config_path, output_dir, robot_config_path)`
  - Writes: `policy.onnx`, `policy_spec.json`, `robot_config.yaml` snapshot, `checksums.json`, `wildrobot_config.json`
  - `wildrobot_config.json` is generated from `runtime/configs/wr_runtime_config.json` with `policy_onnx_path=./policy.onnx`
  - Ensures: `policy_spec.obs_dim/action_dim` match the exported ONNX and `assets/v1/robot_config.yaml` (or the selected variant config)

## 7) Policy Bundle Artifact (what gets deployed)

Instead of deploying only `policy.onnx`, deploy a folder (or zip/tar):

`wildrobot_policy_bundle/`
- `policy.onnx`
- `policy_spec.json`  ← versioned contract + all semantic knobs used in training
- `robot_config.yaml` ← snapshot used to train/export (optional but recommended)
- `checksums.json`    ← sha256 of the above for integrity
- `wildrobot_config.json` ← runtime JSON seeded from `runtime/configs/wr_runtime_config.json` (hardware settings)
- `wildrobot.xml` ← MJCF snapshot for actuator order validation
- `README.md`         ← quick provenance + expected runtime command

Runtime loads a *bundle*, not a raw ONNX path.

Runtime config should point to the bundle’s ONNX path (same folder as the spec):
```
"policy_onnx_path": "../policies/wildrobot_policy_bundle/policy.onnx"
```

If you use the generated `wildrobot_config.json` colocated with the bundle, it uses:
```
"policy_onnx_path": "./policy.onnx"
```

Bundle validation can be run directly on the bundle folder:
```
wildrobot-validate-bundle --bundle <bundle_dir>
```

Policy execution should also be bundle-oriented:
```
wildrobot-run-policy --bundle <bundle_dir>
```

## 8) Runtime Design (bridging training logic cleanly)

### 8.1 Core interfaces
- `RobotIO`
  - `read_sensors() -> SensorSample` (IMU, foot switches, joint pos/vel)
  - `apply_ctrl(ctrl_targets_rad: np.ndarray) -> None`
- `PolicyRunner`
  - owns `PolicySpec`, `Calib`, `ObsBuilder`, `OnnxPolicy`
  - loop:
    1) read sensors
    2) build obs using contract semantics
    3) `action = policy(obs)`
    4) apply declared filters/clamps
    5) map to `ctrl_targets_rad` using contract mapping
    6) send to actuators

### 8.2 Startup validation (fail fast)
Runtime must refuse to run if any of these mismatch:
- `actuator_names` != MJCF actuator order
- policy `obs_dim/action_dim` != contract
- missing `servo_ids` for required joints
- contract version unsupported by runtime

### 8.3 Logging for sim2real debugging
Always log at least:
- wall-clock loop dt + inference latency
- raw sensor readings (downsampled if needed)
- final obs vector (or selected slices)
- action before/after filters, ctrl targets

This is what lets you do “playback in sim” and reproduce issues.

## 9) Training-side Changes (export pipeline)

Add a single canonical exporter entry point:
- `training/exports/export_policy_bundle.py` (library)
- `training/exports/export_policy_bundle_cli.py` (CLI)
  - loads checkpoint
  - exports deterministic ONNX (reuse `training/export_onnx.py`)
  - generates `policy_spec.json` from:
    - `assets/v1/robot_config.yaml`
    - policy_contract observation layout breakdown
    - training env knobs that affect inference (action_filter_alpha, etc.)
  - writes bundle folder

Stage 1 gate: exported bundle passes runtime validation + a short hardware smoke test.

## 9.1 Training enforcement (avoid “last-minute” contract drift)

Goal: ensure the exported `policy_spec.json` is not a last-minute interpretation layer, but a contract the policy has been trained against.

### Principle: train against the contract

Training should construct a `PolicySpec` (or a small set of spec fields) at startup and use it as the single source of truth for:
- `obs_dim` and observation layout (`observation.layout_id`)
- action semantics (`action.mapping_id`)
- inference-time postprocessing (`action.postprocess_id` + params)

Concretely, training should call the shared implementation:
- `policy_contract/jax/obs.py:build_observation(...)`
- `policy_contract/jax/calib.py:action_to_ctrl(...)`
- `policy_contract/jax/action.py:postprocess_action(...)` (if filters are part of the contract)

This ensures the policy is optimized for exactly the tensors and control semantics it will see on hardware.

### Validations to add (cheap, fail-fast)

At training startup (before long runs):
- `validate_spec(spec)`
- assert the policy network input dim == `spec.model.obs_dim`
- assert the policy network output/action dim == `spec.model.action_dim`
- assert env-reported `observation_size` matches `spec.model.obs_dim`

At export time (before copying to robot):
- `validate_spec(spec)` again
- export ONNX
- `validate_runtime_compat(...)` using:
  - MJCF-derived actuator order
  - ONNX input/output dims

### Scope rule: what belongs in `PolicySpec` (and what does not)

To keep the contract clean, treat `PolicySpec` as **only what the model expects at inference time**:
- observation layout + dims
- action dims + postprocess + mapping semantics
- actuator ordering and joint ranges needed for normalization/mapping

Explicit non-goals for `PolicySpec`:
- hardware calibration and device units (servo offsets, bus timing) → runtime config / HardwareIO
- sim-only indexing (qpos/qvel addresses) → `training/sim_adapter`
- safety supervision (watchdogs, tilt thresholds) → runtime safety layer

With this scoping, the rule becomes simple:
- **Everything in `PolicySpec` must be enforced consistently by training (JAX) and runtime (NumPy)** via `policy_contract`.

### Privileged signals (recommended pattern)

If you want to use additional simulator state (including linvel), treat it as **privileged**:
- still use it for rewards, diagnostics, or critic-only inputs
- do not feed it into the actor observation tensor defined by `PolicySpec`

## 9.1) Human-like walking (AMP + AMASS): contract discipline

When introducing AMP (and later AMASS-derived reference motion), the most common failure mode is not “AMP math” but **feature and interface drift**. Recommended policy for this repo:

- **Keep the actor contract hardware-realizable.** The deployed policy observation must remain limited to signals available on the robot (IMU quat/gyro, joint pos/vel (or estimated), foot switches, prev_action, command). Do not add sim-only channels to the actor just because they are convenient in sim.
- **Use asymmetric training.**
  - **Actor**: consumes only `PolicySpec` observations produced by `policy_contract` (deployment contract).
  - **Critic and AMP discriminator**: may consume richer sim state (`qpos/qvel`, root velocities, site positions) as *privileged training-only* features.
  - This preserves sim2real while still giving AMP a clean learning signal.
- **Version and test AMP feature definitions.**
  - Treat AMP feature construction as an ABI: frame conventions, ordering, normalization, and masking must be stable and versioned.
  - Add deterministic parity tests for AMP features (JAX vs NumPy where applicable) and “golden” vectors so refactors cannot silently change semantics.
- **Separate feasibility from style.**
  - Stage 1 learns a robot-native gait manifold (feasible locomotion) using task rewards.
  - AMP/AMASS should be introduced as a *style prior* layered on top of a feasible controller, not as the primary feasibility signal.
- **Prefer invariant frames.**
  - Compute reference features in yaw-invariant frames (e.g., heading-local) and explicitly document frame transforms in `policy_contract`.
  - If contact estimation differs between sim and hardware, avoid making discriminator hinge on brittle contact edges; prefer robust kinematic/pose features and smooth contact proxies.

## 10) Validation Strategy (what “seamless” means)

### 10.1 Parity tests (unit)
- Generate deterministic synthetic inputs and assert:
  - `policy_contract/numpy/obs.build_observation(...)` matches `policy_contract/jax/obs.build_observation(...)`
  - numpy CAL-lite `action_to_ctrl(...)` matches training CAL on the same robot_config

### 10.2 Integration test (sim playback)
- Record a short trajectory on hardware (sensors + actions)
- Re-run the same observation builder + policy in a “replay” script to confirm determinism

### 10.3 Hardware smoke test (Stage 1)
- “standing/walking-in-place” safe script:
  - clamps + very small action_scale (or direct ctrl mapping with tight ranges)
  - immediate e-stop on excessive tilt, servo comms loss, or loop deadline misses

## 10.4 Runtime safety notes (must-haves)

- **Clamps:** runtime should clamp control targets to joint limits (or tighter safety limits) before sending to actuators.
- **Action filter:** if `spec.action.postprocess_id` declares a low-pass filter, runtime must apply it exactly as specified.
- **E-stop:** runtime should stop and unload actuators on excessive tilt, comms loss, or missed deadlines.
- **No overrides:** runtime must never override the exported `lowpass_v1.alpha`. Changing it changes inference-time semantics and invalidates parity.

## 11) Migration Plan (incremental, low risk)

1) **Introduce PolicySpec + bundle format** (no runtime behavior change yet).
2) **Move runtime obs building** to shared contract builder:
   - implement normalized joint pos/vel and declared frames.
3) **Move runtime action mapping** to contract (mirror_sign + joint range mapping):
   - stop using `action_scale_rad * action` as “delta around zero” unless explicitly trained that way.
4) **Add parity tests** so changes can’t regress silently.

## 12) Open Questions (to decide together)

- Do we want runtime to compute `heading_local` exactly, or retrain with body-frame gyro/vel to simplify?
- Do we want a “reference pose” offset (e.g., keyframe pose) baked into ctrl mapping, or pure joint-range mapping only?
- Should runtime drive servos in (radians) or in native servo units? (Contract should stay in radians; the hardware adapter can convert.)
- For angular velocity in the contract, should sim derive it from:
  - `chest_imu_gyro` sensor (preferred; matches hardware IMU gyro semantics), or
  - finite-difference of base orientation / `qvel` (convenient but can drift vs hardware)?

---

## Appendix: Practical Implementation Notes for This Repo

- `assets/v1/robot_config.yaml` already contains `observation_breakdown`, joint ranges, `mirror_sign`, and `max_velocity`. That is enough to define the contract for Stage 1.
- Runtime should have a startup validator (e.g. `runtime/wr_runtime/validation/startup_validator.py`) that checks dims and order; extend it to also validate **semantic knobs** from `policy_spec.json` (frames, normalization declared, action mapping type, filter alpha).
- Training currently uses CAL (`training/cal/cal.py`) as the “truth”; long-term, move those semantics into `policy_contract/` and keep training’s MuJoCo-specific logic in `training/sim_adapter/`.
