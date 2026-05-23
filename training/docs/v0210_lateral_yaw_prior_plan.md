# Plan: v0.21.0 — Lateral (vy) + Yaw (wz) Prior Support for WildRobot

**Goal**: Extend WR's offline ZMP prior and policy I/O from forward-only (vx) to full 3-DoF locomotion commands (vx, vy, wz), mirroring ToddlerBot's unified ZMP design, so the policy can learn to strafe and turn alongside walking forward.

**Architecture**:
- **Single unified ZMP prior**, not a separate "lateral module". Add vy + wz axes to the existing `ZMPWalkGenerator`, `ReferenceLibrary`, env sampler, observation, and reward path. Mirror TB's `toddlerbot/algorithms/zmp_walk.py` translation branch (line 259) for `(vx, vy)` and pure-rotation branch (line 269) for wz.
- **Command obs widened from 1D → 3D** (`velocity_cmd` channel becomes `(vx, vy, wz)`). New layout `wr_obs_v8_cmd3d` cloned from `wr_obs_v7_phase_proprio`. Old layouts remain functional with vy/wz=0.
- **Reward scaffolding already partially exists**: `cmd_velocity_track_dim=2` is wired (smoke13 uses it) but `vy_cmd` is hard-coded to 0 at `wildrobot_env.py:1217`. P5 just threads the real cmd through, and adds a yaw-rate tracking term.
- **WR-scaled command ranges**: vy ∈ ±0.13 m/s (same vy/vx aspect ratio as TB), wz ∈ ±0.25 rad/s (matches TB foot-arc-per-swing kinematic difficulty).

**Tech Stack**: Python 3.12, JAX, Brax, MuJoCo MJX, PPO. Tests run via `uv run pytest training/tests/test_xxx.py -q` from repo root `/Users/ygli/projects/wildrobot`.

**Cross-robot scaling reference**:
| Quantity | TB | WR | Ratio |
|---|---|---|---|
| Leg length | 0.2115 m | 0.373 m | 1.76× |
| Hip width | 0.036 m | 0.107 m | 2.98× |
| cycle_time | 0.72 s | 0.96 s | 1.33× |
| vx_max (training) | 0.10 m/s | 0.26 m/s | 2.6× |
| **vy_max (this plan)** | 0.05 m/s | **0.13 m/s** | 2.6× (preserve vy/vx ratio) |
| **wz_max (this plan)** | 1.0 rad/s | **0.25 rad/s** | 0.25× (preserve foot-arc-per-swing) |

---

## Task Dependencies

| Group | Steps | Parallelizable | Notes |
|-------|-------|----------------|-------|
| G1 | P1.1–P1.5 (planner) | No (sequential within file) | Touches `control/zmp/zmp_walk.py` only |
| G2 | P2.1–P2.4 (library) | After G1 | Touches `control/references/*.py` only |
| G3 | P3.1–P3.3 (sampler) + P4.1–P4.3 (obs layout) | **Yes** (different files) | P3 in `wildrobot_env.py::_sample_velocity_cmd`; P4 in `policy_contract/jax/obs.py` |
| G4 | P5.1–P5.3 (reward) + P6.1 (residual scales) | After G3 | P5 in `wildrobot_env.py`; P6 in YAML |
| G5 | P7.x (integration tests) | After G4 | Cross-layer tests |
| G6 | P8.1 (smoke YAML) + P8.2 (docs gates) | **Yes** | Independent files |
| G7 | P9.1 (changelog) | After smoke results | Final |

---

## P1 — Extend ZMP planner with vy + wz axes

**File**: `/Users/ygli/projects/wildrobot/control/zmp/zmp_walk.py`

### P1.1 — Add `command_vy` and `command_yaw_rate` to `ZMPWalkConfig` (failing test)

**Failing test**: `/Users/ygli/projects/wildrobot/training/tests/test_zmp_walk_config_3d_cmd.py`
```python
from __future__ import annotations
import pytest
from control.zmp.zmp_walk import ZMPWalkConfig


def test_zmp_walk_config_accepts_vy_and_yaw_rate() -> None:
    cfg = ZMPWalkConfig(command_vx=0.20, command_vy=0.05, command_yaw_rate=0.10)
    assert cfg.command_vy == pytest.approx(0.05)
    assert cfg.command_yaw_rate == pytest.approx(0.10)


def test_zmp_walk_config_defaults_vy_and_yaw_to_zero() -> None:
    cfg = ZMPWalkConfig(command_vx=0.20)
    assert cfg.command_vy == 0.0
    assert cfg.command_yaw_rate == 0.0
```

**Run**: `cd /Users/ygli/projects/wildrobot && uv run pytest training/tests/test_zmp_walk_config_3d_cmd.py -q` — expect 2 failures (`unexpected keyword argument 'command_vy'`).

### P1.2 — Implement `command_vy` and `command_yaw_rate` fields

Add fields to `ZMPWalkConfig` dataclass (file currently has `command_vx`). Default both to `0.0`.

**Verify**: rerun P1.1 — both pass.

### P1.3 — Lateral footstep planning (failing test)

**Failing test**: same file, append:
```python
import numpy as np
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_planner_produces_nonzero_lateral_foot_displacement_for_vy() -> None:
    gen = ZMPWalkGenerator(ZMPWalkConfig(command_vx=0.0, command_vy=0.05))
    traj = gen.generate()
    foot_y = traj.left_foot_pos[:, 1]
    # Foot must drift in +y direction over the cycle.
    assert (foot_y.max() - foot_y.min()) > 0.005
```

**Run**: expect failure (no vy stride yet).

### P1.4 — Implement lateral stride in footstep planner

In the footstep-placement block of `ZMPWalkGenerator`, mirror TB's combined-translation branch at `/Users/ygli/projects/toddlerbot/toddlerbot/algorithms/zmp_walk.py:259`:
```python
stride_xy = np.array([cfg.command_vx, cfg.command_vy]) * total_time / (2 * num_cycles - 1)
```
Apply to both foot targets (alternating sign on lateral component is handled by the existing left/right footstep alternation; lateral component is *added* to nominal stance offset, not signed).

**Verify**: rerun P1.3 — passes.

### P1.5 — Pure-rotation footstep branch (failing + impl)

**Failing test**:
```python
def test_planner_yaws_pelvis_for_nonzero_yaw_rate() -> None:
    gen = ZMPWalkGenerator(ZMPWalkConfig(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10))
    traj = gen.generate()
    # Pelvis quat z-component (small-angle) must accumulate nonzero yaw across the cycle.
    quat_z = traj.pelvis_quat[:, 3]
    assert abs(quat_z[-1] - quat_z[0]) > 1e-3
```

**Impl**: add a third branch in the planner — when `|vx| + |vy| < eps` and `|yaw_rate| > eps`, plan footsteps that rotate about the pelvis center by `yaw_rate * cycle_time` per cycle. Mirror TB `zmp_walk.py:269`. Emit pelvis_quat with accumulated yaw.

**Verify**: P1.5 passes. Also re-run P1.3 and P1.1.

### P1.6 — Persist new cmd into `ReferenceTrajectory`

**File**: `/Users/ygli/projects/wildrobot/control/references/reference_library.py` — verify `ReferenceTrajectory` already has `command_vy` and `command_yaw_rate` fields (research confirmed line 45). In `ZMPWalkGenerator.generate()`, fill them with `cfg.command_vy` / `cfg.command_yaw_rate` instead of `0.0`.

**Test**: append to P1.1 file:
```python
def test_generated_trajectory_carries_cmd_axes() -> None:
    gen = ZMPWalkGenerator(ZMPWalkConfig(command_vx=0.2, command_vy=0.04, command_yaw_rate=0.10))
    traj = gen.generate()
    assert traj.command_vx == pytest.approx(0.2)
    assert traj.command_vy == pytest.approx(0.04)
    assert traj.command_yaw_rate == pytest.approx(0.10)
```

**Verify**: passes.

### P1.7 — Commit
```bash
git add control/zmp/zmp_walk.py control/references/reference_library.py training/tests/test_zmp_walk_config_3d_cmd.py
git commit -m "v0.21.0 P1: ZMP planner accepts (vy, wz) and emits 3D cmd in ReferenceTrajectory"
```

---

## P2 — 3D reference library lookup

**Files**: `/Users/ygli/projects/wildrobot/control/references/reference_library.py`, `runtime_reference_service.py`, `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py`.

### P2.1 — Library accepts 3D grid (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_reference_library_3d.py`
```python
from __future__ import annotations
import pytest
from control.references.reference_library import ReferenceLibrary, ReferenceLibraryMeta


def test_meta_carries_vy_and_wz_grids() -> None:
    meta = ReferenceLibraryMeta(
        vx_grid=(0.20,),
        vy_grid=(-0.05, 0.0, 0.05),
        yaw_rate_grid=(-0.10, 0.0, 0.10),
    )
    assert meta.vy_grid == (-0.05, 0.0, 0.05)
    assert meta.yaw_rate_grid == (-0.10, 0.0, 0.10)
```

**Run**: expect failure (`vy_grid` / `yaw_rate_grid` not accepted, or default-only).

### P2.2 — Impl: add `vy_grid` and `yaw_rate_grid` to `ReferenceLibraryMeta`

In `ReferenceLibraryMeta` (`reference_library.py:340`), add fields with default `(0.0,)`. Update `__post_init__` validation.

**Verify**: P2.1 passes.

### P2.3 — Library 3D-key nearest lookup (failing + impl)

**Failing test** in same file:
```python
import numpy as np


def _make_traj(vx: float, vy: float, wz: float):
    # Stub: build a minimal ReferenceTrajectory tagged with the command axes.
    from control.references.reference_library import ReferenceTrajectory
    return ReferenceTrajectory(
        command_vx=vx, command_vy=vy, command_yaw_rate=wz,
        # ... other fields can be zero-filled for the lookup test
    )


def test_library_lookup_picks_nearest_3d_key() -> None:
    lib = ReferenceLibrary(
        meta=ReferenceLibraryMeta(
            vx_grid=(0.20,),
            vy_grid=(-0.05, 0.0, 0.05),
            yaw_rate_grid=(-0.10, 0.0, 0.10),
        ),
        trajectories={
            (0.20, vy, wz): _make_traj(0.20, vy, wz)
            for vy in (-0.05, 0.0, 0.05)
            for wz in (-0.10, 0.0, 0.10)
        },
    )
    picked = lib.lookup(vx=0.20, vy=0.04, yaw_rate=-0.08)
    assert picked.command_vy == pytest.approx(0.05)
    assert picked.command_yaw_rate == pytest.approx(-0.10)
```

**Impl**: extend `ReferenceLibrary.lookup` (currently 1-D over `vx`) to accept `(vx, vy, yaw_rate)` and pick the trajectory with min L2 distance over the 3-vector. Mirror TB `walk_zmp_ref.py:138-140`.

**Verify**: P2.3 passes.

### P2.4 — `RuntimeReferenceService.to_jax_arrays()` widens the key

In `runtime_reference_service.py::to_jax_arrays`, build a stacked array of shape `(N, T, ...)` where `N = len(vx_grid) * len(vy_grid) * len(yaw_rate_grid)`. Build a parallel `cmd_keys: jnp.ndarray of shape (N, 3)`.

Update `lookup_jax(cmd_vx, cmd_vy, cmd_wz)` to return `argmin(L2(cmd_keys - cmd))`.

**Test**: add JAX-side test:
```python
def test_runtime_service_jax_lookup_returns_correct_bin() -> None:
    # Build a 3x3x3 grid, call lookup_jax with off-grid cmd, assert returned trajectory's
    # tagged (vx,vy,wz) matches the nearest grid point.
    ...
```

**Verify**: passes.

### P2.5 — Env init wires the 3D grid

In `wildrobot_env.py::_init_offline_service` (line 764), if `EnvConfig.loc_ref_offline_command_vy_grid` / `_yaw_rate_grid` are non-empty, pass them into `ZMPWalkGenerator` and `ReferenceLibraryMeta`. Default both to `(0.0,)` so old configs continue building forward-only libraries.

Update `_lookup_offline_window` (line 1248): pass current `(vy_cmd, yaw_rate_cmd)` to `lookup_jax`.

**Test**: add to `test_reference_library_3d.py`:
```python
def test_env_init_with_3d_grid_builds_27_bins() -> None:
    from training.envs.wildrobot_env import WildRobotEnv
    # Use a tiny config that requests vy_grid=(-0.05,0,0.05), yaw_rate_grid=(-0.1,0,0.1).
    # Assert env._offline_service.num_bins == 27.
    ...
```

### P2.6 — Commit
```bash
git add control/references/ training/envs/wildrobot_env.py training/tests/test_reference_library_3d.py
git commit -m "v0.21.0 P2: ReferenceLibrary indexed on (vx, vy, yaw_rate) with L2 nearest-key lookup"
```

---

## P3 — Env command space: 3D cmd sampler

**File**: `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py` (around line 2539).
**File**: `/Users/ygli/projects/wildrobot/training/configs/training_runtime_config.py` (around line 108).

### P3.1 — `EnvConfig` accepts vy/wz ranges (failing test)

**Failing test**: `/Users/ygli/projects/wildrobot/training/tests/test_env_config_3d_cmd.py`
```python
from training.configs.training_runtime_config import EnvConfig


def test_env_config_accepts_vy_and_yaw_rate_ranges() -> None:
    cfg = EnvConfig(
        min_velocity=0.18, max_velocity=0.26,
        min_velocity_y=-0.13, max_velocity_y=0.13,
        max_yaw_rate=0.25,
    )
    assert cfg.min_velocity_y == -0.13
    assert cfg.max_velocity_y == 0.13
    assert cfg.max_yaw_rate == 0.25


def test_env_config_defaults_vy_wz_to_zero() -> None:
    cfg = EnvConfig(min_velocity=0.18, max_velocity=0.26)
    assert cfg.min_velocity_y == 0.0
    assert cfg.max_velocity_y == 0.0
    assert cfg.max_yaw_rate == 0.0
```

### P3.2 — Impl: add fields to `EnvConfig`

In `training_runtime_config.py`, add the three fields with default `0.0`. Update YAML schema validation if any.

**Verify**: P3.1 passes.

### P3.3 — `_sample_velocity_cmd` returns 3-vector (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_sample_velocity_cmd_3d.py`
```python
from __future__ import annotations
import jax
import jax.numpy as jp
import numpy as np
from training.envs.wildrobot_env import WildRobotEnv
from training.configs.training_config import load_training_config


def _env(yaml_path: str):
    cfg = load_training_config(yaml_path)
    return WildRobotEnv(cfg)


def test_sample_velocity_cmd_returns_3_vector() -> None:
    env = _env("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    key = jax.random.PRNGKey(0)
    cmd = env._sample_velocity_cmd(key)
    assert cmd.shape == (3,)


def test_sample_velocity_cmd_respects_ranges() -> None:
    env = _env("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    assert (cmds[:, 0] >= 0.0).all() and (cmds[:, 0] <= 0.26).all()
    assert (jp.abs(cmds[:, 1]) <= 0.13 + 1e-6).all()
    assert (jp.abs(cmds[:, 2]) <= 0.25 + 1e-6).all()


def test_sample_velocity_cmd_branches_zero_turn_walk() -> None:
    # With zero_chance=0.2, turn_chance=0.2 the marginal distribution
    # must be ~20%/20%/60%; check empirical fractions in 4096 samples.
    ...
```

### P3.4 — Impl: branched sampler mirroring TB

In `_sample_velocity_cmd` (line 2539), implement:
```python
def _sample_velocity_cmd(self, key: jax.Array) -> jax.Array:
    cfg = self._config.env
    k_branch, k_vx, k_vy, k_wz = jax.random.split(key, 4)
    u = jax.random.uniform(k_branch)

    # Branch A: zero
    zero_cmd = jp.zeros(3, dtype=jp.float32)

    # Branch B: pure turn (vx=vy=0, wz ~ U[-max, max])
    wz = jax.random.uniform(k_wz, minval=-cfg.max_yaw_rate, maxval=cfg.max_yaw_rate)
    turn_cmd = jp.array([0.0, 0.0, wz], dtype=jp.float32)

    # Branch C: walk (sample (vx, vy) on an ellipse; wz = 0)
    theta = jax.random.uniform(k_vx, minval=0.0, maxval=2 * jp.pi)
    r = jax.random.uniform(k_vy, minval=0.5, maxval=1.0)  # avoid sub-deadzone
    vx_max, vy_max = cfg.max_velocity, cfg.max_velocity_y
    vx = r * vx_max * jp.cos(theta)
    vy = r * vy_max * jp.sin(theta)
    walk_cmd = jp.array([vx, vy, 0.0], dtype=jp.float32)

    cmd = jp.where(u < cfg.cmd_zero_chance, zero_cmd,
            jp.where(u < cfg.cmd_zero_chance + cfg.cmd_turn_chance, turn_cmd, walk_cmd))
    return cmd
```

Mirror TB's branched sampler at `/Users/ygli/projects/toddlerbot/toddlerbot/locomotion/walk_env.py:235-321`.

**Verify**: P3.3 passes.

### P3.5 — Commit
```bash
git add training/configs/training_runtime_config.py training/envs/wildrobot_env.py training/tests/test_env_config_3d_cmd.py training/tests/test_sample_velocity_cmd_3d.py
git commit -m "v0.21.0 P3: env command space extended to (vx, vy, wz) with branched sampler"
```

---

## P4 — Observation widening: `wr_obs_v8_cmd3d` layout

**File**: `/Users/ygli/projects/wildrobot/policy_contract/jax/obs.py` (around line 151).

### P4.1 — New layout `wr_obs_v8_cmd3d` (failing test)

**Failing test**: `/Users/ygli/projects/wildrobot/training/tests/test_obs_v8_cmd3d.py`
```python
import jax.numpy as jp
from policy_contract.jax.obs import build_obs, get_obs_layout


def test_v8_layout_velocity_cmd_slot_is_3d() -> None:
    layout = get_obs_layout("wr_obs_v8_cmd3d")
    assert layout.slot_size("velocity_cmd") == 3


def test_v8_layout_inherits_phase_and_proprio_from_v7() -> None:
    v7 = get_obs_layout("wr_obs_v7_phase_proprio")
    v8 = get_obs_layout("wr_obs_v8_cmd3d")
    for slot in ("phase_sin_cos", "proprio_history"):
        assert v8.slot_size(slot) == v7.slot_size(slot)
```

### P4.2 — Impl: clone v7 → v8 with widened cmd

In `policy_contract/jax/obs.py`:
1. Add layout key `"wr_obs_v8_cmd3d"` to the layout registry.
2. Replace `jnp.atleast_1d(velocity_cmd).reshape(1)` with `jnp.asarray(velocity_cmd, dtype=jnp.float32).reshape(3)` in the v8 branch.
3. Keep v7 / v6 / older layouts untouched.

**Verify**: P4.1 passes.

### P4.3 — Env selects v8 when `obs_version == "wr_obs_v8_cmd3d"`

In `wildrobot_env.py::_get_obs` (line 1439), the dispatch already routes by `obs_version`; just confirm v8 is reachable. Add an integration test:

```python
def test_env_with_v8_layout_produces_obs_with_3d_cmd() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    cmd_slot = env._obs_layout.slice_obs(state.obs, "velocity_cmd")
    assert cmd_slot.shape == (3,)
```

### P4.4 — Commit
```bash
git add policy_contract/jax/obs.py training/tests/test_obs_v8_cmd3d.py
git commit -m "v0.21.0 P4: new obs layout wr_obs_v8_cmd3d (velocity_cmd widened 1D → 3D)"
```

---

## P5 — Reward updates: consume real vy_cmd and yaw_rate_cmd

**File**: `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py` (lines 1163–1244, 1777–1800, 3050–3120).

### P5.1 — Thread real vy_cmd into existing dim=2 reward (failing test)

**Failing test**: `/Users/ygli/projects/wildrobot/training/tests/test_cmd_lateral_velocity_tracking.py`
```python
import jax.numpy as jp
import pytest
from training.envs.wildrobot_env import WildRobotEnv
from training.configs.training_config import load_training_config


def test_reward_peaks_when_lateral_velocity_matches_cmd() -> None:
    env = _env_with_v8_layout()
    # Hand-craft a state where forward_velocity == vx_cmd AND lateral_velocity == vy_cmd.
    # _cmd_velocity_track_reward must return ~1.0.
    ...


def test_reward_decays_when_lateral_velocity_diverges_from_cmd() -> None:
    # Same as above but lateral_velocity = vy_cmd + 0.1
    # Reward must be strictly less than the matched case.
    ...
```

### P5.2 — Impl: remove hard-coded `vy_cmd = 0`

At `wildrobot_env.py:1163`:
1. Rename `_cmd_forward_velocity_track_reward` → `_cmd_velocity_track_reward` (no behavior change for dim=1).
2. Add `vy_cmd: jax.Array` parameter.
3. Remove lines 1217–1218 (`# WR has no lateral command; vy_cmd = 0.\n vy_err_cmd = lateral_velocity`); replace with `vy_err_cmd = lateral_velocity - vy_cmd`.
4. At call site (line 1790), pass the actual `velocity_cmd[1]` instead of relying on the hard-coded 0.
5. Update comment at line 1195 to say "WR receives a 3D command (vx, vy, wz) under obs layout v8".

**Verify**: P5.1 passes.

### P5.3 — Add yaw-rate tracking reward term (failing + impl)

**Failing test**:
```python
def test_yaw_rate_tracking_reward_peaks_at_cmd() -> None:
    env = _env_with_v8_layout()
    # Hand-craft state where ang_vel_z == yaw_rate_cmd. Reward ~1.0.
    ...
```

**Impl**:
1. Add `_yaw_rate_track_reward(ang_vel_z, yaw_rate_cmd) -> Tuple[reward, err]` modelled on `_cmd_velocity_track_reward`.
2. Add new metric `tracking/yaw_rate_err` to `metrics_registry` (file `/Users/ygli/projects/wildrobot/training/core/metrics_registry.py`).
3. Wire into `_compute_reward_terms` (line 1591) and aggregator weight `cmd_yaw_rate_track` in `RewardWeightsConfig` (default `0.0` so old configs ignore it).

**Verify**: P5.3 passes.

### P5.4 — Commit
```bash
git add training/envs/wildrobot_env.py training/core/metrics_registry.py training/configs/training_runtime_config.py training/tests/test_cmd_lateral_velocity_tracking.py
git commit -m "v0.21.0 P5: reward path consumes real (vy_cmd, yaw_rate_cmd) from 3D command"
```

---

## P6 — Residual scales for lateral DoFs

**File**: `/Users/ygli/projects/wildrobot/training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml` (new — created in P8.1).

### P6.1 — Bump *_hip_roll, *_ankle_roll, *_hip_yaw residual scales

In the new smoke YAML's `env.loc_ref_residual_scale_per_joint`:
- Copy `loc_ref_residual_scale_per_joint` from smoke14 baseline.
- Multiply `left_hip_roll`, `right_hip_roll`, `left_ankle_roll`, `right_ankle_roll`, `left_hip_yaw`, `right_hip_yaw` entries by **1.5×**.
- Leave forward/sagittal joints unchanged.

**Why 1.5× not 2×?** WR's hip width is 2.98× TB's; lateral residual demand per stride is lower than vx residual demand. Start conservative.

**Test**: add to `test_config_load_smoke13.py` style file `test_config_load_v0210_smoke1.py`:
```python
def test_smoke1_has_bumped_lateral_residual_scales() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    scales = cfg.env.loc_ref_residual_scale_per_joint
    base = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml").env.loc_ref_residual_scale_per_joint
    for j in ("left_hip_roll", "right_hip_roll", "left_ankle_roll", "right_ankle_roll", "left_hip_yaw", "right_hip_yaw"):
        assert scales[j] == pytest.approx(base[j] * 1.5)
```

### P6.2 — Commit (after P8.1 creates the YAML)
```bash
git add training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml training/tests/test_config_load_v0210_smoke1.py
git commit -m "v0.21.0 P6: 1.5x residual scale on lateral hip/ankle/yaw DoFs in smoke1"
```

---

## P7 — Integration tests

### P7.1 — End-to-end: env step with 3D cmd produces non-zero lateral motion in window

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_env_step_lateral_yaw_e2e.py`
```python
import jax
import jax.numpy as jp
from training.envs.wildrobot_env import WildRobotEnv
from training.configs.training_config import load_training_config


def test_env_rollout_with_lateral_cmd_shows_pelvis_drift_y() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    # Force command to (0.20, 0.10, 0.0) for the rollout.
    state = state.replace(info={**state.info, "velocity_cmd": jp.array([0.20, 0.10, 0.0])})
    # Rollout 50 steps with zero action.
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size))
    # Pelvis y must have drifted positive.
    assert state.pipeline_state.qpos[1] > 0.005


def test_env_rollout_with_yaw_cmd_shows_pelvis_rotation() -> None:
    # Same pattern with cmd (0.0, 0.0, 0.20). Assert quat z component changes.
    ...
```

### P7.2 — Sampler ellipse coverage test

In `test_sample_velocity_cmd_3d.py` (extend from P3.3): after 4096 samples, assert the (vx, vy) walk-branch samples have empirical correlation `|corr(vx, vy)| < 0.05` and per-axis stdev within 5% of the expected ellipse marginal.

### P7.3 — Reward sanity: scalar mode still works (regression guard)

**File**: `test_cmd_velocity_tracking_mode.py` (existing) — confirm `test_scalar_cmd_tracking_ignores_lateral_velocity` and `test_smoke13_sets_2d_cmd_velocity_tracking_mode` still pass after P5 changes. No code changes; just rerun.

### P7.4 — Old layouts continue to work (regression guard)

```python
def test_v7_layout_still_returns_1d_cmd_slot() -> None:
    layout = get_obs_layout("wr_obs_v7_phase_proprio")
    assert layout.slot_size("velocity_cmd") == 1


def test_smoke14_env_init_unaffected() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    cmd_slot = env._obs_layout.slice_obs(state.obs, "velocity_cmd")
    assert cmd_slot.shape == (1,)
```

### P7.5 — Commit
```bash
git add training/tests/test_env_step_lateral_yaw_e2e.py training/tests/test_obs_v8_cmd3d.py
git commit -m "v0.21.0 P7: end-to-end lateral/yaw rollout tests and v7 regression guard"
```

---

## P8 — Smoke YAML + walking_training.md gates

### P8.1 — `ppo_walking_v0210_smoke1_lateral_yaw.yaml`

**File**: `/Users/ygli/projects/wildrobot/training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`

Clone `ppo_walking_v0201_smoke14.yaml` and apply diffs:
```yaml
env:
  obs_version: wr_obs_v8_cmd3d              # was wr_obs_v7_phase_proprio
  min_velocity: 0.18
  max_velocity: 0.26
  min_velocity_y: -0.13                      # NEW
  max_velocity_y:  0.13                      # NEW
  max_yaw_rate:    0.25                      # NEW
  cmd_zero_chance: 0.20
  cmd_turn_chance: 0.20                      # was reserved/unused; now active
  cmd_deadzone: [0.02, 0.02, 0.05]           # 3D deadzone

  eval_velocity_cmd:    [0.18, 0.04, 0.10]   # vx_eval, vy_eval, wz_eval

  loc_ref_version: v3b_offline_library_3d    # NEW (was v3_offline_library)
  loc_ref_offline_command_vx_grid: [0.18, 0.22, 0.26]
  loc_ref_offline_command_vy_grid: [-0.10, -0.05, 0.0, 0.05, 0.10]
  loc_ref_offline_command_yaw_rate_grid: [-0.20, -0.10, 0.0, 0.10, 0.20]

  loc_ref_residual_scale_per_joint:
    # (cloned from smoke14, then lateral DoFs * 1.5)
    left_hip_roll:    <smoke14_value> * 1.5
    right_hip_roll:   <smoke14_value> * 1.5
    left_ankle_roll:  <smoke14_value> * 1.5
    right_ankle_roll: <smoke14_value> * 1.5
    left_hip_yaw:     <smoke14_value> * 1.5
    right_hip_yaw:    <smoke14_value> * 1.5
    # ... all other joints unchanged

reward_weights:
  cmd_velocity_track_dim: 2                  # consume vy
  cmd_velocity_track: 2.0
  cmd_yaw_rate_track:  1.5                   # NEW
  # ... other weights from smoke14 unchanged
```

### P8.2 — Add G6 / G7 gates to `walking_training.md`

**File**: `/Users/ygli/projects/wildrobot/training/docs/walking_training.md`

Append a new section under "Gate definitions":
```markdown
### G6 — Lateral command tracking (v0.21.0+)
Definition: |lateral_velocity - vy_cmd| < 0.5 * |vy_cmd| over the last 100 eval steps,
averaged across ≥3 seeds, when eval cmd is (0.18, 0.04, 0.0).

### G7 — Yaw command tracking (v0.21.0+)
Definition: |ang_vel_z - yaw_rate_cmd| < 0.5 * |yaw_rate_cmd| over the last 100 eval
steps, averaged across ≥3 seeds, when eval cmd is (0.18, 0.0, 0.10).

### Promotion criterion (v0.21.0)
G4 + G5 (forward) MUST not regress relative to v0.20.1-smoke14 baseline.
G6 + G7 MUST clear ≥50% tracking on first smoke before increasing vy/wz ranges.
```

### P8.3 — Commit
```bash
git add training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml training/docs/walking_training.md training/tests/test_config_load_v0210_smoke1.py
git commit -m "v0.21.0 P8: smoke1 YAML + G6/G7 gate definitions"
```

---

## P9 — CHANGELOG entry (after smoke results)

### P9.1 — `training/CHANGELOG.md`

**Only after smoke1 run completes and is analyzed**:
```markdown
## v0.21.0-smoke1-lateral-yaw — <ISO date> — wandb: <run id>

**Scope**: First lateral+yaw smoke. Extends ZMP prior to 3 axes; adds obs layout v8; activates dim=2 cmd reward with real vy_cmd; adds yaw_rate tracking term.

**Gates**:
- G4 forward stability: <pass/fail>
- G5 forward step amplitude: <pass/fail>
- G6 lateral tracking: <%>
- G7 yaw tracking: <%>

**Outcomes**:
- <observations from the wandb run>
- <next-version proposals>
```

---

## End-to-end verification

After P9.1 the full pipeline should be:
1. `uv run pytest training/tests/ -q` — all green.
2. `uv run python training/train.py --config training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml --max-steps 500 --seeds 3` — runs without crash for 500 PPO steps.
3. Eval rollout with cmd `(0.18, 0.04, 0.10)` shows visible strafe + turn motion in the rendered video (no black frames).
4. wandb metrics `tracking/lateral_velocity_abs` and `tracking/yaw_rate_err` populate and trend toward zero.
5. Forward G4/G5 metrics within 10% of smoke14 baseline.

---

## Files touched (summary)

**Create**:
- `training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`
- `training/docs/v0210_lateral_yaw_prior_plan.md` (this file)
- `training/tests/test_zmp_walk_config_3d_cmd.py`
- `training/tests/test_reference_library_3d.py`
- `training/tests/test_env_config_3d_cmd.py`
- `training/tests/test_sample_velocity_cmd_3d.py`
- `training/tests/test_obs_v8_cmd3d.py`
- `training/tests/test_cmd_lateral_velocity_tracking.py`
- `training/tests/test_env_step_lateral_yaw_e2e.py`
- `training/tests/test_config_load_v0210_smoke1.py`

**Modify**:
- `control/zmp/zmp_walk.py` — `ZMPWalkConfig`, footstep planner branches
- `control/references/reference_library.py` — `ReferenceLibraryMeta` (+ `vy_grid`, `yaw_rate_grid`), `ReferenceLibrary.lookup`
- `control/references/runtime_reference_service.py` — `to_jax_arrays`, `lookup_jax`
- `training/envs/wildrobot_env.py` — `_init_offline_service`, `_lookup_offline_window`, `_sample_velocity_cmd`, `_cmd_velocity_track_reward` (renamed), `_get_obs`, `_compute_reward_terms`
- `training/configs/training_runtime_config.py` — `EnvConfig` (+ `min/max_velocity_y`, `max_yaw_rate`, `loc_ref_offline_command_vy_grid`, `loc_ref_offline_command_yaw_rate_grid`); `RewardWeightsConfig` (+ `cmd_yaw_rate_track`)
- `policy_contract/jax/obs.py` — register `wr_obs_v8_cmd3d`
- `training/core/metrics_registry.py` — add `tracking/yaw_rate_err`
- `training/docs/walking_training.md` — G6 / G7 sections
- `training/CHANGELOG.md` — v0.21.0 entry (after smoke)

---

## Risks & rollback

- **Risk**: Forward G4/G5 regresses when lateral residual scales are bumped. **Mitigation**: P6 keeps non-lateral joints unchanged; if regression observed, fall back to 1.0× scales on lateral joints.
- **Risk**: 3D lookup table grows too large (3×5×5 = 75 bins × episode_len arrays). **Mitigation**: smoke1 uses small grids (3 vx × 5 vy × 5 wz = 75). If memory-bound, drop wz_grid to 3 entries.
- **Risk**: Obs version bump breaks downstream eval/inference. **Mitigation**: v7 / v6 layouts untouched; old configs continue to work. New `loc_ref_version: v3b_offline_library_3d` is opt-in.
- **Rollback**: revert by switching smoke YAML back to `wr_obs_v7_phase_proprio` and `loc_ref_version: v3_offline_library`. All v0.21 code paths default to vy=wz=0 when grids are `(0.0,)`.
