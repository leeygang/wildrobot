# System Architecture: Actuator Ordering

**Last updated:** 2026-04-12

---

## Problem

The project has two independent actuator orderings:

- **MuJoCo model order** (from XML): groups joints by body side  
  `left_hip_pitch, left_hip_roll, left_knee_pitch, ..., right_hip_pitch, ...`

- **PolicySpec order** (from `mujoco_robot_config.json`): interleaves L/R pairs  
  `left_hip_pitch, right_hip_pitch, left_hip_roll, right_hip_roll, ...`

If ctrl values are written in the wrong order, each actuator receives the
wrong target — e.g., the knee bend command goes to the hip pitch actuator.
This was the root cause of all v0.19.3+ loc_ref training failures.

---

## Principle: Single Ctrl-Write API

**All code that writes to actuator ctrl must use `CtrlOrderMapper`.**

There must be exactly one code path for converting PolicySpec-ordered ctrl
arrays into MuJoCo-ordered ctrl arrays.  No inline `mj_data.ctrl[i] = ...`
loops that assume a particular ordering.

This applies to:

- **Env** (JAX/MJX): `env._ctrl_mapper.to_mj_jax(ctrl_policy_order)`
- **Viewer** (MuJoCo C API): `ctrl_mapper.set_ctrl_from_qpos(mj_data)` or
  `ctrl_mapper.set_all_ctrl(mj_data, ctrl_policy_order)`
- **Eval scripts**: same `CtrlOrderMapper` instance

---

## API: `CtrlOrderMapper`

**Location:** `training/utils/ctrl_order.py`

```python
from training.utils.ctrl_order import CtrlOrderMapper

mapper = CtrlOrderMapper(mj_model, policy_spec.robot.actuator_names)
```

### JAX path (env, training)

```python
mj_ctrl = mapper.to_mj_jax(ctrl_policy_order)  # jax.Array → jax.Array
data = data.replace(ctrl=mj_ctrl)
```

### NumPy path (viewer, eval)

```python
mj_ctrl = mapper.to_mj_np(ctrl_policy_order)    # np.ndarray → np.ndarray
mapper.set_all_ctrl(mj_data, ctrl_policy_order)  # write directly to MjData
mapper.set_ctrl_from_qpos(mj_data)               # set ctrl = current qpos
mapper.set_ctrl_by_name(mj_data, "left_knee_pitch", 0.99)  # single actuator
```

### Properties

```python
mapper.policy_to_mj_order       # np.ndarray: perm[policy_idx] = mj_ctrl_idx
mapper.policy_to_mj_order_jax   # jax.Array: same, for JIT contexts
```

---

## Internal Ordering Convention

| Array | Ordering | Notes |
|---|---|---|
| `nominal_q_ref` | PolicySpec | IK output, walking reference |
| `_default_joint_qpos` | PolicySpec | Home pose joint values |
| `_idx_*` variables | PolicySpec | `_actuator_name_to_index` indices |
| `_actuator_qpos_addrs` | PolicySpec | But values are MuJoCo qpos addresses (name-based lookup) |
| `action_to_ctrl()` output | PolicySpec | Policy contract operates in PolicySpec order |
| `data.ctrl` | MuJoCo | **Must be permuted before writing** |
| `data.qpos` | MuJoCo | Read via `_actuator_qpos_addrs` (correct) |

---

## Regression Test

`training/tests/test_actuator_ordering.py` — 5 tests:

1. Permutation validity (unique, covers 0..nu-1)
2. All PolicySpec names exist in MuJoCo model
3. `to_mj_ctrl` maps each value to the correct MuJoCo slot
4. `qpos` read consistency via `_actuator_qpos_addrs`
5. After reset + step, `data.ctrl[mj_id]` matches `nominal_q_ref[policy_idx]`

Run: `JAX_PLATFORMS=cpu uv run python -m pytest training/tests/test_actuator_ordering.py -v`
