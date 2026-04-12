"""Test actuator ordering consistency between PolicySpec and MuJoCo model.

Regression test for the ctrl-ordering bug: PolicySpec and MuJoCo may list
actuators in different orders.  The env must permute ctrl values so that each
MuJoCo actuator receives the correct target.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from training.configs.training_config import load_training_config, load_robot_config
from training.envs.wildrobot_env import WildRobotEnv
from training.envs.env_info import WR_INFO_KEY
from training.utils.ctrl_order import CtrlOrderMapper

WALKING_CONFIG = PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0193a.yaml"
ROBOT_CONFIG = PROJECT_ROOT / "assets" / "v2" / "mujoco_robot_config.json"


@pytest.fixture(scope="module")
def env():
    cfg = load_training_config(str(WALKING_CONFIG))
    load_robot_config(str(ROBOT_CONFIG))
    cfg.freeze()
    return WildRobotEnv(config=cfg)


@pytest.fixture(scope="module")
def ctrl_mapper(env):
    return env._ctrl_mapper


# ── Test 1: permutation is valid ──────────────────────────────────────────


def test_permutation_is_valid(ctrl_mapper):
    """_policy_to_mj_ctrl_order must be a valid permutation of 0..nu-1."""
    perm = ctrl_mapper.policy_to_mj_order
    assert set(perm.tolist()) == set(range(len(perm)))


# ── Test 2: every PolicySpec name exists in MuJoCo ────────────────────────


def test_all_policy_names_in_mujoco(env):
    """Every actuator name in PolicySpec must exist in the MuJoCo model."""
    for name in env._policy_spec.robot.actuator_names:
        mj_id = mujoco.mj_name2id(
            env._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        assert mj_id >= 0, f"Actuator '{name}' not found in MuJoCo model"


# ── Test 3: _to_mj_ctrl maps to correct MuJoCo indices ───────────────────


def test_to_mj_ctrl_correctness(ctrl_mapper, env):
    """Setting a unique value per PolicySpec index must land at the right
    MuJoCo ctrl slot."""
    nu = env._mj_model.nu
    policy_ctrl = jnp.arange(1, nu + 1, dtype=jnp.float32)
    mj_ctrl = np.asarray(ctrl_mapper.to_mj_jax(policy_ctrl))

    for policy_idx, name in enumerate(env._policy_spec.robot.actuator_names):
        mj_id = mujoco.mj_name2id(
            env._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        expected_val = float(policy_idx + 1)
        actual_val = float(mj_ctrl[mj_id])
        assert actual_val == pytest.approx(expected_val), (
            f"Actuator '{name}': policy_idx={policy_idx} should map to "
            f"mj_id={mj_id}, expected ctrl={expected_val}, got {actual_val}"
        )


# ── Test 4: qpos read consistency ────────────────────────────────────────


def test_qpos_read_consistency(env):
    """data.qpos[_actuator_qpos_addrs[i]] must read the joint for the i-th
    PolicySpec actuator."""
    state = env.reset(jax.random.PRNGKey(0))
    qpos = np.asarray(state.data.qpos)
    addrs = np.asarray(env._actuator_qpos_addrs)

    for policy_idx, name in enumerate(env._policy_spec.robot.actuator_names):
        # Get the joint address via MuJoCo name lookup.
        act_id = mujoco.mj_name2id(
            env._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        joint_id = int(env._mj_model.actuator_trnid[act_id][0])
        expected_addr = int(env._mj_model.jnt_qposadr[joint_id])
        actual_addr = int(addrs[policy_idx])
        assert actual_addr == expected_addr, (
            f"Actuator '{name}': qpos addr mismatch — "
            f"expected {expected_addr}, got {actual_addr}"
        )


# ── Test 5: ctrl matches nominal_q_ref after one step ────────────────────


def test_ctrl_matches_qref_after_step(env):
    """After reset + 1 step with zero action, data.ctrl[mj_id] must match
    nominal_q_ref[policy_idx] for every actuator (within action-filter
    tolerance)."""
    state = env.reset(jax.random.PRNGKey(0))
    zero_action = jnp.zeros((env.action_size,), dtype=jnp.float32)
    state1 = env.step(state, zero_action)

    ctrl_mj = np.asarray(state1.data.ctrl)
    qref = np.asarray(state1.info[WR_INFO_KEY].nominal_q_ref)

    for policy_idx, name in enumerate(env._policy_spec.robot.actuator_names):
        mj_id = mujoco.mj_name2id(
            env._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        ctrl_val = float(ctrl_mj[mj_id])
        qref_val = float(qref[policy_idx])
        assert ctrl_val == pytest.approx(qref_val, abs=0.05), (
            f"Actuator '{name}': ctrl[mj={mj_id}]={ctrl_val:.4f} != "
            f"q_ref[policy={policy_idx}]={qref_val:.4f}"
        )
