#!/usr/bin/env python3
"""Compare checkpoint actions against runtime ONNX on a native-MuJoCo trace.

This diagnostic answers one narrow deploy question: given the exact observation
vectors produced by the native-MuJoCo eval adapter, does the exported runtime
ONNX policy emit the same deterministic raw action as the training checkpoint?

If the actions match on a walking sim trace, export/runtime inference is not the
reason hardware remains planted; the next evidence target is hardware
observations, contact state, or actuation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from policy_contract.spec import PolicyBundle
from wr_runtime.inference.onnx_policy import OnnxPolicy
from training.cal.cal import ControlAbstractionLayer
from training.configs.training_config import load_robot_config, load_training_config
from training.eval.v6_eval_adapter import V6EvalAdapter
from training.eval.visualize_policy import (
    _build_policy_spec,
    _network_activation_name,
    _validate_user_fixed_velocity_cmd,
)
from training.algos.ppo.ppo_core import create_networks, sample_actions
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter


DEFAULT_BUNDLE = (
    PROJECT_ROOT / "runtime" / "bundles" / "walking_v0210_smoke6_ckpt1650"
)
LEG_JOINTS = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
)
LR_PAIRS = (
    ("HP", "left_hip_pitch", "right_hip_pitch"),
    ("HR", "left_hip_roll", "right_hip_roll"),
    ("K", "left_knee_pitch", "right_knee_pitch"),
    ("AP", "left_ankle_pitch", "right_ankle_pitch"),
    ("AR", "left_ankle_roll", "right_ankle_roll"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic native-MuJoCo checkpoint trace and compare "
            "checkpoint raw actions to the runtime ONNX bundle."
        )
    )
    parser.add_argument("--bundle", type=str, default=str(DEFAULT_BUNDLE))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--velocity-cmd",
        type=float,
        nargs=3,
        default=[0.20, 0.0, 0.0],
        metavar=("VX", "VY", "WZ"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--reset-noise",
        action="store_true",
        help="Enable native-eval reset noise/DR. Default is deterministic reset.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Optional path for compressed obs/action/contact trace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_path = _resolve_path(args.bundle)
    bundle = PolicyBundle.load(bundle_path)
    provenance = _load_bundle_provenance(bundle_path)

    config_path = _resolve_optional_path(
        args.config,
        fallback=provenance.get("training_config"),
        label="training config",
    )
    checkpoint_path = _resolve_optional_path(
        args.checkpoint,
        fallback=str(bundle_path / "checkpoint.pkl")
        if (bundle_path / "checkpoint.pkl").exists()
        else _source_checkpoint_path(provenance),
        label="checkpoint",
    )

    print("Sim trace ONNX parity")
    print(f"  bundle:     {bundle_path}")
    print(f"  onnx:       {bundle.model_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config:     {config_path}")

    training_cfg = load_training_config(config_path)
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    robot_cfg = load_robot_config(robot_cfg_path)

    action_filter_alpha = float(training_cfg.env.action_filter_alpha)
    policy_spec = _build_policy_spec(training_cfg, robot_cfg, action_filter_alpha)
    _validate_contract_match(policy_spec, bundle)

    velocity_cmd = _validate_user_fixed_velocity_cmd(
        training_cfg, np.asarray(args.velocity_cmd, dtype=np.float32)
    )
    print(
        "  cmd:        "
        f"vx={velocity_cmd[0]:.3f} vy={velocity_cmd[1]:.3f} wz={velocity_cmd[2]:.3f}"
    )
    print(
        "  reset:      "
        f"{'with noise/DR' if args.reset_noise else 'deterministic no-noise'}"
    )

    checkpoint_policy = _load_checkpoint_policy(
        checkpoint_path=checkpoint_path,
        obs_dim=int(policy_spec.model.obs_dim),
        action_dim=int(policy_spec.model.action_dim),
        training_cfg=training_cfg,
    )
    onnx_policy = OnnxPolicy(
        str(bundle.model_path),
        input_name=bundle.spec.model.input_name,
        output_name=bundle.spec.model.output_name,
        expected_obs_dim=int(bundle.spec.model.obs_dim),
        expected_action_dim=int(bundle.spec.model.action_dim),
    )

    mj_model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / training_cfg.env.model_path))
    mj_data = mujoco.MjData(mj_model)
    cal = ControlAbstractionLayer(mj_model, robot_cfg)
    signals_adapter = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        policy_spec=policy_spec,
        foot_switch_threshold=training_cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=training_cfg,
        mj_model=mj_model,
        policy_spec=policy_spec,
        signals_adapter=signals_adapter,
        action_dim=int(policy_spec.model.action_dim),
    )

    rng = np.random.default_rng(int(args.seed))
    adapter.reset_native_mj_state(
        mj_data,
        apply_noise=bool(args.reset_noise),
        rng=rng,
        perturb_pose=bool(args.reset_noise),
        apply_dr=bool(args.reset_noise),
    )

    ctrl_dt = float(training_cfg.env.ctrl_dt)
    n_substeps = max(1, int(round(ctrl_dt / float(mj_model.opt.timestep))))
    actuator_names = list(policy_spec.robot.actuator_names)
    leg_indices = _indices(actuator_names, LEG_JOINTS)

    obs_trace: list[np.ndarray] = []
    checkpoint_actions: list[np.ndarray] = []
    onnx_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    contacts_trace: list[np.ndarray] = []
    diffs: list[float] = []

    first_bad_step: int | None = None
    for step in range(int(args.steps)):
        obs = adapter.compute_obs(mj_data=mj_data, velocity_cmd=velocity_cmd)
        checkpoint_action = checkpoint_policy(obs)
        onnx_action = onnx_policy.predict(obs)
        diff = float(np.max(np.abs(checkpoint_action - onnx_action)))
        if first_bad_step is None and diff > float(args.tolerance):
            first_bad_step = step

        applied = adapter.apply_action(mj_data, checkpoint_action)
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)
        adapter.post_physics(mj_data)

        contacts = np.asarray(
            cal.get_geom_based_foot_contacts(mj_data, normalize=False),
            dtype=np.float32,
        ).reshape(-1)

        obs_trace.append(np.asarray(obs, dtype=np.float32).copy())
        checkpoint_actions.append(np.asarray(checkpoint_action, dtype=np.float32).copy())
        onnx_actions.append(np.asarray(onnx_action, dtype=np.float32).copy())
        applied_actions.append(np.asarray(applied, dtype=np.float32).copy())
        contacts_trace.append(contacts.copy())
        diffs.append(diff)

        if args.print_every > 0 and (
            step % int(args.print_every) == 0 or step == int(args.steps) - 1
        ):
            print(
                f"[step {step:4d}] "
                f"diff|max={diff:.6g} "
                f"chk|raw|max={_max_abs(checkpoint_action):.3f} "
                f"onnx|raw|max={_max_abs(onnx_action):.3f} "
                f"leg|chk|max={_max_abs(checkpoint_action[leg_indices]):.3f} "
                f"{_format_lr_delta(checkpoint_action, actuator_names)} "
                f"foot_f={np.round(contacts, 1).tolist()}"
            )

    obs_arr = np.asarray(obs_trace, dtype=np.float32)
    checkpoint_arr = np.asarray(checkpoint_actions, dtype=np.float32)
    onnx_arr = np.asarray(onnx_actions, dtype=np.float32)
    applied_arr = np.asarray(applied_actions, dtype=np.float32)
    contacts_arr = np.asarray(contacts_trace, dtype=np.float32)
    diff_arr = np.asarray(diffs, dtype=np.float32)

    if args.output_npz:
        out_path = _resolve_output_path(args.output_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            observations=obs_arr,
            checkpoint_actions=checkpoint_arr,
            onnx_actions=onnx_arr,
            applied_actions=applied_arr,
            contacts=contacts_arr,
            velocity_cmd=velocity_cmd,
        )
        print(f"  wrote trace: {out_path}")

    print("\nSummary:")
    print(f"  steps: {len(diff_arr)}")
    print(f"  action_diff|max: {float(np.max(diff_arr)):.9g}")
    print(f"  action_diff|p95: {float(np.percentile(diff_arr, 95.0)):.9g}")
    print(f"  action_diff|mean: {float(np.mean(diff_arr)):.9g}")
    print(f"  checkpoint_raw|max: {_max_abs(checkpoint_arr):.6g}")
    print(f"  checkpoint_leg_raw|p95: {_p95_abs(checkpoint_arr[:, leg_indices]):.6g}")
    print(f"  applied_leg|p95: {_p95_abs(applied_arr[:, leg_indices]):.6g}")
    print(f"  contact_force|max: {_max_abs(contacts_arr):.6g}")
    if first_bad_step is None:
        print(
            "Result: PASS. ONNX matches checkpoint on this sim trace within "
            f"tolerance {float(args.tolerance):.1e}."
        )
        return 0

    print(
        "Result: FAIL. ONNX diverged from checkpoint at "
        f"step {first_bad_step} with tolerance {float(args.tolerance):.1e}."
    )
    return 2


def _load_checkpoint_policy(*, checkpoint_path: Path, obs_dim: int, action_dim: int, training_cfg):
    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)
    policy_hidden = tuple(training_cfg.networks.actor.hidden_sizes)
    value_hidden = tuple(training_cfg.networks.critic.hidden_sizes)
    activation = _network_activation_name(training_cfg)
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=policy_hidden,
        value_hidden_dims=value_hidden,
        activation=activation,
    )
    policy_params = checkpoint["policy_params"]
    processor_params = checkpoint.get("processor_params", ())
    if processor_params is None:
        processor_params = ()

    @jax.jit
    def _get_action(obs):
        obs_batch = obs[None, ...]
        action, _, _ = sample_actions(
            processor_params,
            policy_params,
            ppo_network,
            obs_batch,
            jax.random.PRNGKey(0),
            True,
        )
        return action[0]

    _ = _get_action(jnp.zeros((obs_dim,), dtype=jnp.float32))

    def _predict(obs_np: np.ndarray) -> np.ndarray:
        return np.asarray(_get_action(jnp.asarray(obs_np, dtype=jnp.float32)), dtype=np.float32)

    return _predict


def _load_bundle_provenance(bundle_path: Path) -> dict:
    spec_path = bundle_path / "policy_spec.json"
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    provenance = data.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _source_checkpoint_path(provenance: dict) -> str | None:
    source = provenance.get("source_checkpoint")
    if isinstance(source, dict):
        path = source.get("path")
        if path:
            return str(path)
    return None


def _resolve_optional_path(value: str | None, *, fallback: str | None, label: str) -> Path:
    candidate = value or fallback
    if not candidate:
        raise FileNotFoundError(f"No {label} path was provided or found in provenance.")
    return _resolve_path(candidate)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {value}")


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _validate_contract_match(policy_spec, bundle: PolicyBundle) -> None:
    errors = []
    if policy_spec.observation.layout_id != bundle.spec.observation.layout_id:
        errors.append(
            "layout_id "
            f"sim={policy_spec.observation.layout_id!r} "
            f"bundle={bundle.spec.observation.layout_id!r}"
        )
    if int(policy_spec.model.obs_dim) != int(bundle.spec.model.obs_dim):
        errors.append(
            f"obs_dim sim={policy_spec.model.obs_dim} bundle={bundle.spec.model.obs_dim}"
        )
    if int(policy_spec.model.action_dim) != int(bundle.spec.model.action_dim):
        errors.append(
            "action_dim "
            f"sim={policy_spec.model.action_dim} bundle={bundle.spec.model.action_dim}"
        )
    if list(policy_spec.robot.actuator_names) != list(bundle.spec.robot.actuator_names):
        errors.append("actuator_names differ between sim config and bundle spec")
    if errors:
        raise ValueError("Sim/bundle contract mismatch: " + "; ".join(errors))


def _indices(names: list[str], targets: tuple[str, ...]) -> list[int]:
    by_name = {name: idx for idx, name in enumerate(names)}
    return [by_name[name] for name in targets if name in by_name]


def _max_abs(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _p95_abs(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(np.abs(arr), 95.0))


def _format_lr_delta(action: np.ndarray, actuator_names: list[str]) -> str:
    by_name = {name: idx for idx, name in enumerate(actuator_names)}
    parts = []
    for short, left_name, right_name in LR_PAIRS:
        li = by_name.get(left_name)
        ri = by_name.get(right_name)
        if li is None or ri is None:
            continue
        parts.append(f"{short}={float(action[li] - action[ri]):+.3f}")
    return "raw_lr=[" + " ".join(parts) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
