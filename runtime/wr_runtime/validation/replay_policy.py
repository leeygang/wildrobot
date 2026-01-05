from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from policy_contract.numpy.action import postprocess_action
from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PolicyBundle, validate_spec

from ..inference.onnx_policy import OnnxPolicy


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def replay_policy(
    *,
    bundle_path: Path,
    input_npz: Path,
    output_npz: Path,
    limit: Optional[int],
) -> None:
    bundle = PolicyBundle.load(bundle_path)
    spec = bundle.spec
    validate_spec(spec)

    policy = OnnxPolicy(str(bundle.model_path), input_name=spec.model.input_name, output_name=spec.model.output_name)

    data = _load_npz(input_npz)
    quat_xyzw = np.asarray(data["quat_xyzw"], dtype=np.float32)
    gyro_rad_s = np.asarray(data["gyro_rad_s"], dtype=np.float32)
    joint_pos_rad = np.asarray(data["joint_pos_rad"], dtype=np.float32)
    joint_vel_rad_s = np.asarray(data.get("joint_vel_rad_s", np.zeros_like(joint_pos_rad)), dtype=np.float32)
    foot_switches = np.asarray(data["foot_switches"], dtype=np.float32)

    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(f"Expected quat_xyzw shape (T,4), got {quat_xyzw.shape}")
    if gyro_rad_s.shape != quat_xyzw[:, :3].shape:
        raise ValueError(f"Expected gyro_rad_s shape {quat_xyzw[:, :3].shape}, got {gyro_rad_s.shape}")
    if joint_pos_rad.ndim != 2 or joint_pos_rad.shape[1] != spec.model.action_dim:
        raise ValueError(
            f"Expected joint_pos_rad shape (T,{spec.model.action_dim}), got {joint_pos_rad.shape}"
        )
    if joint_vel_rad_s.shape != joint_pos_rad.shape:
        raise ValueError(
            f"Expected joint_vel_rad_s shape {joint_pos_rad.shape}, got {joint_vel_rad_s.shape}"
        )
    if foot_switches.ndim != 2 or foot_switches.shape[1] != 4:
        raise ValueError(f"Expected foot_switches shape (T,4), got {foot_switches.shape}")

    velocity_cmd = data.get("velocity_cmd")
    if velocity_cmd is None:
        velocity_cmd = np.zeros((quat_xyzw.shape[0], 1), dtype=np.float32)
    else:
        velocity_cmd = np.asarray(velocity_cmd, dtype=np.float32)
        if velocity_cmd.ndim == 0:
            velocity_cmd = np.full((quat_xyzw.shape[0], 1), float(velocity_cmd), dtype=np.float32)
        elif velocity_cmd.ndim == 1:
            velocity_cmd = velocity_cmd.reshape(-1, 1)
        if velocity_cmd.shape[0] != quat_xyzw.shape[0]:
            raise ValueError(
                f"velocity_cmd length {velocity_cmd.shape[0]} != T={quat_xyzw.shape[0]}"
            )

    T = quat_xyzw.shape[0] if limit is None else min(limit, quat_xyzw.shape[0])
    obs_out = np.zeros((T, spec.model.obs_dim), dtype=np.float32)
    action_raw_out = np.zeros((T, spec.model.action_dim), dtype=np.float32)
    action_post_out = np.zeros((T, spec.model.action_dim), dtype=np.float32)
    ctrl_out = np.zeros((T, spec.model.action_dim), dtype=np.float32)

    state = PolicyState.init(spec)

    for t in range(T):
        signals = Signals(
            quat_xyzw=quat_xyzw[t],
            gyro_rad_s=gyro_rad_s[t],
            joint_pos_rad=joint_pos_rad[t],
            joint_vel_rad_s=joint_vel_rad_s[t],
            foot_switches=foot_switches[t],
        )

        obs = build_observation(
            spec=spec,
            state=state,
            signals=signals,
            velocity_cmd=velocity_cmd[t],
        )

        action_raw = policy.predict(obs)
        action_post, state = postprocess_action(spec=spec, state=state, action_raw=action_raw)
        ctrl = NumpyCalibOps.action_to_ctrl(spec=spec, action=action_post)

        obs_out[t] = obs
        action_raw_out[t] = action_raw
        action_post_out[t] = action_post
        ctrl_out[t] = ctrl

    np.savez(
        output_npz,
        obs=obs_out,
        action_raw=action_raw_out,
        action_post=action_post_out,
        ctrl_targets_rad=ctrl_out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a policy on logged signals and emit obs/actions.")
    parser.add_argument("--bundle", type=str, required=True, help="Bundle directory (policy_spec.json + policy.onnx)")
    parser.add_argument("--input", type=str, required=True, help="Input .npz with logged signals")
    parser.add_argument("--output", type=str, default="replay_output.npz", help="Output .npz path")
    parser.add_argument("--limit", type=int, default=None, help="Max steps to process")
    args = parser.parse_args()

    replay_policy(
        bundle_path=Path(args.bundle),
        input_npz=Path(args.input),
        output_npz=Path(args.output),
        limit=args.limit,
    )
    print(f"Wrote replay output to {args.output}")


if __name__ == "__main__":
    main()
