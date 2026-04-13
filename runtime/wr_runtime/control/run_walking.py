"""Walking-mode inference loop for v0.19.4 (nominal) and v0.19.5 (residual PPO).

This is the runtime entry point for walking deployment.  It supports two modes:

- nominal_only (v0.19.4): walking reference → IK → servo targets.  No neural
  network required.
- residual_ppo (v0.19.5): walking reference → IK → q_ref → ONNX policy →
  q_ref + delta_q → servo targets.

Usage::

    python -m runtime.wr_runtime.control.run_walking --config runtime/configs/walking_v0194.json

Config format (JSON):

    {
      "controller_mode": "nominal_only",  // or "residual_ppo"
      "forward_speed_mps": 0.15,
      "policy_path": null,                // required for residual_ppo
      "dt_s": 0.02,
      "ref_config": { ... },              // WalkingRefV2Config overrides
      "ik_config": { ... }                // NominalIkConfig overrides
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from control.locomotion.walking_controller import (
    ControllerMode,
    WalkingController,
    WalkingControllerConfig,
)
from control.locomotion.nominal_ik_adapter import NominalIkConfig
from control.references.walking_ref_v2 import WalkingRefV2Config


def load_walking_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a walking runtime config JSON file."""
    with open(config_path) as f:
        cfg = json.load(f)
    if "controller_mode" not in cfg:
        raise ValueError("Config must specify 'controller_mode'")
    mode = cfg["controller_mode"]
    if mode not in ("nominal_only", "residual_ppo"):
        raise ValueError(f"Unknown controller_mode: {mode!r}")
    if mode == "residual_ppo" and not cfg.get("policy_path"):
        raise ValueError("residual_ppo mode requires 'policy_path'")
    return cfg


def build_controller_from_config(cfg: Dict[str, Any]) -> WalkingController:
    """Build a WalkingController from a JSON config dict."""
    mode = ControllerMode(cfg["controller_mode"])

    ref_overrides = cfg.get("ref_config", {})
    ik_overrides = cfg.get("ik_config", {})

    ref_config = WalkingRefV2Config(**ref_overrides)
    ik_config = NominalIkConfig(**ik_overrides)

    controller_config = WalkingControllerConfig(
        mode=mode,
        ref_config=ref_config,
        ik_config=ik_config,
        residual_scale=cfg.get("residual_scale", 0.18),
        dt_s=cfg.get("dt_s", 0.02),
    )

    return WalkingController(controller_config)


def run_walking_loop(
    controller: WalkingController,
    *,
    forward_speed_mps: float = 0.15,
    max_steps: int = 500,
    policy_fn=None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the walking loop (dry_run=True for testing without hardware).

    Args:
        controller: Configured WalkingController.
        forward_speed_mps: Commanded forward speed.
        max_steps: Maximum steps to run.
        policy_fn: Optional callable(obs) → delta_q for residual_ppo mode.
        dry_run: If True, simulates sensor readings instead of reading hardware.

    Returns:
        Summary dict with step count and timing info.
    """
    controller.reset()
    dt = controller.config.dt_s
    step_count = 0
    q_ref_log = []

    for step_i in range(max_steps):
        t0 = time.monotonic()

        if dry_run:
            # Simulate sensor readings
            root_pitch = 0.0
            root_pitch_rate = 0.0
            left_loaded = True
            right_loaded = step_i % 50 > 25  # Alternate contacts
        else:
            # Hardware reads would go here
            raise NotImplementedError("Hardware mode not yet implemented")

        out = controller.step(
            forward_speed_mps=forward_speed_mps,
            root_pitch_rad=root_pitch,
            root_pitch_rate_rad_s=root_pitch_rate,
            left_foot_loaded=left_loaded,
            right_foot_loaded=right_loaded,
        )

        if controller.mode == ControllerMode.RESIDUAL_PPO and policy_fn is not None:
            # Build obs, run policy, compose residual
            delta_q = policy_fn(out.q_ref)
            q_target = controller.compose_residual(out.q_ref, delta_q)
        else:
            q_target = out.q_target

        q_ref_log.append(out.q_ref.copy())

        if not dry_run:
            # Hardware writes would go here
            pass

        step_count += 1
        elapsed = time.monotonic() - t0
        sleep_time = max(0.0, dt - elapsed)
        if not dry_run and sleep_time > 0:
            time.sleep(sleep_time)

    return {
        "steps": step_count,
        "mode": controller.mode.value,
        "q_ref_log": np.array(q_ref_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WildRobot walking controller")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to walking config JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware (simulated sensors)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps to run",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Override forward speed (m/s)",
    )
    args = parser.parse_args()

    cfg = load_walking_config(args.config)
    controller = build_controller_from_config(cfg)

    speed = args.speed if args.speed is not None else cfg.get("forward_speed_mps", 0.15)

    policy_fn = None
    if controller.mode == ControllerMode.RESIDUAL_PPO:
        policy_path = cfg["policy_path"]
        if not Path(policy_path).exists():
            print(f"Error: policy not found at {policy_path}", file=sys.stderr)
            sys.exit(1)
        try:
            from runtime.wr_runtime.inference.onnx_policy import OnnxPolicy

            policy = OnnxPolicy(policy_path)
            # For residual mode, the policy obs is built externally
            # This is a placeholder — full obs building requires the env's
            # observation builder
            print(f"Loaded ONNX policy from {policy_path}")
        except ImportError:
            print("Warning: onnxruntime not available, running nominal-only")
            policy_fn = None

    print(f"Walking controller: mode={controller.mode.value}, speed={speed:.2f} m/s")
    result = run_walking_loop(
        controller,
        forward_speed_mps=speed,
        max_steps=args.max_steps,
        policy_fn=policy_fn,
        dry_run=args.dry_run,
    )
    print(f"Completed {result['steps']} steps in {result['mode']} mode")


if __name__ == "__main__":
    main()
