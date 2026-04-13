#!/usr/bin/env python3
"""Walking-mode inference loop for v0.19.4 (nominal) and v0.19.5 (residual PPO).

This is the runtime entry point for walking deployment.  It supports two modes:

- nominal_only (v0.19.4): walking reference -> IK -> servo targets.
- residual_ppo (v0.19.5): walking reference -> IK -> q_ref -> ONNX policy ->
  q_ref + delta_q -> servo targets.

Usage::

    # From repo root (no dependency on runtime.wr_runtime.__init__):
    uv run python runtime/wr_runtime/control/run_walking.py \\
        --config runtime/configs/walking_v0194.json --dry-run

Config format (JSON):

    {
      "controller_mode": "nominal_only",  // or "residual_ppo"
      "forward_speed_mps": 0.15,
      "policy_path": null,                // required for residual_ppo
      "dt_s": 0.02,
      "residual_scale": 0.18,
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

# Ensure repo root is on sys.path so control/ imports work when run as a script.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the walking loop (dry_run=True for testing without hardware).

    Currently nominal-only.  Residual PPO support will be added after
    a trained checkpoint exists (see run_walking.py main() guard).
    """
    controller.reset()
    dt = controller.config.dt_s
    q_ref_log = []

    for step_i in range(max_steps):
        t0 = time.monotonic()

        if dry_run:
            root_pitch = 0.0
            root_pitch_rate = 0.0
            left_loaded = True
            right_loaded = step_i % 50 > 25
        else:
            raise NotImplementedError("Hardware mode not yet implemented")

        out = controller.step(
            forward_speed_mps=forward_speed_mps,
            root_pitch_rad=root_pitch,
            root_pitch_rate_rad_s=root_pitch_rate,
            left_foot_loaded=left_loaded,
            right_foot_loaded=right_loaded,
        )

        q_ref_log.append(out.q_ref.copy())

        if not dry_run:
            pass  # Hardware writes go here

        elapsed = time.monotonic() - t0
        sleep_time = max(0.0, dt - elapsed)
        if not dry_run and sleep_time > 0:
            time.sleep(sleep_time)

    return {
        "steps": max_steps,
        "mode": controller.mode.value,
        "q_ref_log": np.array(q_ref_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WildRobot walking controller")
    parser.add_argument("--config", type=str, required=True, help="Path to walking config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Run without hardware")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps to run")
    parser.add_argument("--speed", type=float, default=None, help="Override forward speed (m/s)")
    args = parser.parse_args()

    cfg = load_walking_config(args.config)
    controller = build_controller_from_config(cfg)

    speed = args.speed if args.speed is not None else cfg.get("forward_speed_mps", 0.15)

    policy_fn = None
    if controller.mode == ControllerMode.RESIDUAL_PPO:
        raise NotImplementedError(
            "Residual PPO runtime is deferred until a trained checkpoint exists.\n"
            "Deployment checklist (see training/docs/walking_training.md v0.19.5):\n"
            "  1. Train with ppo_walking_v0195.yaml\n"
            "  2. Export bundle with training/exports/export_policy_bundle_cli.py\n"
            "  3. Read obs_dim/action_dim from the bundle's policy_spec.json\n"
            "  4. Build obs matching the exported spec (not hardcoded dims)\n"
            "  5. Store prev_action as filtered composed action (not raw delta_q)\n"
            "Use --config with controller_mode=nominal_only for v0.19.4 deployment."
        )

    print(f"Walking controller: mode={controller.mode.value}, speed={speed:.2f} m/s")
    result = run_walking_loop(
        controller,
        forward_speed_mps=speed,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
    )
    print(f"Completed {result['steps']} steps in {result['mode']} mode")


if __name__ == "__main__":
    main()
