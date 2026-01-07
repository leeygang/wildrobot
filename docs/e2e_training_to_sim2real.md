E2E Training to Sim2Real Runbook
================================

Goal: train a policy, validate it, export a runtime bundle, and prepare for
sim2real deployment using the policy_contract.

Prerequisites
-------------
- Python 3.12+
- `uv` installed
- `assets/robot_config.yaml` generated (run `cd assets && python post_process.py` if needed)

0) Environment Setup
--------------------
1) Create or update the virtual environment:
```
uv sync
```

2) Validate training setup:
```
uv run python scripts/validate_training_setup.py
```

1) Train (Stage 1 PPO)
----------------------
Standing:
```
uv run python training/train.py --config training/configs/ppo_standing.yaml --no-amp
```

Walking (after standing is stable):
```
uv run python training/train.py --config training/configs/ppo_walking.yaml --no-amp
```

Quick smoke test:
```
uv run python training/train.py --config training/configs/ppo_walking.yaml --no-amp --verify
```

2) Evaluate (Headless, MJX)
---------------------------
Quantitative gate (deterministic by default):
```
uv run python training/scripts/eval_policy.py \
  --checkpoint <checkpoint.pkl> \
  --config training/configs/ppo_standing.yaml \
  --num-steps 500 \
  --compare-metrics training/wandb/<run>/files/metrics.jsonl
```

Notes:
- Use `--stochastic` to match training noise.
- Compare `reward_per_step` first; reward sums depend on rollout length.

3) Visualize (Native MuJoCo Viewer)
-----------------------------------
Interactive:
```
uv run mjpython training/training/visualize_policy.py \
  --checkpoint <checkpoint.pkl> \
  --config training/configs/ppo_standing.yaml
```

Headless (single episode):
```
uv run python training/training/visualize_policy.py \
  --headless --num-episodes 1 \
  --checkpoint <checkpoint.pkl> \
  --config training/configs/ppo_standing.yaml
```

4) Export Runtime Bundle (ONNX + Spec)
--------------------------------------
```
uv run python training/scripts/export_policy_bundle.py \
  --checkpoint <checkpoint.pkl> \
  --config training/configs/ppo_standing.yaml \
  --output-dir training/checkpoints/wildrobot_policy_bundle
```

This produces:
- `policy.onnx`
- `policy_spec.json`
- `checksums.json`
- `robot_config.yaml` snapshot

5) Validate Bundle (Static Pose Check)
--------------------------------------
```
uv run python runtime/wr_runtime/scripts/check_pose.py \
  --bundle training/checkpoints/wildrobot_policy_bundle
```

Expected: round-trip error << 1e-4 rad.

6) Runtime Readiness Checklist
------------------------------
- Signals:
  - IMU quat + gyro wired (policy uses gravity + angular velocity).
  - Foot switches wired (4 touch sensors).
  - Joint encoders ordered as `spec.robot.actuator_names`.
- Safety:
  - Action clamping to joint limits (or tighter).
  - E-stop on tilt, comms loss, or loop deadline misses.
  - Apply `lowpass_v1.alpha` exactly as exported (no runtime overrides).
- Calibration:
  - Verify joint offsets and mirror signs.
  - Use `spec.robot.home_ctrl_rad` as the canonical safe pose.

7) Sim2Real Smoke Test (Hardware)
---------------------------------
1) Dry run off-ground (feet in air):
   - Verify no violent output and correct sensor wiring.
2) Low-torque ground contact:
   - Verify balance without drift; enable e-stop.

Common Troubleshooting
----------------------
- `ModuleNotFoundError` on scripts: ensure you run from repo root with `uv run ...`.
- Reward mismatch: compare `reward_per_step`, not reward sum.
- Success rate 0 during eval: increase `--num-steps` to full episode length (e.g., 500).

Artifacts to Track
------------------
- Training run ID: `training/wandb/<run>`
- Checkpoint: `training/checkpoints/<run>/checkpoint_*.pkl`
- Bundle: `training/checkpoints/wildrobot_policy_bundle/`
