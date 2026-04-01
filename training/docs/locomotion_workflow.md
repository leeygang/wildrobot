# Locomotion Workflow (`v0.19.0`)

This is the canonical practical flow for locomotion-first work on current WildRobot.

Standing is now a regression gate. Active feature work is locomotion-oriented.

## 1) Train (locomotion stack entrypoint)

Use the standard training entrypoint:

```bash
uv run python training/train.py --config training/configs/ppo_walking.yaml
```

For quick smoke checks:

```bash
uv run python training/train.py --config training/configs/ppo_walking.yaml --verify
```

## 2) Evaluate

Use deterministic policy eval:

```bash
uv run python training/eval/eval_policy.py --checkpoint <ckpt.pkl> --config training/configs/ppo_walking.yaml
```

Optional visualization pass:

```bash
uv run python training/eval/visualize_policy.py --checkpoint <ckpt.pkl> --config training/configs/ppo_walking.yaml --headless --num-episodes 1
```

Use fixed-ladder standing regression to prevent regressions while locomotion evolves:

```bash
uv run python training/eval/eval_ladder_v0170.py --checkpoint <ckpt.pkl> --config training/configs/ppo_walking.yaml
```

## 3) Export / Deploy

Export a bundle:

```bash
uv run python training/exports/export_policy_bundle_cli.py --checkpoint <ckpt.pkl> --training-config training/configs/ppo_walking.yaml --bundle-path training/checkpoints/<bundle_name>
```

Run on hardware:

```bash
cd runtime
uv run wildrobot-run-policy --bundle ../training/checkpoints/<bundle_name>
```

The deployed artifact remains a learned policy.

## 4) Replay / Calibration loop

Capture calibration and replay logs:

```bash
cd runtime
uv run python scripts/calibrate.py --runtime-config ~/.wildrobot/config.json
uv run wildrobot-run-policy --bundle ../training/checkpoints/<bundle_name> --log-path runtime/logs/<run>.npz --log-steps 2000
uv run wildrobot-inspect-log --input runtime/logs/<run>.npz
```

Milestone 0 scaffold tools:

```bash
uv run python tools/sysid/run_capture_stub.py --output-dir runtime/logs/sysid
uv run python tools/sim2real_eval/replay_compare_stub.py --sim-log <sim.npz> --real-log <real.npz>
```
