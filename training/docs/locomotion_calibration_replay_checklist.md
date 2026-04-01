# Locomotion Calibration and Replay Checklist (`v0.19.0`)

Use this checklist before collecting locomotion transfer evidence.

## A. Joint calibration

- Run runtime joint calibration flow (`runtime/scripts/calibrate.py`).
- Verify servo IDs, offsets, and motor signs are correct.
- Save calibrated runtime config (`~/.wildrobot/config.json` or team-standard path).

## B. IMU sanity checks

- Confirm quaternion stream is finite and normalized.
- Confirm gravity-local Z is stable near upright during quiet stand.
- Confirm gyro baseline is low at rest.

## C. Foot-switch sanity checks

- Verify all four channels (`left_toe`, `left_heel`, `right_toe`, `right_heel`) toggle.
- Verify standing contact pattern is consistent across short repeats.

## D. Standing replay capture

- Run quiet standing replay capture on hardware using the policy runner with logging.
- Save log under `runtime/logs/standing/`.

## E. Squat replay capture

- Run a scripted squat motion or squat-like policy phase with logging.
- Save log under `runtime/logs/squat/`.

## F. Required log fields (shared schema)

Every replay log used for sim-vs-real comparison must include:

- `quat_xyzw`
- `gyro_rad_s`
- `joint_pos_rad`
- `joint_vel_rad_s`
- `foot_switches`
- `velocity_cmd`
- `yaw_rate_cmd`

`velocity_cmd` and `yaw_rate_cmd` must match the per-step command tuple used by
the run (`forward_speed_mps`, `yaw_rate_rps`).

These fields are defined by `runtime/wr_runtime/logging/locomotion_log_schema.py`.

## G. Storage conventions

- Keep raw logs in `runtime/logs/<scenario>/`.
- Keep comparison outputs in `runtime/logs/comparisons/`.
- Preserve metadata: policy bundle path, runtime config path, and git commit.

## H. Standing regression gate reminder

- Run standing regression ladder before tagging locomotion experiments as promotable:

```bash
uv run python training/eval/eval_ladder_v0170.py --checkpoint <ckpt.pkl> --config training/configs/ppo_walking.yaml
```
