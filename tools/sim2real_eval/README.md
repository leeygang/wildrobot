# tools/sim2real_eval

`v0.19.1` adds a practical replay-comparison CLI for actuator realism checks.

## Compare command

```bash
uv run python tools/sim2real_eval/replay_compare.py \
  --real-log runtime/logs/sysid/real_step_left_knee_pitch.npz \
  --sim-log runtime/logs/sysid/sim_step_left_knee_pitch.npz \
  --output-dir runtime/logs/comparisons \
  --summary-json left_knee_step_summary.json \
  --tag left_knee_step
```

## Outputs

- Summary JSON with mismatch metrics:
  - `delay_s`
  - `overshoot_*`
  - `settling_time_s_real`
  - `steady_state_error_*`
  - `position_rmse_rad`
  - `imu_gyro_rmse_rad_s` (when available)
  - `foot_switch_timing_mismatch_frac` (when available)
- Two PNGs:
  - `<tag>_overlay.png`
  - `<tag>_error.png`

## M1 “close enough” practical gate

Use this as a first pass:

- delay magnitude stays within one control step (about `0.02 s` at `50 Hz`)
- position RMSE is materially below commanded amplitude
- overshoot and steady-state error trends are directionally similar between real and sim

This is an engineering gate for transfer readiness, not final transfer hardening.
