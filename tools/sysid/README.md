# tools/sysid

`v0.19.1` adds a usable capture path for representative joint SysID traces.

## Capture commands

Step response:

```bash
uv run python tools/sysid/run_capture.py \
  --mode step \
  --joint-name left_knee_pitch \
  --runtime-config runtime/configs/runtime_config_v2.json \
  --output-dir runtime/logs/sysid
```

Hold response:

```bash
uv run python tools/sysid/run_capture.py \
  --mode hold \
  --joint-name left_knee_pitch \
  --hold-rad 0.15 \
  --output-dir runtime/logs/sysid
```

Chirp response:

```bash
uv run python tools/sysid/run_capture.py \
  --mode chirp \
  --joint-name left_knee_pitch \
  --amplitude-rad 0.2 \
  --chirp-start-hz 0.2 \
  --chirp-end-hz 3.0 \
  --output-dir runtime/logs/sysid
```

By default, this commands real hardware through the configured Hiwonder servo board and records measured readback.

Offline/synthetic fallback is still available for CI/dev:

```bash
uv run python tools/sysid/run_capture.py \
  --capture-source synthetic \
  --mode step \
  --joint-name left_knee_pitch \
  --output-dir runtime/logs/sysid
```

## Outputs

Each run writes:

- `<prefix>_<mode>_<joint>.npz` with arrays:
  - `command_rad`
  - `measured_position_rad`
  - `measured_velocity_rad_s`
  - `timestamps_s`
- `<prefix>_<mode>_<joint>.json` manifest with metadata:
  - joint name
  - mode
  - sample rate
  - runtime config path
  - asset/runtime context
  - realism profile reference (when configured)

## Notes

- Milestone `v0.19.1` now includes a real measurement loop (`--capture-source hardware`, default) plus structured exports.
- Full parameter fitting/optimization loops remain intentionally deferred to `v0.19.2+`.
