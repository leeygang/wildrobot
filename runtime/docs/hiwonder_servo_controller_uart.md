# Legacy Hiwonder LSC Controller UART via FT232

This path is deprecated for policy runtime. `wildrobot-run-policy` now requires
the raw Hiwonder/HTD TTL bus backend (`servo_controller.type:
"hiwonder_ttl_bus"`) through the USB TTL debug board. Keep this note only for
legacy LSC controller-board diagnostics.

This note covers the serial wiring used when a Raspberry Pi 4 talks to the
Hiwonder servo controller through an FT232 USB-to-UART adapter.

## Connection

Wire UART signals crossed:

| FT232 pin | Hiwonder servo controller pin |
|---|---|
| `TXD` | `RXD` |
| `RXD` | `TXD` |
| `GND` | `GND` |

Do not connect FT232 `VCC` unless the specific Hiwonder board documentation says
the UART header needs it. The FT232 is only the serial adapter in this setup.

## Power

The Hiwonder servo controller and servos need their own external power. Do not
power servos from the Pi USB port or from the FT232 adapter.

## Logic Level

Use TTL UART levels, not RS-232 voltage levels. Prefer `3.3V` FT232 logic if the
Hiwonder board accepts it; use `5V` TTL only when the board documentation or
silkscreen explicitly indicates that its `RXD`/`TXD` header expects 5V TTL.

## Runtime Config

After the FT232 is plugged into the Pi, it usually appears as `/dev/ttyUSB0`.
Legacy LSC controller-board diagnostics use:

```json
"servo_controller": {
  "type": "hiwonder",
  "port": "/dev/ttyUSB0",
  "baudrate": 9600
}
```

Do not use this config for `wildrobot-run-policy`.

## Test

From the Pi:

```bash
cd ~/wildrobot/runtime
uv run python scripts/test_hiwonder_board.py --port /dev/ttyUSB0 --baudrate 9600 --servo-ids 1,2,3
```

If this fails:

- Confirm `TXD` and `RXD` are crossed, not straight-through.
- Confirm FT232 `GND` and Hiwonder `GND` are connected.
- Confirm the Hiwonder controller and servos have external power.
- Confirm the FT232 device path with `ls /dev/ttyUSB*`.
