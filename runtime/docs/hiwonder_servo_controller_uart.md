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

This legacy LSC controller-board path is not part of the current runtime config.
Do not use it for `wildrobot-run-policy`.

## Test

Use the raw TTL runtime tools instead of the legacy LSC board diagnostic. For
example, `scripts/calibrate.py --go-home` uses the same TTL servo path as policy
runtime.

If this fails:

- Confirm `TXD` and `RXD` are crossed, not straight-through.
- Confirm FT232 `GND` and Hiwonder `GND` are connected.
- Confirm the Hiwonder controller and servos have external power.
- Confirm the FT232 device path with `ls /dev/ttyUSB*`.
