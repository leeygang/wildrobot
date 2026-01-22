# Foot Switch Integration (Omron SS-10GL) on Raspberry Pi 4

This runbook describes how to wire and use 4x mechanical foot/contact switches (e.g. Omron `SS-10GL`) with the WildRobot runtime on a Raspberry Pi 4.

WildRobot runtime integration points:
- GPIO driver: `runtime/wr_runtime/hardware/foot_switches.py`
- Runtime config field: `foot_switches` (see `runtime/README.md` and `runtime/configs/wr_runtime_config.json`)

## 1) Electrical behavior (what the runtime expects)

Each switch is read as a **GPIO input with an internal pull-up**:
- **Not pressed / open circuit** → input pulled up → reads `HIGH` → runtime reports `False`
- **Pressed / closed to ground** → input shorted to GND → reads `LOW` → runtime reports `True`

The runtime uses Blinka `digitalio.Pull.UP` and inverts the read value:
- `FootSwitchSample.switches` is ordered as: `(left_toe, left_heel, right_toe, right_heel)`
- `True` means **pressed/closed**

For this to work correctly, wire the switch as **normally-open to ground** (NO → GPIO, COM → GND).

## 2) Wiring (per switch)

An Omron `SS-10GL` is a simple 3-terminal microswitch:
- `COM` (common)
- `NO` (normally open)
- `NC` (normally closed)

Recommended wiring (NO behavior):
- `COM` → Raspberry Pi **GND**
- `NO` → Raspberry Pi **GPIO** input pin
- Leave `NC` unconnected

Notes:
- Use **3.3V GPIO only**. Do not connect external voltage (5V/12V) to GPIO pins.
- All 4 switches can share the same Pi ground.
- If your wiring is long/noisy, consider: twisted pair to GND, a small series resistor near the Pi (e.g. `220–1kΩ`), and optional RC filtering (e.g. `0.01–0.1µF` from GPIO→GND at the Pi end).

## 3) Pin mapping (runtime config names → Raspberry Pi pins)

The runtime config uses Adafruit Blinka `board` pin names like `"D5"`. On Raspberry Pi, Blinka `D<n>` corresponds to **BCM GPIO `<n>`**.

Default pins from `runtime/configs/wr_runtime_config.json`:

| Runtime name | Blinka pin | BCM GPIO | Physical header pin |
|---|---:|---:|---:|
| `left_toe` | `D5` | GPIO5 | pin 29 |
| `left_heel` | `D6` | GPIO6 | pin 31 |
| `right_toe` | `D13` | GPIO13 | pin 33 |
| `right_heel` | `D19` | GPIO19 | pin 35 |

Any convenient ground works, for example:
- **GND** physical pin 30, 34, or 39 (or others)

## 4) Configure the runtime

In your runtime JSON (default: `~/wildrobot_config.json`), set the `foot_switches` section:

```json
"foot_switches": {
  "left_toe": "D5",
  "left_heel": "D6",
  "right_toe": "D13",
  "right_heel": "D19"
}
```

If you change wiring, update these values to match (Blinka pin names).

## 5) Install deps on the Pi (Blinka GPIO)

The `FootSwitches` driver imports:
- `board`
- `digitalio`

These come from Adafruit Blinka. Typical install paths:
- If you install the runtime package: `cd runtime && pip install -e .`
- If you still get `ModuleNotFoundError: board`, install Blinka: `pip install adafruit-blinka`

## 6) Smoke test (read the switches)

Run this on the Pi to verify each switch toggles correctly:

```bash
python3 - <<'PY'
import time
from runtime.wr_runtime.hardware.foot_switches import FootSwitches

foot = FootSwitches(
    {
        "left_toe": "D5",
        "left_heel": "D6",
        "right_toe": "D13",
        "right_heel": "D19",
    }
)

try:
    print("Order: left_toe, left_heel, right_toe, right_heel")
    while True:
        s = foot.read()
        print(s.switches)
        time.sleep(0.05)
finally:
    foot.close()
PY
```

Expected:
- With no switches pressed: `[False, False, False, False]`
- Pressing each switch toggles its corresponding entry to `True`

Alternative (interactive):

```bash
uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --calibrate-footswitch
```

This will:
- list the selectable targets (`left_toe`, `left_heel`, `right_toe`, `right_heel`, `left_foot`, `right_foot`)
- print their live pressed/open status until you quit

## 7) Runtime behavior (where the signals go)

When running `wildrobot-run-policy`, foot switches are read every control tick and included in the `Signals` struct:
- Hardware read path: `runtime/wr_runtime/hardware/robot_io.py`
- Logging: if you pass `--log-path`, `foot_switches` is saved as an array shaped `(T, 4)`

## Troubleshooting

- Reads always `True` (pressed): you likely wired the GPIO to **3.3V** or used `NC` by mistake; use `COM→GND` and `NO→GPIO`.
- Reads always `False`: wrong pin name in config, broken wire, or the switch never closes; check continuity (COM↔NO closes when pressed).
- Flickers/chatter when pressed: contact bounce or noise; shorten wiring, add RC filtering, or implement software debouncing upstream of whatever consumes `foot_switches`.

