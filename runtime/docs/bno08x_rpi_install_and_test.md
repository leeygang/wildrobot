# BNO08X (GY-BNO08X) on Raspberry Pi 4B: Install, Wire, Test, Troubleshoot

This runbook targets the purple `GY-BNO08X` breakout (pins labeled `VCC_3V3`, `GND`, `SCL/SCK/RX`, `SDA/MISO/TX`, `ADDR/MOSI`, `CS`, `INT`, `RST`, `PS0`, `PS1`) wired to a Raspberry Pi 4B over **I2C** or **SPI**.

The WildRobot runtime IMU driver is `runtime/wr_runtime/hardware/bno085.py` (using the Adafruit `adafruit_bno08x` library over I2C or SPI). SPI is the recommended WildRobot runtime wiring after the I2C freshness issues seen during walking tests.

## 1) Wiring (I2C mode)

**Use 3.3V only.** Do not power this IMU from 5V unless your specific board explicitly supports it.

Wire:
- `VCC_3V3` → Pi **3V3** (physical pin 1 or 17)
- `GND` → Pi **GND** (e.g., physical pin 6)
- `SCL/SCK/RX` → Pi **SCL1** (GPIO3, physical pin 5)
- `SDA/MISO/TX` → Pi **SDA1** (GPIO2, physical pin 3)

Optional:
- `INT` → optional (not required by the current runtime driver)
- `RST` → optional (not required by the current runtime driver)

Leave these unconnected for I2C:
- `CS` (SPI chip select)
- SPI pins if you’re not using SPI (`SCK/MOSI/MISO`)

### I2C address selection

The `ADDR/MOSI` pin typically selects the I2C address:
- common addresses: `0x4A` or `0x4B`

You’ll confirm the address with `i2cdetect` (below).

### Interface mode pins (PS0/PS1)

`PS0`/`PS1` select the sensor’s interface at boot (I2C vs UART vs SPI). Many GY-BNO08X boards are pre-strapped for I2C.

For the Adafruit BNO085 breakout pinout, which matches the BNO08x protocol-select behavior:

| `PS1` | `PS0` | Mode |
| --- | --- | --- |
| Low | Low | I2C |
| Low | High | UART-RVC |
| High | Low | UART |
| High | High | SPI |

If `i2cdetect` shows **no device**, the board may be strapped to a non-I2C mode; see Troubleshooting.

### SPI wiring

**Use 3.3V only.** Power down the Pi and IMU before changing the `PS0`/`PS1` straps; those mode-select pins are sampled at sensor boot.

Wire:

| Your Board Label | SPI Alternate Name | Raspberry Pi 4 Pin | Pin Function |
| --- | --- | --- | --- |
| `VCC` / `3V3` / `VIN` | Power | Pin 1 or 17 (3.3V) | 3.3V Power |
| `GND` | Ground | Pin 6, 9, or 14 (GND) | Ground |
| `SCL/SCK/RX` | `SCK` | Pin 23 (GPIO 11) | SPI Clock |
| `SDA/MISO/TX` | `MISO` | Pin 21 (GPIO 9) | Master In Slave Out |
| `ADDR/MOSI` | `MOSI` | Pin 19 (GPIO 10) | Master Out Slave In |
| `CS` | `CS` | Pin 24 (GPIO 8) | Chip Select (CE0) |
| `INT` | `INT` / `IRQ` | Pin 11 (GPIO 17) | Host Interrupt |
| `RST` | Reset | Pin 13 (GPIO 27) | Hardware Reset |
| `PS0` | Protocol Select 0 | Connect to 3.3V | SPI mode select |
| `PS1` | Protocol Select 1 | Connect to 3.3V | SPI mode select |

Board silkscreens vary. Follow the SPI alternate function printed on your board, not just the short I2C label, and confirm against the board schematic before applying power. A swapped `MISO`/`MOSI` wire commonly produces corrupt SPI packet headers during BNO08X reset.

## 2) Enable the Raspberry Pi bus (Ubuntu / Raspberry Pi OS)

### SPI mode

Check that the SPI device exists:

```bash
ls -la /dev/spidev*
```

Expected for the wiring above: `/dev/spidev0.0`.

If it is missing, enable SPI:

```bash
sudo raspi-config
```

Enable: **Interface Options → SPI** then reboot.

On images without `raspi-config`, edit `/boot/firmware/config.txt` (Ubuntu) or `/boot/config.txt` (Pi OS) and ensure:

```ini
dtparam=spi=on
```

Reboot:

```bash
sudo reboot
```

### I2C mode

Check that the I2C device exists:

```bash
ls -la /dev/i2c-*
```

If you do not see `/dev/i2c-1`, enable I2C:

### Option A: `raspi-config` (if available)

```bash
sudo raspi-config
```

Enable: **Interface Options → I2C** then reboot.

### Option B: edit boot config (common on Ubuntu for Pi)

Edit `/boot/firmware/config.txt` (Ubuntu) or `/boot/config.txt` (Pi OS) and ensure:

```ini
dtparam=i2c_arm=on
```

Reboot:

```bash
sudo reboot
```

## 3) Install tools + Python deps

### Optional I2C tools

```bash
sudo apt-get update
sudo apt-get install -y i2c-tools
```

### Runtime Python deps

Install the runtime package on the robot (from repo root):

```bash
cd runtime
pip install -e .
```

The IMU driver imports these at runtime on the Pi:
- `board`
- `busio`
- `adafruit_bno08x`

If those aren’t present, install them (exact method depends on your robot image; common is):

```bash
pip install adafruit-circuitpython-bno08x adafruit-blinka
```

## 4) Detect the sensor on I2C

Run:

```bash
sudo i2cdetect -y 1
```

Expected: you should see a device at `0x4a` or `0x4b`.

Set your runtime JSON accordingly:
- `bno085.i2c_address: "0x4A"` (or `"0x4B"`)

For SPI runtime, set your runtime JSON accordingly:
- `bno085.transport: "spi"`
- `bno085.spi_baudrate: 50000`
- `bno085.spi_read_skip_bytes: 2`
- `bno085.spi_cs_pin: "D8"`
- `bno085.spi_int_pin: "D17"`
- `bno085.spi_reset_pin: "D27"`

The 50 kHz / 2-byte read-skip profile matches the WR Raspberry Pi 4 + BNO08x
breakout measured with `scripts/probe_bno085_spi_raw.py`: SPI mode 3 returns a
valid startup SHTP packet after two leading zero bytes. Adafruit's BNO08x SPI
driver and SparkFun's BNO080 library both use SPI mode 3; this setting only
accounts for the observed leading-byte preamble on this board path.

## 5) Smoke test via the runtime IMU driver

From repo root on the Pi:

```bash
python3 - <<'PY'
import time
import numpy as np
from runtime.wr_runtime.hardware.bno085 import BNO085IMU

imu = BNO085IMU(
    transport="spi",
    spi_baudrate=50000,
    spi_read_skip_bytes=2,
    spi_cs_pin="D8",
    spi_int_pin="D17",
    spi_reset_pin="D27",
    upside_down=False,
    sampling_hz=50,
)
try:
    time.sleep(0.2)
    for i in range(20):
        s = imu.read()
        q = np.asarray(s.quat_xyzw, dtype=np.float32)
        w = float(np.linalg.norm(q))
        g = np.asarray(s.gyro_rad_s, dtype=np.float32)
        print(f\"{i:02d} | quat={q} | |quat|={w:.4f} | gyro(rad/s)={g}\")
        time.sleep(0.05)
finally:
    imu.close()
PY
```

Checks:
- `|quat|` should be ~`1.0` (close to 1, not 0 or NaN).
- When stationary, `gyro(rad/s)` should be near `[0, 0, 0]` (small noise is normal).

If you do not get updates (values stuck), check wiring, power, `PS0`/`PS1`, `/dev/spidev0.0`, and the configured transport.

## 6) Orientation sanity checks (mounting / axis mapping)

The breakout prints an axis diagram. You must ensure the runtime’s IMU frame matches the frame assumptions used in training/runtime:
- `+X` forward
- `+Y` left
- `+Z` up

Practical checks:
- Rotate the robot/yaw to the right around vertical axis: one gyro axis should spike consistently (yaw rate).
- Pitch forward: a different gyro axis should spike.
- Roll right/left: a third axis should spike.

If axes are swapped/negated due to mounting, update the remap in:
- `runtime/wr_runtime/hardware/bno085.py`:
  - `upside_down` handles a simple inverted mount.
  - `axis_map` handles right-handed permutation + sign (e.g., `["+X", "-Y", "-Z"]`) and can be calibrated interactively via
    `runtime/scripts/calibrate.py --calibrate-imu`.

## 7) Troubleshooting

### `i2cdetect` shows nothing

- Confirm 3.3V power and common GND.
- Confirm Pi I2C enabled (`/dev/i2c-1` exists).
- Check wiring: SDA/SCL swapped is common.
- Keep wires short; ensure pull-ups are present (most breakouts include them).
- The board may be strapped to UART/SPI (PS0/PS1). If so, reconfigure straps/jumpers for I2C mode per your board’s schematic.

### `i2cdetect` shows a device, but the Python driver fails

- Confirm the I2C address (`0x4A` vs `0x4B`).
- Ensure `adafruit-blinka` and `adafruit-circuitpython-bno08x` are installed.
- Confirm you’re running on a Pi (the driver imports `board`/`busio`).
- Try lowering bus speed if you suspect signal integrity issues (start with 100kHz).

### SPI debug prints `SHTP READ packet header: ['0xff', '0xff', '0xff', '0xff']`

This means the Pi is reading an all-high/floating MISO line instead of a valid BNO08x SHTP header.

- Confirm `PS1=High` and `PS0=High` before powering the IMU; low/low is I2C mode.
- Confirm `CS` is wired to the configured chip-select pin (`D8` / CE0 / physical pin 24 by default).
- Confirm `SDA/MISO/TX` is wired to Pi MISO (physical pin 21), and `ADDR/MOSI` is wired to Pi MOSI (physical pin 19).
- Confirm `INT` and `RST` are connected; the Adafruit SPI driver requires both for stable SPI operation.
- Confirm common ground and 3.3V power.

### Quaternion looks wrong (norm far from 1, or gravity seems inverted)

- First verify stable power and reliable reads.
- Then verify mounting orientation; set `upside_down` if mounted inverted.
- If still wrong, implement an explicit axis remap (recommended over stacking ad-hoc sign flips).
