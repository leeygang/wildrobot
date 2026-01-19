# BNO08X (GY-BNO08X) on Raspberry Pi 4B: Install, Wire, Test, Troubleshoot

This runbook targets the purple `GY-BNO08X` breakout (pins labeled `VCC_3V3`, `GND`, `SCL/SCK/RX`, `SDA/MISO/TX`, `ADDR/MOSI`, `CS`, `INT`, `RST`, `PS0`, `PS1`) wired to a Raspberry Pi 4B over **I2C**.

The WildRobot runtime IMU driver is `runtime/wr_runtime/hardware/bno085.py` (it uses the Adafruit `adafruit_bno08x` library over I2C).

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

If `i2cdetect` shows **no device**, the board may be strapped to a non-I2C mode; see Troubleshooting.

## 2) Enable I2C on Raspberry Pi (Ubuntu / Raspberry Pi OS)

### Check that the I2C device exists

```bash
ls -la /dev/i2c-*
```

If you don’t see `/dev/i2c-1`, enable I2C:

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

### Install I2C tools

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

## 5) Smoke test via the runtime IMU driver

From repo root on the Pi:

```bash
python3 - <<'PY'
import time
import numpy as np
from runtime.wr_runtime.hardware.bno085 import BNO085IMU

imu = BNO085IMU(i2c_address=0x4A, upside_down=False, sampling_hz=50)
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

If you don’t get updates (values stuck), check `i2cdetect`, wiring, and power.

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
  - `axis_map` handles permutation + sign (e.g., `["+X", "-Y", "+Z"]`) and can be calibrated interactively via
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

### Quaternion looks wrong (norm far from 1, or gravity seems inverted)

- First verify stable power and reliable reads.
- Then verify mounting orientation; set `upside_down` if mounted inverted.
- If still wrong, implement an explicit axis remap (recommended over stacking ad-hoc sign flips).
