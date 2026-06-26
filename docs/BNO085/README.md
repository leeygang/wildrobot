# IMU Documents In This Folder

This folder is named `BNO085`, but the checked-in files here are for the
TDK/InvenSense `ICM-20948`:

- `DS-000189-ICM-20948-v1.3.pdf`
- `GY-20948V2-SCH.png`
- `SparkFun_9DoF_IMU_Breakout_-_ICM_20948_-_Arduino_Library.rar`

Do not use these files as the BNO085/BNO08x protocol reference. WildRobot's
current runtime driver, `runtime/wr_runtime/hardware/bno085.py`, targets the
BNO08x family through Adafruit's `adafruit_bno08x` SHTP driver.

Quick hardware check:

- BNO08x/BNO085 I2C addresses are commonly `0x4A` or `0x4B`.
- ICM-20948 I2C addresses are `0x68` or `0x69`; `WHO_AM_I` is `0xEA`.

If `i2cdetect -y 1` shows `0x68`/`0x69`, the robot has an ICM-20948-class IMU
and the BNO085 runtime driver is the wrong driver. If it shows `0x4A`/`0x4B`,
use the BNO08x/BNO085 driver and BNO08x protocol docs instead.
