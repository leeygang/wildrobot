from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np

from policy_contract.io import RobotIO
from policy_contract.numpy.frames import normalize_quat_xyzw
from policy_contract.numpy.signals import Signals

from .actuators import Actuators
from .foot_switches import FootSwitches
from .imu import Imu


@dataclass
class HardwareRobotIO(RobotIO[Signals]):
    actuator_names: List[str]
    control_dt: float
    actuators: Actuators
    imu: Imu
    foot_switches: FootSwitches
    max_cached_imu_age_s: float = 0.25

    _imu_nonfresh_consecutive: int = 0
    _last_fresh_imu_sample: Optional[object] = None
    _last_fresh_imu_wall_time_s: Optional[float] = None
    _last_imu_warn_time_s: float = 0.0

    def _imu_sample_is_startup_ready(self, imu_sample: object) -> bool:
        if not bool(getattr(imu_sample, "valid", True)) or not bool(getattr(imu_sample, "fresh", True)):
            return False
        diag = getattr(self.imu, "diag", None)
        if not isinstance(diag, dict):
            return True
        quat_status = str(diag.get("quat_status", ""))
        gyro_status = str(diag.get("gyro_status", ""))
        if quat_status.startswith("integrated_from_gyro"):
            return False
        if quat_status in {"missing", "bad_norm", "bad_axis_map_norm"}:
            return False
        if gyro_status in {"missing", "bad"}:
            return False
        return True

    def wait_for_valid_imu_sample(
        self,
        *,
        timeout_s: float = 3.0,
        poll_s: float = 0.02,
        required_consecutive: int = 6,
    ) -> None:
        """Wait until the IMU has produced at least one valid sample.

        BNO08X reports can be unavailable for a short period after enabling
        features. Entering the control loop before a valid quaternion/gyro sample
        exists is unsafe, so hardware startup waits here and fails with IMU
        diagnostics if the sensor never becomes valid.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        consecutive_ready = 0
        required_consecutive = max(1, int(required_consecutive))
        while time.monotonic() <= deadline:
            imu_sample = self.imu.read()
            if self._imu_sample_is_startup_ready(imu_sample):
                consecutive_ready += 1
            else:
                consecutive_ready = 0
            if consecutive_ready >= required_consecutive:
                self._imu_nonfresh_consecutive = 0
                self._last_fresh_imu_sample = imu_sample
                self._last_fresh_imu_wall_time_s = time.monotonic()
                return
            time.sleep(max(0.0, float(poll_s)))

        extra = ""
        if hasattr(self.imu, "error_count") or hasattr(self.imu, "last_error") or hasattr(self.imu, "diag"):
            parts = []
            for attr in ("error_count", "last_error", "diag"):
                if hasattr(self.imu, attr):
                    try:
                        parts.append(f"{attr}={getattr(self.imu, attr)}")
                    except Exception:
                        pass
            if parts:
                extra = " (" + ", ".join(parts) + ")"
        raise RuntimeError(
            "IMU did not produce enough direct fresh valid samples within "
            f"{float(timeout_s):.2f}s; required_consecutive={required_consecutive}; "
            f"abort for safety{extra}"
        )

    def read(self) -> Signals:
        imu_sample = self.imu.read()
        sample_valid = bool(getattr(imu_sample, "valid", True))
        sample_fresh = bool(getattr(imu_sample, "fresh", True))
        if not sample_valid or not sample_fresh:
            self._imu_nonfresh_consecutive += 1
            if self._last_fresh_imu_sample is None:
                raise RuntimeError("IMU reported no fresh valid sample before startup; aborting for safety")
            now = time.monotonic()
            last_fresh_s = self._last_fresh_imu_wall_time_s
            cached_age_s = float("inf") if last_fresh_s is None else now - last_fresh_s
            diag = getattr(self.imu, "diag", None)
            if cached_age_s > float(self.max_cached_imu_age_s):
                extra = ""
                if hasattr(self.imu, "error_count") and hasattr(self.imu, "last_error"):
                    try:
                        extra = f" (imu_error_count={getattr(self.imu, 'error_count')}, last_error={getattr(self.imu, 'last_error')})"
                    except Exception:
                        extra = ""
                raise RuntimeError(
                    "IMU cached sample is too old; aborting for safety. "
                    f"cached_age_s={cached_age_s:.3f} max={self.max_cached_imu_age_s:.3f} "
                    f"nonfresh_consecutive={self._imu_nonfresh_consecutive} "
                    f"sample_valid={sample_valid} sample_fresh={sample_fresh} diag={diag}{extra}"
                )

            if now - self._last_imu_warn_time_s > 1.0:
                self._last_imu_warn_time_s = now
                print(
                    "Warning: IMU sample not fresh; reusing last fresh sample "
                    f"(nonfresh_consecutive={self._imu_nonfresh_consecutive}, "
                    f"cached_age_s={cached_age_s:.3f}/{self.max_cached_imu_age_s:.3f}, "
                    f"sample_valid={sample_valid}, sample_fresh={sample_fresh}, diag={diag}).",
                    flush=True,
                )
            imu_sample = self._last_fresh_imu_sample
        else:
            self._imu_nonfresh_consecutive = 0
            self._last_fresh_imu_sample = imu_sample
            self._last_fresh_imu_wall_time_s = time.monotonic()

        joint_pos = self.actuators.get_positions_rad()
        if joint_pos is None:
            port = getattr(self.actuators, "port", "unknown")
            baudrate = getattr(self.actuators, "baudrate", "unknown")
            last_error = getattr(self.actuators, "_last_error", None)
            err_msg = ""
            if last_error is not None:
                err_msg = f"; actuator_error={repr(last_error)}"
            raise RuntimeError(
                f"Failed to read joint positions for {self.actuator_names} on port {port} baud {baudrate}{err_msg}"
            )
        joint_vel = self.actuators.estimate_velocities_rad_s(self.control_dt)
        foot_sample = self.foot_switches.read()
        foot = np.array(foot_sample.switches, dtype=np.float32)
        quat_xyzw = normalize_quat_xyzw(np.asarray(imu_sample.quat_xyzw, dtype=np.float32))

        return Signals(
            quat_xyzw=quat_xyzw,
            gyro_rad_s=np.asarray(imu_sample.gyro_rad_s, dtype=np.float32),
            joint_pos_rad=np.asarray(joint_pos, dtype=np.float32),
            joint_vel_rad_s=np.asarray(joint_vel, dtype=np.float32),
            foot_switches=foot,
            timestamp_s=float(getattr(imu_sample, "timestamp_s", 0.0) or 0.0),
        )

    def write_ctrl(self, ctrl_targets_rad) -> None:
        self.actuators.set_targets_rad(np.asarray(ctrl_targets_rad, dtype=np.float32), move_time_ms=None)

    def close(self) -> None:
        try:
            self.foot_switches.close()
        finally:
            try:
                self.imu.close()
            finally:
                self.actuators.disable()
                self.actuators.close()
