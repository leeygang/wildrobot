#!/usr/bin/env python3
"""
Deploy trained policy to WildRobot hardware.

This script integrates the trained policy with the runtime package for
hardware deployment on Raspberry Pi 5.
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import runtime package
try:
    from runtime.wildrobot_runtime import control, hardware, inference
except ImportError:
    print("ERROR: runtime package not found. Make sure to install it:")
    print("  cd runtime && pip install -e .")
    sys.exit(1)


class PolicyRunner:
    """Runs a trained policy on hardware."""

    def __init__(self, policy_path: str, config_path: str, use_onnx: bool = True):
        """
        Initialize the policy runner.

        Args:
            policy_path: Path to policy file (.onnx or .pkl)
            config_path: Path to hardware config JSON
            use_onnx: Use ONNX runtime (faster) vs JAX (more accurate)
        """
        print(f"Initializing PolicyRunner...")
        print(f"  Policy: {policy_path}")
        print(f"  Config: {config_path}")
        print(f"  Runtime: {'ONNX' if use_onnx else 'JAX'}")

        # Load policy
        if use_onnx:
            self.policy = inference.ONNXPolicy(policy_path)
        else:
            self.policy = inference.JAXPolicy(policy_path)

        # Initialize hardware
        self.config = hardware.load_config(config_path)
        self.motor_controller = hardware.MotorController(self.config)
        self.imu = hardware.IMU(self.config)

        # Control loop
        self.control_loop = control.ControlLoop(
            motor_controller=self.motor_controller,
            imu=self.imu,
            policy=self.policy,
            control_freq=50  # 50Hz to match training
        )

        # Observation buffer
        self.obs_buffer = np.zeros(57)  # Match training observation size
        self.prev_action = np.zeros(11)  # 11 DOFs

        print("PolicyRunner initialized successfully!")

    def build_observation(self) -> np.ndarray:
        """
        Build observation vector matching the training format.

        Returns:
            Observation array [57]: qpos(11) + qvel(11) + gravity(3) +
                                   angvel(3) + gyro(9) + accel(9) + prev_action(11)
        """
        # Get joint positions and velocities
        qpos = self.motor_controller.get_positions()  # [11]
        qvel = self.motor_controller.get_velocities()  # [11]

        # Get IMU data
        imu_data = self.imu.read()
        gravity = imu_data['gravity']  # [3] - normalized up vector
        angvel = imu_data['gyro']  # [3] - angular velocity

        # For multiple IMUs, concatenate readings
        # chest_imu, left_knee_imu, right_knee_imu
        gyro = np.concatenate([
            imu_data['chest_gyro'],
            imu_data['left_knee_gyro'],
            imu_data['right_knee_gyro'],
        ])  # [9]

        accel = np.concatenate([
            imu_data['chest_accel'],
            imu_data['left_knee_accel'],
            imu_data['right_knee_accel'],
        ])  # [9]

        # Concatenate all observations
        obs = np.concatenate([
            qpos,               # 11
            qvel,               # 11
            gravity,            # 3
            angvel,             # 3
            gyro,               # 9
            accel,              # 9
            self.prev_action,   # 11
        ])

        return obs

    def run(self, duration: float = 10.0):
        """
        Run policy on hardware for specified duration.

        Args:
            duration: Time to run in seconds
        """
        print(f"\nRunning policy for {duration}s...")
        print("Press Ctrl+C to stop")

        try:
            # Enable motors
            self.motor_controller.enable()
            time.sleep(0.5)

            start_time = time.time()
            step_count = 0

            while (time.time() - start_time) < duration:
                step_start = time.time()

                # Build observation
                obs = self.build_observation()

                # Get action from policy
                action = self.policy.predict(obs)

                # Clip actions to safe ranges
                action = np.clip(action, -1.0, 1.0)

                # Send commands to motors
                self.motor_controller.set_targets(action)

                # Store for next step
                self.prev_action = action

                step_count += 1

                # Maintain 50Hz control loop
                elapsed = time.time() - step_start
                if elapsed < 0.02:  # 50Hz = 20ms
                    time.sleep(0.02 - elapsed)

                if step_count % 50 == 0:  # Print every 1 second
                    print(f"Step {step_count}, time: {time.time() - start_time:.1f}s")

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            # Safely disable motors
            print("Disabling motors...")
            self.motor_controller.disable()
            print("Done!")

    def test_motors(self):
        """Test motor communication without running policy."""
        print("\nTesting motor communication...")
        self.motor_controller.enable()

        try:
            # Move to default standing pose
            default_pos = np.array([
                -0.4, 0.0, 0.8, -0.4, 0.0,  # Right leg
                -0.4, 0.0, 0.8, -0.4, 0.0,  # Left leg
                0.0  # Waist
            ])

            print("Moving to standing pose...")
            self.motor_controller.set_targets(default_pos)
            time.sleep(2.0)

            print("Reading positions:")
            positions = self.motor_controller.get_positions()
            for i, pos in enumerate(positions):
                print(f"  Joint {i}: {pos:.3f} rad")

        finally:
            self.motor_controller.disable()


def main():
    parser = argparse.ArgumentParser(description="Deploy policy to hardware")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy file")
    parser.add_argument("--config", type=str, default="runtime/configs/wildrobot_pi5.json",
                       help="Path to hardware config")
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration (seconds)")
    parser.add_argument("--test-motors", action="store_true", help="Test motors only")
    parser.add_argument("--use-jax", action="store_true", help="Use JAX instead of ONNX")

    args = parser.parse_args()

    runner = PolicyRunner(
        policy_path=args.policy,
        config_path=args.config,
        use_onnx=not args.use_jax
    )

    if args.test_motors:
        runner.test_motors()
    else:
        runner.run(duration=args.duration)


if __name__ == "__main__":
    main()
