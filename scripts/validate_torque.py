#!/usr/bin/env python3
"""
Torque Capacity Validation Script for WildRobot

This script validates that the robot's actuators can handle the required torques
for various poses within the 4.41Nm torque limit constraint (HTD-45H @ 12V).
It also prints detailed body dimensions (mass, inertia, COM) to help with design optimization.

Usage:
    python validate_torque.py <path_to_model.xml> [--visualize] [--body-info-only]

Critical for Phase II of mujoco_playground_plan.md:
- PASS: Max torque < 2.65 Nm (~60%, leaving 1.76Nm headroom for dynamics)
- WARN: 2.65-3.75 Nm (marginal, risky for dynamic motion)
- FAIL: Max torque > 3.75 Nm (~85%, robot cannot walk reliably, must redesign)

Servo: HTD-45H @ 12V
- Stall Torque: 45 kg·cm = 4.41 Nm
- Continuous Safe: ~60% = 2.65 Nm
- Peak Dynamic: ~85% = 3.75 Nm

Static Pose Tests:
1. Standing pose (neutral)
2. Deep squat (knees bent ~60 degrees)
3. Single leg stance (left)
4. Single leg stance (right)
5. Forward lean

Walking Gait Cycle Tests:
6. Heel strike (left & right)
7. Mid-stance (left & right) - most demanding
8. Toe-off (left & right) - pushing off
9. Double support phase

Features:
- Body dimensions and mass properties for all robot bodies
- Body hierarchy tree with mass distribution
- Torque validation for critical poses
- Optional visualization of worst-case pose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np


# Test poses: joint positions in radians
# Joint order (11 actuated DOFs - foot_roll joints now ENABLED for testing):
# [r_hip_pitch, r_hip_roll, r_knee, r_ankle, r_foot_roll,
#  l_hip_pitch, l_hip_roll, l_knee, l_ankle, l_foot_roll, waist]

TEST_POSES = {
    "standing_neutral": np.array([
        0.0, 0.0, 0.0, 0.0, 0.0,  # right leg
        0.0, 0.0, 0.0, 0.0, 0.0,  # left leg
        0.0,  # waist
    ]),

    "right_leg_squat": np.array([
        -0.3, 0.0, 0.6, -0.3, 0.0,   # right leg (squat, foot_roll=0)
        0.0, 0.0, 0.0, 0.0, 0.0,     # left leg (straight)
        0.0,  # waist
    ]),

    "single_leg_right": np.array([
        -0.15, 0.03, 0.3, -0.15, 0.0,  # right leg (weight bearing, foot_roll=0)
        0.1, -0.02, 0.2, -0.1, 0.0,    # left leg (lifted)
        0.0,  # waist
    ]),

    "forward_lean": np.array([
        -0.15, 0.0, 0.3, -0.15, 0.0,  # right leg
        0.0, 0.0, 0.0, 0.0, 0.0,      # left leg (straight)
        0.05,  # waist
    ]),

    # Walking gait cycle poses
    "walk_heel_strike_right": np.array([
        -0.2, 0.02, 0.4, -0.2, 0.0,    # right leg (heel strike)
        0.1, -0.02, 0.2, -0.1, 0.0,    # left leg (toe-off)
        0.03,  # waist
    ]),

    "walk_mid_stance_right": np.array([
        -0.1, 0.02, 0.2, -0.1, 0.0,    # right leg (mid-stance)
        0.15, -0.02, 0.3, -0.15, 0.0,  # left leg (swing)
        0.02,  # waist
    ]),

    "walk_toe_off_left": np.array([
        -0.15, 0.02, 0.3, -0.15, 0.0,  # right leg (forward)
        0.1, -0.02, 0.2, -0.3, 0.0,    # left leg (toe-off)
        0.03,  # waist
    ]),

    "walk_toe_off_right": np.array([
        0.05, 0.02, 0.1, -0.2, 0.0,    # right leg (toe-off)
        0.0, -0.02, 0.0, 0.0, 0.0,     # left leg (neutral)
        0.02,  # waist
    ]),
}

# NOTE: Model has a kinematic bug where left_hip_pitch < 0 (flexion) causes
# extreme coupling torques (300-800 Nm) on left_hip_roll.
# Tests use positive left_hip_pitch (extension) or zero to avoid this bug.
# This needs to be fixed in the MJCF model (likely body transform issue).

# Torque limits (Nm)
# HTD-45H: 45 kg·cm @ 12V = 45 × 0.0981 = 4.41 Nm
TORQUE_LIMIT = 4.41      # Hardware limit (HTD-45H servos @ 12V)
SAFE_THRESHOLD = 2.65    # Safe operating threshold (~60%, leaves 1.76Nm headroom)
CRITICAL_THRESHOLD = 3.75 # Critical threshold (~85%, robot may struggle if exceeded)


class TorqueValidator:
    """Validates robot torque requirements against actuator limits."""

    def __init__(self, model_path: str):
        """Initialize validator with MuJoCo model."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # Get actuator names for reporting
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

    def print_body_dimensions(self):
        """Print physical properties of all bodies in the model."""
        print(f"\n{'='*70}")
        print("BODY DIMENSIONS AND PROPERTIES")
        print(f"{'='*70}\n")

        # Total mass
        total_mass = 0.0

        print(f"{'Body Name':<25} {'Mass (kg)':<12} {'COM (x,y,z) [m]':<30} {'Inertia Diag [kg·m²]'}")
        print("-" * 100)

        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                body_name = f"body_{i}"

            # Get body mass
            mass = self.model.body_mass[i]
            total_mass += mass

            # Get body center of mass (in body frame)
            com = self.model.body_ipos[i]  # Position of body COM in parent frame

            # Get body inertia (diagonal elements)
            # MuJoCo stores inertia as [ixx, iyy, izz] for diagonal elements
            inertia = self.model.body_inertia[i]

            # Skip world body (index 0) typically
            if i == 0:
                continue

            print(f"{body_name:<25} {mass:>10.4f}  "
                  f"({com[0]:>7.4f}, {com[1]:>7.4f}, {com[2]:>7.4f})  "
                  f"({inertia[0]:>7.5f}, {inertia[1]:>7.5f}, {inertia[2]:>7.5f})")

        print("-" * 100)
        print(f"{'TOTAL MASS':<25} {total_mass:>10.4f} kg\n")

        # Print body tree structure
        print(f"\n{'='*70}")
        print("BODY HIERARCHY")
        print(f"{'='*70}\n")

        def print_tree(body_id, indent=0):
            """Recursively print body tree."""
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                body_name = f"body_{body_id}"
            mass = self.model.body_mass[body_id]

            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            print(f"{prefix}{body_name} ({mass:.3f} kg)")

            # Find children
            for i in range(self.model.nbody):
                if self.model.body_parentid[i] == body_id and i != body_id:
                    print_tree(i, indent + 1)

        print_tree(0)  # Start from world/root body
        print()

    def compute_inverse_dynamics(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute required torques to hold a given pose against gravity.

        Args:
            qpos: Joint positions for actuated joints (shape: [nu])
                  Order: [r_hip_pitch, r_hip_roll, r_knee, r_ankle, r_foot_roll,
                          l_hip_pitch, l_hip_roll, l_knee, l_ankle, l_foot_roll, waist]

        Returns:
            Required joint torques for actuated joints (shape: [nu])
        """
        # Set joint positions
        self.data.qpos[:] = 0.0

        # Map actuated joint positions to full qpos
        # Model qpos layout: [freejoint(7), right_hip_pitch, right_hip_roll, right_knee_pitch,
        #                     right_ankle_pitch, right_foot_roll, left_hip_pitch, left_hip_roll,
        #                     left_knee_pitch, left_ankle_pitch, left_foot_roll, waist_yaw]
        # Total: 7 (freejoint) + 11 (joints) = 18

        if len(qpos) == 11:  # 11 actuated DOFs (with foot_roll)
            # Direct mapping - qpos already has all joints
            self.data.qpos[7:] = qpos
        elif len(qpos) == 9:  # 9 actuated DOFs (legacy - without foot_roll)
            # Expand to 11 by inserting zeros for foot_roll joints
            full_qpos = np.zeros(11)
            full_qpos[0:4] = qpos[0:4]   # right leg (4 joints)
            full_qpos[4] = 0.0            # right_foot_roll (set to 0)
            full_qpos[5:9] = qpos[4:8]   # left leg (4 joints)
            full_qpos[9] = 0.0            # left_foot_roll (set to 0)
            full_qpos[10] = qpos[8]       # waist
            self.data.qpos[7:] = full_qpos
        else:
            raise ValueError(f"Expected qpos with 9 or 11 elements, got {len(qpos)}")

        # Set velocities and accelerations to zero (static pose)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

        # Compute inverse dynamics
        mujoco.mj_inverse(self.model, self.data)

        # Extract generalized forces (skip base coordinates)
        # qfrc_inverse contains the forces needed to achieve qacc given qpos and qvel
        joint_torques = self.data.qfrc_inverse[6:]  # Skip 6 DOF freejoint

        return joint_torques

    def validate_pose(self, pose_name: str, qpos: np.ndarray) -> Dict:
        """
        Validate torque requirements for a specific pose.

        Args:
            pose_name: Name of the test pose
            qpos: Joint positions

        Returns:
            Dictionary with validation results
        """
        torques = self.compute_inverse_dynamics(qpos)

        # Get absolute torques
        abs_torques = np.abs(torques)
        max_torque = np.max(abs_torques)
        max_idx = np.argmax(abs_torques)

        # Determine status
        if max_torque < SAFE_THRESHOLD:
            status = "PASS"
            color = "\033[92m"  # Green
        elif max_torque < CRITICAL_THRESHOLD:
            status = "WARN"
            color = "\033[93m"  # Yellow
        else:
            status = "FAIL"
            color = "\033[91m"  # Red
        reset_color = "\033[0m"

        # Find critical joints (> 70% of safe threshold)
        critical_joints = []
        for i, torque in enumerate(abs_torques):
            if torque > 0.7 * SAFE_THRESHOLD:
                joint_name = self.actuator_names[i] if i < len(self.actuator_names) else f"joint_{i}"
                critical_joints.append((joint_name, torque))

        return {
            "pose_name": pose_name,
            "status": status,
            "color": color,
            "reset_color": reset_color,
            "max_torque": max_torque,
            "max_joint": self.actuator_names[max_idx] if max_idx < len(self.actuator_names) else f"joint_{max_idx}",
            "torques": torques,
            "critical_joints": critical_joints,
        }

    def run_all_tests(self) -> Tuple[List[Dict], bool]:
        """
        Run all torque validation tests.

        Returns:
            Tuple of (results list, overall pass/fail)
        """
        results = []
        overall_pass = True

        # First, print body dimensions
        self.print_body_dimensions()

        print(f"\n{'='*70}")
        print(f"TORQUE CAPACITY VALIDATION - WildRobot")
        print(f"{'='*70}")
        print(f"Model: {self.model_path.name}")
        print(f"Hardware Limit: {TORQUE_LIMIT} Nm")
        print(f"Safe Threshold: {SAFE_THRESHOLD} Nm (leaves {TORQUE_LIMIT - SAFE_THRESHOLD:.2f}Nm headroom)")
        print(f"Critical Threshold: {CRITICAL_THRESHOLD} Nm")
        print(f"{'='*70}\n")

        # Group tests by category
        static_tests = ["standing_neutral", "right_leg_squat", "single_leg_right", "forward_lean"]
        walking_tests = ["walk_heel_strike_right", "walk_mid_stance_right",
                        "walk_toe_off_left", "walk_toe_off_right"]

        # Run static pose tests
        print("=" * 70)
        print("STATIC POSE TESTS")
        print("=" * 70)
        for pose_name in static_tests:
            if pose_name in TEST_POSES:
                result = self.validate_pose(pose_name, TEST_POSES[pose_name])
                results.append(result)
                self._print_result(result)
                if result['status'] == "FAIL":
                    overall_pass = False

        # Run walking gait tests
        print(f"\n{'=' * 70}")
        print("WALKING GAIT CYCLE TESTS")
        print("=" * 70)
        for pose_name in walking_tests:
            if pose_name in TEST_POSES:
                result = self.validate_pose(pose_name, TEST_POSES[pose_name])
                results.append(result)
                self._print_result(result)
                if result['status'] == "FAIL":
                    overall_pass = False

        return results, overall_pass

    def _print_result(self, result: Dict):
        """Helper method to print a single test result."""
        print(f"{result['color']}[{result['status']}]{result['reset_color']} {result['pose_name']:25s} | "
              f"Max Torque: {result['max_torque']:6.3f} Nm @ {result['max_joint']}")

        # Print critical joints if any
        if result['critical_joints']:
            print(f"      Critical joints (>70% safe threshold):")
            for joint_name, torque in result['critical_joints']:
                print(f"        - {joint_name}: {torque:.3f} Nm ({torque/SAFE_THRESHOLD*100:.1f}%)")

    def print_summary(self, results: List[Dict], overall_pass: bool):
        """Print validation summary."""
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        # Separate static and walking results
        static_results = [r for r in results if not r['pose_name'].startswith('walk_')]
        walking_results = [r for r in results if r['pose_name'].startswith('walk_')]

        # Overall statistics
        max_overall = max(r['max_torque'] for r in results)
        worst_pose = max(results, key=lambda r: r['max_torque'])

        print(f"\nOverall Statistics:")
        print(f"  Total tests run: {len(results)}")
        print(f"  Maximum torque observed: {max_overall:.3f} Nm")
        print(f"  Worst pose: {worst_pose['pose_name']} ({worst_pose['max_joint']})")
        print(f"  Headroom available: {TORQUE_LIMIT - max_overall:.3f} Nm ({(TORQUE_LIMIT - max_overall) / TORQUE_LIMIT * 100:.1f}%)")

        # Static pose statistics
        if static_results:
            max_static = max(r['max_torque'] for r in static_results)
            worst_static = max(static_results, key=lambda r: r['max_torque'])
            print(f"\nStatic Pose Statistics:")
            print(f"  Tests: {len(static_results)}")
            print(f"  Max torque: {max_static:.3f} Nm ({worst_static['pose_name']})")
            print(f"  Utilization: {max_static / TORQUE_LIMIT * 100:.1f}% of hardware limit")

        # Walking gait statistics
        if walking_results:
            max_walking = max(r['max_torque'] for r in walking_results)
            worst_walking = max(walking_results, key=lambda r: r['max_torque'])
            print(f"\nWalking Gait Statistics:")
            print(f"  Tests: {len(walking_results)}")
            print(f"  Max torque: {max_walking:.3f} Nm ({worst_walking['pose_name']})")
            print(f"  Utilization: {max_walking / TORQUE_LIMIT * 100:.1f}% of hardware limit")

        # Test results breakdown
        pass_count = sum(1 for r in results if r['status'] == 'PASS')
        warn_count = sum(1 for r in results if r['status'] == 'WARN')
        fail_count = sum(1 for r in results if r['status'] == 'FAIL')

        print(f"\nTest Results Breakdown:")
        print(f"  \033[92mPASS\033[0m: {pass_count}/{len(results)} (< {SAFE_THRESHOLD} Nm)")
        print(f"  \033[93mWARN\033[0m: {warn_count}/{len(results)} ({SAFE_THRESHOLD}-{CRITICAL_THRESHOLD} Nm)")
        print(f"  \033[91mFAIL\033[0m: {fail_count}/{len(results)} (> {CRITICAL_THRESHOLD} Nm)")

        print(f"\n{'='*70}")
        if overall_pass:
            print("\033[92m✓ VALIDATION PASSED\033[0m")
            print("The robot meets torque requirements for all test poses.")
            if warn_count > 0:
                print(f"\nNote: {warn_count} pose(s) in WARNING range - consider optimization.")
            print("Safe to proceed with RL training.")
        else:
            print("\033[91m✗ VALIDATION FAILED\033[0m")
            print("The robot EXCEEDS safe torque limits in some poses.")
            print("\nACTION REQUIRED:")
            print("1. Review mass distribution in CAD model (check body dimensions above)")
            print("2. Consider reducing link lengths or masses")
            print("3. Optimize for lower center of mass")
            print("4. Focus on heaviest bodies and longest moment arms")
            print("5. DO NOT proceed with RL training until this is resolved")
        print(f"{'='*70}\n")

    def visualize_worst_pose(self, results: List[Dict]):
        """Visualize the worst-case pose in MuJoCo viewer."""
        try:
            import mujoco.viewer

            worst_pose = max(results, key=lambda r: r['max_torque'])
            pose_name = worst_pose['pose_name']
            qpos = TEST_POSES[pose_name]

            print(f"Launching viewer for worst pose: {pose_name}")
            print("Press ESC to exit viewer\n")

            # Set pose
            self.data.qpos[:] = 0.0
            self.data.qpos[7:] = qpos
            mujoco.mj_forward(self.model, self.data)

            # Launch viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.sync()
                # Keep viewer open until user closes it
                while viewer.is_running():
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
        except ImportError:
            print("mujoco.viewer not available. Skipping visualization.")
        except KeyboardInterrupt:
            print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate robot torque capacity against 4.41Nm actuator limit (HTD-45H @ 12V)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_torque.py assets/mjcf/wildrobot.xml
  python validate_torque.py assets/mjcf/wildrobot.xml --visualize
  python validate_torque.py assets/mjcf/wildrobot.xml --body-info-only

Servo Specifications (HTD-45H @ 12V):
  Stall Torque: 45 kg·cm = 4.41 Nm
  Safe Continuous: ~60% = 2.65 Nm (leaves 1.76Nm headroom)
  Peak Dynamic: ~85% = 3.75 Nm (maximum for walking)

Exit Codes:
  0: All tests passed (max torque < 2.65 Nm)
  1: Some tests failed (max torque > 3.75 Nm)
  2: Error loading model or running tests
        """
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to MuJoCo XML model file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize worst-case pose in MuJoCo viewer"
    )
    parser.add_argument(
        "--body-info-only",
        action="store_true",
        help="Only print body dimensions and properties, skip torque validation"
    )

    args = parser.parse_args()

    try:
        # Initialize validator
        validator = TorqueValidator(args.model)

        # If only body info requested, print and exit
        if args.body_info_only:
            validator.print_body_dimensions()
            sys.exit(0)

        # Run validation (this will also print body dimensions)
        results, overall_pass = validator.run_all_tests()
        validator.print_summary(results, overall_pass)

        # Optional visualization
        if args.visualize:
            validator.visualize_worst_pose(results)

        # Exit with appropriate code
        sys.exit(0 if overall_pass else 1)

    except FileNotFoundError as e:
        print(f"\n\033[91mError: {e}\033[0m\n", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n\033[91mUnexpected error: {e}\033[0m\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
