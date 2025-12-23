#!/usr/bin/env python3
"""Analyze velocity distribution in reference motion data.

Checks if reference motion velocities match the commanded speed range (0.5-1.0 m/s).
A mismatch could cause the discriminator to penalize the policy for legitimate speed differences.

Usage:
    python scripts/analyze_reference_velocities.py
"""

import pickle
import numpy as np
from pathlib import Path


def analyze_reference_velocities(data_path: str):
    """Analyze velocity distribution in reference motion data."""

    print(f"Loading reference data from: {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Extract features
    if isinstance(data, dict) and "features" in data:
        features = np.array(data["features"])
    else:
        features = np.array(data)

    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {features.shape[-1]}")

    # AMP feature layout (29-dim):
    # - joint_pos: 0-8 (9 dims)
    # - joint_vel: 9-17 (9 dims)
    # - root_linvel: 18-20 (3 dims) <-- THIS IS WHAT WE CARE ABOUT
    # - root_angvel: 21-23 (3 dims)
    # - root_height: 24 (1 dim)
    # - foot_contacts: 25-28 (4 dims)

    # Extract root linear velocity (x, y, z)
    root_linvel = features[:, 18:21]

    # Forward velocity is typically X or the magnitude of XY
    vel_x = root_linvel[:, 0]
    vel_y = root_linvel[:, 1]
    vel_z = root_linvel[:, 2]

    # Forward speed (magnitude of horizontal velocity)
    forward_speed = np.sqrt(vel_x**2 + vel_y**2)

    print("\n" + "=" * 60)
    print("ROOT LINEAR VELOCITY ANALYSIS")
    print("=" * 60)

    print("\nVelocity X (forward):")
    print(f"  Min:  {vel_x.min():.3f} m/s")
    print(f"  Max:  {vel_x.max():.3f} m/s")
    print(f"  Mean: {vel_x.mean():.3f} m/s")
    print(f"  Std:  {vel_x.std():.3f} m/s")

    print("\nVelocity Y (lateral):")
    print(f"  Min:  {vel_y.min():.3f} m/s")
    print(f"  Max:  {vel_y.max():.3f} m/s")
    print(f"  Mean: {vel_y.mean():.3f} m/s")
    print(f"  Std:  {vel_y.std():.3f} m/s")

    print("\nVelocity Z (vertical):")
    print(f"  Min:  {vel_z.min():.3f} m/s")
    print(f"  Max:  {vel_z.max():.3f} m/s")
    print(f"  Mean: {vel_z.mean():.3f} m/s")
    print(f"  Std:  {vel_z.std():.3f} m/s")

    print("\nForward Speed (horizontal magnitude):")
    print(f"  Min:  {forward_speed.min():.3f} m/s")
    print(f"  Max:  {forward_speed.max():.3f} m/s")
    print(f"  Mean: {forward_speed.mean():.3f} m/s")
    print(f"  Std:  {forward_speed.std():.3f} m/s")

    # Compare with commanded velocity range
    cmd_min, cmd_max = 0.5, 1.0

    print("\n" + "=" * 60)
    print("COMPARISON WITH COMMANDED VELOCITY RANGE")
    print("=" * 60)
    print(f"Commanded range: {cmd_min} - {cmd_max} m/s")
    print(f"Reference mean:  {forward_speed.mean():.3f} m/s")

    # Check what percentage of reference data falls in commanded range
    in_range = np.sum((forward_speed >= cmd_min) & (forward_speed <= cmd_max))
    below_range = np.sum(forward_speed < cmd_min)
    above_range = np.sum(forward_speed > cmd_max)
    total = len(forward_speed)

    print(f"\nReference data distribution:")
    print(f"  Below {cmd_min} m/s: {below_range:,} ({100*below_range/total:.1f}%)")
    print(f"  In range [{cmd_min}-{cmd_max}]: {in_range:,} ({100*in_range/total:.1f}%)")
    print(f"  Above {cmd_max} m/s: {above_range:,} ({100*above_range/total:.1f}%)")

    # Percentiles
    print(f"\nReference velocity percentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  {p}th percentile: {np.percentile(forward_speed, p):.3f} m/s")

    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    ref_mean = forward_speed.mean()
    if ref_mean < cmd_min:
        print(f"⚠️  MISMATCH: Reference data is SLOWER than commanded range")
        print(f"   Reference mean ({ref_mean:.2f}) < Commanded min ({cmd_min})")
        print(f"   The discriminator may penalize the policy for moving too fast!")
    elif ref_mean > cmd_max:
        print(f"⚠️  MISMATCH: Reference data is FASTER than commanded range")
        print(f"   Reference mean ({ref_mean:.2f}) > Commanded max ({cmd_max})")
        print(f"   The discriminator may penalize the policy for moving too slow!")
    else:
        print(f"✓  Reference velocity ({ref_mean:.2f} m/s) is within commanded range")

    if in_range / total < 0.5:
        print(f"\n⚠️  WARNING: Only {100*in_range/total:.1f}% of reference data is in commanded range")
        print(f"   Consider adjusting commanded velocity range or using different reference motions")

    return {
        "forward_speed_mean": forward_speed.mean(),
        "forward_speed_std": forward_speed.std(),
        "pct_in_range": in_range / total,
    }


if __name__ == "__main__":
    # Default path
    data_path = "playground_amp/data/walking_motions_merged.pkl"

    if not Path(data_path).exists():
        print(f"Error: {data_path} not found")
        print("Trying alternative paths...")
        alternatives = [
            "playground_amp/data/walking_motions.pkl",
            "data/walking_motions.pkl",
        ]
        for alt in alternatives:
            if Path(alt).exists():
                data_path = alt
                break
        else:
            print("No reference data found!")
            exit(1)

    analyze_reference_velocities(data_path)
