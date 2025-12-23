#!/usr/bin/env python3
"""Add temporal context to AMP reference data.

v0.5.0: Converts single-frame reference features (29 dims) to temporal
features (87 dims = 3 frames × 29 features) for improved AMP discriminator.

Usage:
    python scripts/add_temporal_context_to_ref.py \
        --input playground_amp/data/walking_motions_normalized_vel.pkl \
        --output playground_amp/data/walking_motions_temporal.pkl \
        --num-frames 3
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def add_temporal_context(
    features: np.ndarray,
    num_frames: int = 3,
) -> np.ndarray:
    """Convert single-frame features to temporal features.

    Creates overlapping windows of consecutive frames:
    - Input: (N, feature_dim)
    - Output: (N - num_frames + 1, num_frames * feature_dim)

    Args:
        features: Single-frame features, shape (N, feature_dim)
        num_frames: Number of frames per temporal window

    Returns:
        Temporal features, shape (N - num_frames + 1, num_frames * feature_dim)
    """
    N = features.shape[0]
    feature_dim = features.shape[1]

    # Number of valid windows
    num_windows = N - num_frames + 1

    if num_windows <= 0:
        raise ValueError(
            f"Not enough frames ({N}) for temporal window of size {num_frames}"
        )

    # Create windows manually for numpy
    temporal_features = np.zeros((num_windows, num_frames * feature_dim), dtype=features.dtype)

    for i in range(num_windows):
        # Concatenate frames [i, i+1, ..., i+num_frames-1]
        window = features[i:i+num_frames].flatten()
        temporal_features[i] = window

    return temporal_features


def main():
    parser = argparse.ArgumentParser(
        description="Add temporal context to AMP reference data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input reference data file (.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file with temporal features (.pkl)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="Number of frames in temporal window (default: 3)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading reference data from: {input_path}")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if "features" not in data:
        print("Error: Input file does not contain 'features' key")
        return 1

    features = data["features"]
    print(f"  Original features shape: {features.shape}")
    print(f"  Feature dim: {features.shape[1]}")
    print(f"  Num frames: {features.shape[0]}")

    # Add temporal context
    print(f"\nAdding temporal context with {args.num_frames} frames...")
    temporal_features = add_temporal_context(features, args.num_frames)

    print(f"  Temporal features shape: {temporal_features.shape}")
    print(f"  Temporal dim: {temporal_features.shape[1]}")
    print(f"  Num samples: {temporal_features.shape[0]}")
    print(f"  Lost frames: {features.shape[0] - temporal_features.shape[0]}")

    # Create output data
    output_data = {
        # Original metadata
        "fps": data.get("fps", 50.0),
        "dt": data.get("dt", 0.02),
        "source_files": data.get("source_files", []),
        "num_motions": data.get("num_motions", 1),
        "joint_names": data.get("joint_names", []),
        "notes": data.get("notes", ""),

        # Temporal features
        "features": temporal_features,
        "feature_dim": temporal_features.shape[1],
        "num_frames": temporal_features.shape[0],
        "duration_sec": temporal_features.shape[0] / data.get("fps", 50.0),

        # Temporal context metadata
        "temporal_context": {
            "num_frames_per_window": args.num_frames,
            "single_frame_dim": features.shape[1],
            "original_num_frames": features.shape[0],
        },
    }

    # Copy additional metadata
    for key in ["velocity_normalized", "dof_pos", "dof_vel"]:
        if key in data:
            output_data[key] = data[key]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\nSaved temporal features to: {output_path}")

    # Print statistics
    print("\n" + "=" * 50)
    print("Temporal Feature Statistics")
    print("=" * 50)

    # Per-frame statistics (split temporal features back into frames)
    for frame_idx in range(args.num_frames):
        start = frame_idx * features.shape[1]
        end = start + features.shape[1]
        frame_features = temporal_features[:, start:end]
        print(f"\nFrame {frame_idx} (t-{args.num_frames - 1 - frame_idx}):")
        print(f"  Mean: {frame_features.mean():.4f}")
        print(f"  Std:  {frame_features.std():.4f}")
        print(f"  Min:  {frame_features.min():.4f}")
        print(f"  Max:  {frame_features.max():.4f}")

    print("\n" + "=" * 50)
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
