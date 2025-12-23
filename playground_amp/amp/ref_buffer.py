"""Reference motion buffer for AMP training.

For AMP training, we need reference motion data. This module provides:
1. Buffer for storing reference motion sequences (29-dim AMP features)
2. Loading reference features from pkl files

AMP Feature Format (29-dim):
- Joint positions: 0-8 (9)
- Joint velocities: 9-17 (9)
- Root linear velocity direction: 18-20 (3) - normalized unit vector
- Root angular velocity: 21-23 (3)
- Root height: 24 (1)
- Foot contacts: 25-28 (4)
"""

from collections import deque

import numpy as np


class ReferenceMotionBuffer:
    """Buffer for storing and sampling reference motion features.

    Stores AMP features (29-dim) directly from reference motion files.
    Features are sampled for discriminator training.
    """

    def __init__(self, max_size: int, seq_len: int, feature_dim: int):
        """Initialize reference motion buffer.

        Args:
            max_size: Maximum number of sequences to store (REQUIRED)
            seq_len: Length of each sequence in timesteps (REQUIRED)
            feature_dim: Feature dimension, must be 29 for AMP features (REQUIRED)
        """
        self.max_size = max_size
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.data = deque(maxlen=max_size)

    def add(self, seq: np.ndarray):
        """Add a sequence to the buffer.

        Args:
            seq: Sequence array of shape (seq_len, feature_dim)
        """
        if seq.shape[0] < self.seq_len:
            return
        if seq.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {seq.shape[1]}"
            )
        self.data.append(seq.copy())

    def sample(self, batch_size: int) -> np.ndarray:
        """Sample random sequences from buffer.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            batch: Array of shape (batch_size, seq_len, feature_dim)
        """
        if len(self.data) == 0:
            raise RuntimeError("Cannot sample from empty buffer. Load reference data first.")

        import random

        batch = random.sample(self.data, min(batch_size, len(self.data)))
        return np.stack(batch, axis=0).astype(np.float32)

    def sample_single_frames(self, batch_size: int) -> np.ndarray:
        """Sample random single frames (not sequences) from buffer.

        Useful for discriminator training where we classify individual states.

        Args:
            batch_size: Number of frames to sample

        Returns:
            frames: Array of shape (batch_size, feature_dim)
        """
        if len(self.data) == 0:
            raise RuntimeError("Cannot sample from empty buffer. Load reference data first.")

        import random

        frames = []
        for _ in range(batch_size):
            seq = random.choice(self.data)
            frame_idx = random.randint(0, seq.shape[0] - 1)
            frames.append(seq[frame_idx])

        return np.stack(frames, axis=0).astype(np.float32)

    def load_from_file(self, filepath: str):
        """Load reference motion data from pickle file.

        Expected format: numpy array of shape (num_frames, feature_dim)
        or dict with 'features' key.

        Args:
            filepath: Path to pkl file (REQUIRED)
        """
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if "features" in data:
                features = data["features"]
                self._load_from_features(features)
            else:
                raise ValueError(
                    f"Unknown dict format. Expected 'features' key, got: {data.keys()}"
                )
        elif isinstance(data, np.ndarray):
            self._load_from_features(data)
        else:
            raise ValueError(
                f"Unknown data format: {type(data)}. Expected dict with 'features' or numpy array."
            )

        print(f"Loaded {len(self.data)} reference motion sequences from {filepath}")

    def _load_from_features(self, features: np.ndarray):
        """Load from feature array of shape (num_frames, feature_dim).

        Stores features directly - no conversion needed since discriminator
        works with features, not observations.
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {features.shape}")

        num_frames, feature_dim = features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {feature_dim}. "
                "Buffer feature_dim must match the feature dimension of reference data."
            )

        # Split into overlapping sequences
        stride = max(1, self.seq_len // 2)  # 50% overlap
        for start_idx in range(0, num_frames - self.seq_len, stride):
            seq = features[start_idx : start_idx + self.seq_len]
            self.add(seq)

    def __len__(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0


def create_reference_buffer(
    feature_dim: int,
    seq_len: int,
    max_size: int,
    filepath: str,
) -> ReferenceMotionBuffer:
    """Create and populate a reference motion buffer from file.

    Args:
        feature_dim: Feature dimension, must be 29 for AMP features (REQUIRED)
        seq_len: Sequence length in timesteps (REQUIRED)
        max_size: Maximum buffer size (REQUIRED)
        filepath: Path to pkl file with reference features (REQUIRED)

    Returns:
        Populated ReferenceMotionBuffer
    """
    buffer = ReferenceMotionBuffer(
        max_size=max_size,
        seq_len=seq_len,
        feature_dim=feature_dim,
    )
    buffer.load_from_file(filepath)
    return buffer
