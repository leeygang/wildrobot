"""Reference motion buffer with synthetic data generation.

For AMP training, we need reference motion data. This module provides:
1. Buffer for storing reference motion sequences
2. Synthetic motion generator (for testing without real MoCap data)
3. Data loading from pkl files (for real MoCap data when available)
"""

import numpy as np
from collections import deque
from typing import Optional
import jax.numpy as jnp


class ReferenceMotionBuffer:
    """Buffer for storing and sampling reference motion sequences.

    Stores sequences of observations (seq_len x obs_dim) up to max_size.
    Supports both synthetic generation and loading from files.
    """

    def __init__(self, max_size: int = 1000, seq_len: int = 32, obs_dim: int = 44):
        """Initialize reference motion buffer.

        Args:
            max_size: Maximum number of sequences to store
            seq_len: Length of each sequence (timesteps)
            obs_dim: Observation dimension
        """
        self.max_size = max_size
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.data = deque(maxlen=max_size)

    def add(self, seq: np.ndarray):
        """Add a sequence to the buffer.

        Args:
            seq: Sequence array of shape (seq_len, obs_dim)
        """
        if seq.shape[0] < self.seq_len:
            return
        if seq.shape[1] != self.obs_dim:
            print(f"Warning: Expected obs_dim={self.obs_dim}, got {seq.shape[1]}")
            return
        self.data.append(seq.copy())

    def sample(self, batch_size: int) -> np.ndarray:
        """Sample random sequences from buffer.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            batch: Array of shape (batch_size, seq_len, obs_dim)
        """
        if len(self.data) == 0:
            return np.zeros((0, self.seq_len, self.obs_dim), dtype=np.float32)

        import random
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        return np.stack(batch, axis=0).astype(np.float32)

    def sample_single_frames(self, batch_size: int) -> np.ndarray:
        """Sample random single frames (not sequences) from buffer.

        Useful for discriminator training where we classify individual states.

        Args:
            batch_size: Number of frames to sample

        Returns:
            frames: Array of shape (batch_size, obs_dim)
        """
        if len(self.data) == 0:
            return np.zeros((0, self.obs_dim), dtype=np.float32)

        import random
        frames = []
        for _ in range(batch_size):
            # Sample random sequence
            seq = random.choice(self.data)
            # Sample random frame from sequence
            frame_idx = random.randint(0, seq.shape[0] - 1)
            frames.append(seq[frame_idx])

        return np.stack(frames, axis=0).astype(np.float32)

    def load_from_file(self, filepath: str):
        """Load reference motion data from pickle file.

        Expected format: numpy array of shape (num_frames, obs_dim)
        or list of sequences.

        Args:
            filepath: Path to pkl file
        """
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            # Single long sequence - split into chunks
            if data.ndim == 2:  # (num_frames, obs_dim)
                num_frames = data.shape[0]
                for start_idx in range(0, num_frames - self.seq_len, self.seq_len // 2):
                    seq = data[start_idx : start_idx + self.seq_len]
                    self.add(seq)
            else:
                raise ValueError(f"Expected 2D array, got shape {data.shape}")

        elif isinstance(data, list):
            # List of sequences
            for seq in data:
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    self.add(seq)

        print(f"Loaded {len(self.data)} reference motion sequences from {filepath}")

    def generate_synthetic_standing_pose(self, num_sequences: int = 100):
        """Generate synthetic reference motion (stable standing pose).

        This is for testing AMP integration without real MoCap data.
        Generates a stable standing pose with small random perturbations.

        Args:
            num_sequences: Number of sequences to generate
        """
        # Default standing pose (11 joint positions + 11 joint velocities = 22 dims)
        # Plus other observation features up to obs_dim

        # Joint positions for standing (roughly zero for small humanoid)
        default_qpos = np.zeros(11, dtype=np.float32)
        default_qvel = np.zeros(11, dtype=np.float32)

        # Full observation (44-dim for WildRobot)
        # Simplified: [base_height, base_ori(6), joint_pos(11), base_lin_vel(3),
        #              base_ang_vel(3), joint_vel(11), contact(4), actions(11)]
        # For reference motion, we care about joint poses primarily

        for _ in range(num_sequences):
            sequence = []
            for t in range(self.seq_len):
                # Standing pose with small noise
                obs = np.zeros(self.obs_dim, dtype=np.float32)

                # Base height (around 0.5m for standing)
                obs[0] = 0.5 + np.random.normal(0, 0.01)

                # Base orientation (identity quaternion: [1,0,0,0] as 6D repr)
                obs[1:7] = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)  # Simplified 6D rot

                # Joint positions (small noise around zero)
                obs[7:18] = default_qpos + np.random.normal(0, 0.02, size=11).astype(np.float32)

                # Velocities (near zero for standing)
                obs[18:32] = np.random.normal(0, 0.01, size=14).astype(np.float32)

                # Joint velocities
                obs[32:43] = default_qvel + np.random.normal(0, 0.05, size=11).astype(np.float32)

                # Contact forces (feet on ground) - last 4 dimensions
                obs[40:44] = np.array([0.2, 0.2, 0, 0], dtype=np.float32) # Front feet have contact

                sequence.append(obs)

            seq_array = np.stack(sequence, axis=0)  # (seq_len, obs_dim)
            self.add(seq_array)

        print(f"Generated {num_sequences} synthetic standing pose sequences")
        print(f"  Buffer now has {len(self.data)} sequences")
        print(f"  Each sequence: ({self.seq_len}, {self.obs_dim})")

    def generate_synthetic_walking_motion(self, num_sequences: int = 100):
        """Generate synthetic walking motion (simple sinusoidal gait).

        This generates a simple periodic walking motion using sinusoids.
        Not as good as real MoCap, but better than standing for AMP testing.

        Args:
            num_sequences: Number of sequences to generate
        """
        for _ in range(num_sequences):
            sequence = []

            # Random phase offset for variety
            phase_offset = np.random.uniform(0, 2 * np.pi)
            stride_freq = 2.0  # Hz (1 step every 0.5s)

            for t in range(self.seq_len):
                time = t * 0.02  # Assuming 50Hz control (0.02s per step)
                phase = 2 * np.pi * stride_freq * time + phase_offset

                obs = np.zeros(self.obs_dim, dtype=np.float32)

                # Base height (slight bobbing during walk)
                obs[0] = 0.5 + 0.02 * np.sin(2 * phase)

                # Base orientation (small pitch oscillation)
                obs[1:7] = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)

                # Joint positions (simplified alternating leg motion)
                hip_l = 0.3 * np.sin(phase)          # Left hip
                hip_r = 0.3 * np.sin(phase + np.pi)  # Right hip (opposite)
                knee_l = 0.5 * np.abs(np.sin(phase))
                knee_r = 0.5 * np.abs(np.sin(phase + np.pi))

                obs[7:18] = np.array([
                    hip_l, knee_l, 0.0,  # Left leg
                    hip_r, knee_r, 0.0,  # Right leg
                    0.0, 0.0,            # Arms
                    0.0, 0.0, 0.0        # Other joints
                ], dtype=np.float32)

                # Base velocities (forward motion)
                obs[18:21] = np.array([0.4, 0.0, 0.0], dtype=np.float32)  # Forward velocity
                obs[21:24] = np.zeros(3, dtype=np.float32)  # Angular velocity

                # Joint velocities (derivatives of positions)
                omega = 2 * np.pi * stride_freq
                obs[32:43] = np.array([
                    0.3 * omega * np.cos(phase),
                    0.5 * omega * np.cos(phase) * np.sign(np.sin(phase)),
                    0.0,
                    0.3 * omega * np.cos(phase + np.pi),
                    0.5 * omega * np.cos(phase + np.pi) * np.sign(np.sin(phase + np.pi)),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ], dtype=np.float32)

                # Contact forces (alternating feet) - last 4 dimensions
                left_contact = 0.5 if np.sin(phase) < 0 else 0.0
                right_contact = 0.5 if np.sin(phase + np.pi) < 0 else 0.0
                obs[40:44] = np.array([left_contact, right_contact, 0, 0], dtype=np.float32)

                sequence.append(obs)

            seq_array = np.stack(sequence, axis=0)
            self.add(seq_array)

        print(f"Generated {num_sequences} synthetic walking motion sequences")
        print(f"  Buffer now has {len(self.data)} sequences")

    def __len__(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0


def create_reference_buffer(
    obs_dim: int = 44,
    seq_len: int = 32,
    max_size: int = 1000,
    mode: str = "walking",
    num_sequences: int = 200,
    filepath: Optional[str] = None
) -> ReferenceMotionBuffer:
    """Create and populate a reference motion buffer.

    Args:
        obs_dim: Observation dimension
        seq_len: Sequence length
        max_size: Maximum buffer size
        mode: Generation mode ("standing", "walking", or "file")
        num_sequences: Number of sequences to generate (if synthetic)
        filepath: Path to pkl file (if mode="file")

    Returns:
        Populated ReferenceMotionBuffer
    """
    buffer = ReferenceMotionBuffer(
        max_size=max_size,
        seq_len=seq_len,
        obs_dim=obs_dim
    )

    if mode == "file" and filepath is not None:
        buffer.load_from_file(filepath)
    elif mode == "walking":
        buffer.generate_synthetic_walking_motion(num_sequences)
    elif mode == "standing":
        buffer.generate_synthetic_standing_pose(num_sequences)
    else:
        print(f"Warning: Unknown mode '{mode}', buffer will be empty")

    return buffer
