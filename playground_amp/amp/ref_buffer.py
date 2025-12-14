import random
import numpy as np
from collections import deque

class ReferenceMotionBuffer:
    """Simple replay buffer for reference motion snippets.

    Stores sequences of observations (seq_len x obs_dim) up to max_size.
    """
    def __init__(self, max_size: int = 1000, seq_len: int = 32):
        self.max_size = max_size
        self.seq_len = seq_len
        self.data = deque(maxlen=max_size)

    def add(self, seq: np.ndarray):
        # seq: (seq_len, obs_dim)
        if seq.shape[0] < self.seq_len:
            return
        self.data.append(seq.copy())

    def sample(self, batch_size: int) -> np.ndarray:
        if len(self.data) == 0:
            return np.zeros((0, self.seq_len, 0))
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        return np.stack(batch, axis=0)

    def __len__(self):
        return len(self.data)
