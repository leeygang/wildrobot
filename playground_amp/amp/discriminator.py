import numpy as np


class AMPDiscriminator:
    """Small MLP discriminator implemented in NumPy for smoke-testing.

    - `score(obs_batch)` returns a scalar score per observation in [0,1]
    - `update(batch)` is a no-op placeholder for now
    """

    def __init__(self, input_dim: int, hidden=(64, 64), seed: int = 0):
        self.input_dim = input_dim
        self.hidden = tuple(hidden)
        self.rng = np.random.RandomState(seed)
        # initialize weights (list of (w,b) tuples)
        layer_sizes = [input_dim] + list(self.hidden) + [1]
        self.params = []
        for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = 0.01 * self.rng.randn(a, b).astype(np.float32)
            bl = np.zeros((b,), dtype=np.float32)
            self.params.append((w, bl))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32)
        for i, (w, b) in enumerate(self.params):
            h = h.dot(w) + b
            if i < len(self.params) - 1:
                h = np.tanh(h)
        # sigmoid output
        out = 1.0 / (1.0 + np.exp(-h))
        return out.squeeze(-1)

    def score(self, obs_batch: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs_batch, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        try:
            return self._forward(arr)
        except Exception:
            return np.zeros((arr.shape[0],), dtype=np.float32)

    def update(self, batch: np.ndarray):
        # placeholder: no training for now
        return
