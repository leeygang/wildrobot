from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime


@dataclass(frozen=True)
class OnnxPolicyInfo:
    input_name: str
    output_name: str
    obs_dim: Optional[int]
    action_dim: Optional[int]


class OnnxPolicy:
    """Thin ONNXRuntime wrapper for running a policy.

    This is intentionally small and Raspberry-friendly.
    """

    def __init__(self, onnx_path: str, input_name: Optional[str] = None, output_name: Optional[str] = None):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        in0 = self.session.get_inputs()[0]
        out0 = self.session.get_outputs()[0]

        self.input_name = input_name or in0.name
        self.output_name = output_name or out0.name

        self.info = OnnxPolicyInfo(
            input_name=self.input_name,
            output_name=self.output_name,
            obs_dim=_try_get_last_dim(in0.shape),
            action_dim=_try_get_last_dim(out0.shape),
        )

    def predict(self, obs_1d: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_1d, dtype=np.float32)
        if obs.ndim != 1:
            raise ValueError(f"Expected obs shape (obs_dim,), got {obs.shape}")

        out = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1)})[0]
        out = np.asarray(out, dtype=np.float32)
        return out.reshape(-1)


def _try_get_last_dim(shape) -> Optional[int]:
    try:
        if not shape:
            return None
        last = shape[-1]
        if isinstance(last, int):
            return last
    except Exception:
        return None
    return None
