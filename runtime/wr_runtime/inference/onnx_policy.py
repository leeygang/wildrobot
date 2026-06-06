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

    def __init__(
        self,
        onnx_path: str,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        *,
        expected_obs_dim: Optional[int] = None,
        expected_action_dim: Optional[int] = None,
    ):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        in0 = self.session.get_inputs()[0]
        out0 = self.session.get_outputs()[0]

        self.input_name = input_name or in0.name
        self.output_name = output_name or out0.name
        # When provided (e.g. from policy_spec.json), these are enforced in
        # predict() so a wrong-sized obs/action can never silently broadcast
        # downstream (e.g. a length-1 action into a full joint target).
        self._expected_obs_dim = (
            int(expected_obs_dim) if expected_obs_dim is not None else None
        )
        self._expected_action_dim = (
            int(expected_action_dim) if expected_action_dim is not None else None
        )

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
        if self._expected_obs_dim is not None and obs.shape[0] != self._expected_obs_dim:
            raise ValueError(
                f"obs has {obs.shape[0]} elements, expected {self._expected_obs_dim}"
            )

        out = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1)})[0]
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if (
            self._expected_action_dim is not None
            and out.shape[0] != self._expected_action_dim
        ):
            raise ValueError(
                f"policy returned {out.shape[0]} actions, expected "
                f"{self._expected_action_dim}"
            )
        return out


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
