#!/usr/bin/env python3
"""Export a `playground_amp` PPO checkpoint (.pkl) to a deterministic ONNX policy.

This exporter targets the PPO policy produced by `playground_amp/train.py`.
The checkpoint contains `policy_params` for a Brax MLP policy network.

The exported ONNX model implements *deterministic* action selection:
- logits = MLP(observation)
- mean = logits[:, :action_dim]
- action = tanh(mean)

That matches Brax's `NormalTanhDistribution.mode()` for deterministic inference.

Usage:
  uv run python playground_amp/export_onnx.py \
    --checkpoint playground_amp/checkpoints/ppo_standing_v00113_final.pkl \
    --output playground_amp/checkpoints/ppo_standing_v00113_final.onnx

Notes:
- The runtime expects input shape [1, obs_dim] and outputs [1, action_dim].
- If you export the wrong dims, the runtime startup validator will fail.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


_LAYER_RE = re.compile(r"^hidden_(\d+)$")


def _as_np(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        raise TypeError(f"Unexpected object array for param: {type(a)}")
    return arr.astype(np.float32)


def _extract_mlp_layers(policy_params: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract [(W, b), ...] from Brax policy_params['params'] MLP."""
    if "params" not in policy_params or not isinstance(policy_params["params"], dict):
        raise ValueError("Expected checkpoint['policy_params'] to have a dict key 'params'.")

    params = policy_params["params"]

    layers: List[Tuple[int, Dict[str, Any]]] = []
    for name, value in params.items():
        if not isinstance(value, dict):
            continue
        m = _LAYER_RE.match(str(name))
        if not m:
            continue
        idx = int(m.group(1))
        layers.append((idx, value))

    if not layers:
        raise ValueError(f"No MLP layers found under policy_params['params'] (keys={list(params.keys())[:20]}...).")

    layers.sort(key=lambda t: t[0])

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx, layer_dict in layers:
        if "kernel" not in layer_dict or "bias" not in layer_dict:
            raise ValueError(f"Layer hidden_{idx} missing kernel/bias keys (keys={list(layer_dict.keys())}).")
        w = _as_np(layer_dict["kernel"])
        b = _as_np(layer_dict["bias"])
        if w.ndim != 2 or b.ndim != 1:
            raise ValueError(f"Unexpected shapes for hidden_{idx}: kernel={w.shape}, bias={b.shape}")
        if w.shape[1] != b.shape[0]:
            raise ValueError(f"Shape mismatch for hidden_{idx}: kernel={w.shape} bias={b.shape}")
        out.append((w, b))

    # Sanity: chain dimensions
    for i in range(1, len(out)):
        prev_w, _ = out[i - 1]
        w, _ = out[i]
        if prev_w.shape[1] != w.shape[0]:
            raise ValueError(
                "Incompatible layer chain: "
                f"layer{i-1}.out={prev_w.shape[1]} != layer{i}.in={w.shape[0]}"
            )

    return out


def get_checkpoint_dims(checkpoint_path: Path) -> tuple[int, int]:
    ckpt = pickle.loads(checkpoint_path.read_bytes())
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint did not load as dict: {checkpoint_path}")

    policy_params = ckpt.get("policy_params")
    if not isinstance(policy_params, dict):
        raise ValueError("Expected checkpoint['policy_params'] to be a dict.")

    layers = _extract_mlp_layers(policy_params)

    obs_dim = int(layers[0][0].shape[0])
    logits_dim = int(layers[-1][1].shape[0])
    if logits_dim % 2 != 0:
        raise ValueError(f"Expected final logits_dim to be even (mean+scale), got {logits_dim}")
    action_dim = logits_dim // 2
    return obs_dim, action_dim


def export_checkpoint_to_onnx(checkpoint_path: Path, output_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'onnx'. Install it in your training env, e.g. 'uv sync' after adding onnx, "
            "or 'uv pip install onnx'."
        ) from exc

    ckpt = pickle.loads(checkpoint_path.read_bytes())
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint did not load as dict: {checkpoint_path}")

    policy_params = ckpt.get("policy_params")
    if not isinstance(policy_params, dict):
        raise ValueError("Expected checkpoint['policy_params'] to be a dict.")

    layers = _extract_mlp_layers(policy_params)

    obs_dim = int(layers[0][0].shape[0])
    logits_dim = int(layers[-1][1].shape[0])
    if logits_dim % 2 != 0:
        raise ValueError(f"Expected final logits_dim to be even (mean+scale), got {logits_dim}")
    action_dim = logits_dim // 2

    # Build ONNX graph
    nodes = []
    initializers = []

    input_name = "observation"
    output_name = "action"

    inputs = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, obs_dim])]
    outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, action_dim])]

    prev = input_name

    def add_initializer(name: str, arr: np.ndarray) -> None:
        initializers.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.FLOAT,
                dims=list(arr.shape),
                vals=arr.flatten().astype(np.float32).tolist(),
            )
        )

    # MLP: (MatMul+Add) + SiLU for hidden layers
    for i, (w, b) in enumerate(layers):
        w_name = f"W{i}"
        b_name = f"b{i}"
        add_initializer(w_name, w)
        add_initializer(b_name, b)

        mm = f"mm{i}"
        lin = f"lin{i}"
        nodes.append(helper.make_node("MatMul", inputs=[prev, w_name], outputs=[mm]))
        nodes.append(helper.make_node("Add", inputs=[mm, b_name], outputs=[lin]))

        is_last = i == (len(layers) - 1)
        if not is_last:
            # SiLU(x) = x * sigmoid(x)
            sig = f"sig{i}"
            act = f"act{i}"
            nodes.append(helper.make_node("Sigmoid", inputs=[lin], outputs=[sig]))
            nodes.append(helper.make_node("Mul", inputs=[lin, sig], outputs=[act]))
            prev = act
        else:
            prev = lin  # logits

    logits = prev

    # mean = logits[:, :action_dim]
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([1, action_dim], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)

    def add_i64(name: str, arr: np.ndarray) -> str:
        initializers.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.INT64,
                dims=list(arr.shape),
                vals=arr.flatten().tolist(),
            )
        )
        return name

    starts_name = add_i64("slice_starts", starts)
    ends_name = add_i64("slice_ends", ends)
    axes_name = add_i64("slice_axes", axes)
    steps_name = add_i64("slice_steps", steps)

    mean = "mean"
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=[logits, starts_name, ends_name, axes_name, steps_name],
            outputs=[mean],
        )
    )

    # action = tanh(mean)
    nodes.append(helper.make_node("Tanh", inputs=[mean], outputs=[output_name]))

    graph = helper.make_graph(
        nodes=nodes,
        name="playground_amp_policy",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="playground_amp.export_onnx",
        opset_imports=[helper.make_opsetid("", 13)],
    )

    # Some embedded/older onnxruntime builds reject newer IR versions.
    # Keep this conservative for Raspberry compatibility.
    model.ir_version = min(int(getattr(model, "ir_version", 11)), 11)

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    print("Exported ONNX policy")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output:     {output_path}")
    print(f"  obs_dim:    {obs_dim}")
    print(f"  action_dim: {action_dim}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export playground_amp PPO checkpoint to deterministic ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pkl")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path (default: рядом с checkpoint)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = Path(args.output)

    export_checkpoint_to_onnx(checkpoint_path=checkpoint_path, output_path=output_path)


if __name__ == "__main__":
    main()
