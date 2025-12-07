#!/usr/bin/env python3
"""
Export trained policy to ONNX format for hardware deployment.

This script converts a trained JAX policy to ONNX format that can be used
with the runtime package on embedded systems (Raspberry Pi 5).
"""

import argparse
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
import onnx
from onnx import helper, TensorProto

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_policy_to_onnx(policy_path: str, output_path: str, obs_size: int = 57, action_size: int = 11):
    """
    Export a trained JAX policy to ONNX format.

    Args:
        policy_path: Path to the .pkl file containing policy parameters
        output_path: Path to save the ONNX model
        obs_size: Size of observation vector (default: 57 for WildRobot)
        action_size: Size of action vector (default: 11 for WildRobot)
    """
    print(f"Loading policy from: {policy_path}")
    with open(policy_path, "rb") as f:
        params = pickle.load(f)

    # Extract policy network parameters
    # This assumes the standard Brax PPO network structure
    policy_params = params['policy']

    print(f"Policy parameters keys: {policy_params.keys()}")

    # Build ONNX graph manually
    # For a simple MLP: obs -> hidden1 -> hidden2 -> hidden3 -> action
    nodes = []
    initializers = []
    inputs = []
    outputs = []

    # Input
    inputs.append(
        helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, obs_size])
    )

    # Extract weights and biases
    layer_idx = 0
    prev_output = "observation"

    for key in sorted(policy_params.keys()):
        if 'kernel' in key or 'w' in key.lower():
            # Weight matrix
            weight = np.array(policy_params[key])
            weight_name = f"weight_{layer_idx}"
            initializers.append(
                helper.make_tensor(
                    weight_name,
                    TensorProto.FLOAT,
                    weight.shape,
                    weight.flatten().tolist()
                )
            )

            # Bias
            bias_key = key.replace('kernel', 'bias').replace('w', 'b')
            if bias_key in policy_params:
                bias = np.array(policy_params[bias_key])
                bias_name = f"bias_{layer_idx}"
                initializers.append(
                    helper.make_tensor(
                        bias_name,
                        TensorProto.FLOAT,
                        bias.shape,
                        bias.flatten().tolist()
                    )
                )
            else:
                bias_name = None

            # MatMul node
            matmul_output = f"matmul_{layer_idx}"
            nodes.append(
                helper.make_node(
                    "MatMul",
                    inputs=[prev_output, weight_name],
                    outputs=[matmul_output]
                )
            )

            # Add bias if present
            if bias_name:
                add_output = f"add_{layer_idx}"
                nodes.append(
                    helper.make_node(
                        "Add",
                        inputs=[matmul_output, bias_name],
                        outputs=[add_output]
                    )
                )
                prev_output = add_output
            else:
                prev_output = matmul_output

            # Activation (ReLU for hidden layers, none for output layer)
            if layer_idx < len([k for k in policy_params.keys() if 'kernel' in k or 'w' in k.lower()]) - 1:
                relu_output = f"relu_{layer_idx}"
                nodes.append(
                    helper.make_node(
                        "Relu",
                        inputs=[prev_output],
                        outputs=[relu_output]
                    )
                )
                prev_output = relu_output

            layer_idx += 1

    # Output
    outputs.append(
        helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_size])
    )

    # Final output node
    nodes.append(
        helper.make_node(
            "Identity",
            inputs=[prev_output],
            outputs=["action"]
        )
    )

    # Create graph
    graph_def = helper.make_graph(
        nodes,
        "wildrobot_policy",
        inputs,
        outputs,
        initializers
    )

    # Create model
    model_def = helper.make_model(graph_def, producer_name="wildrobot_exporter")

    # Check and save
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)
    print(f"ONNX model saved to: {output_path}")
    print(f"Model info:")
    print(f"  Input: observation [{obs_size}]")
    print(f"  Output: action [{action_size}]")
    print(f"  Layers: {layer_idx}")


def main():
    parser = argparse.ArgumentParser(description="Export policy to ONNX")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy .pkl file")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--obs-size", type=int, default=57, help="Observation size")
    parser.add_argument("--action-size", type=int, default=11, help="Action size")

    args = parser.parse_args()

    if args.output is None:
        policy_path = Path(args.policy)
        args.output = str(policy_path.parent / "policy.onnx")

    export_policy_to_onnx(
        args.policy,
        args.output,
        obs_size=args.obs_size,
        action_size=args.action_size
    )


if __name__ == "__main__":
    main()
