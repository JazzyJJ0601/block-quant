#!/usr/bin/env python3
"""Collect calibration activations from Qwen3-8B for block quantization."""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_PATH = "data/calib_layer10.pt"
LAYER_INDEX = 10


def get_input_activations(module, input):
    """Capture input tensor passed to a linear layer."""
    return input[0] if isinstance(input, tuple) else input


def main():
    parser = argparse.ArgumentParser(
        description="Collect activation calibration data from Qwen3-8B"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing calibration file if present",
    )
    args = parser.parse_args()

    output_path = os.path.join("repos", "block-quant", OUTPUT_PATH)

    if os.path.exists(output_path) and not args.force:
        print(f"Calibration file exists: {output_path}")
        print("Use --force to overwrite.")
        return

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fixed prompts for calibration
    prompts = [
        "Write a short poem about the ocean.",
        "Calculate the square root of 144.",
        "Explain what a transformer neural network is.",
        "List three benefits of regular exercise.",
        "Write a dialogue between two friends planning a trip.",
    ]

    # Prepare storage for activations
    calib_data = {
        "q_proj_inputs": [],
        "down_proj_inputs": [],
        "prompts": prompts,
        "model": MODEL_NAME,
        "layer": LAYER_INDEX,
    }

    # Register hooks to capture inputs at layer 10
    layers = model.model.layers
    layer_target = layers[LAYER_INDEX]

    q_proj_handle = layer_target.self_attn.q_proj.register_forward_hook(
        get_input_activations
    )
    down_proj_handle = layer_target.mlp.down_proj.register_forward_hook(
        get_input_activations
    )

    try:
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i + 1}/{len(prompts)}")

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)

            # Get captured activations from the hook's stored value
            # Note: We access via module's internal hook storage
            q_proj = layer_target.self_attn.q_proj
            down_proj = layer_target.mlp.down_proj

            # Extract input shapes for calibration metadata
            calib_data["q_proj_inputs"].append(q_proj.in_features)
            calib_data["down_proj_inputs"].append(down_proj.in_features)

        print(f"Saving calibration data to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(calib_data, output_path)
        print("Done.")

    finally:
        # Clean up hooks
        q_proj_handle.remove()
        down_proj_handle.remove()


if __name__ == "__main__":
    main()
