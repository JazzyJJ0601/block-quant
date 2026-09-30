#!/usr/bin/env python3
"""
Benchmark block-quantization vs static min-max quantization.

Compares reconstruction error (Frobenius norm) at 2, 4, and 8 bits
for weights from Qwen3-8B layer 10, using learned scales vs static scales.
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path

from block_quant import BlockQuantizer, quantize_block

# Model and data settings
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER_INDEX = 10
CALIB_SAMPLES = 20
BLOCK_SIZE = 64
OPTIMIZER = "adam"
LEARNING_RATE = 1e-3
STEPS = 100


def load_layer_weights(model_name: str, layer_idx: int) -> torch.Tensor:
    """Load output projection weights from specified transformer layer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        # Access layer 10 output projection
        weights = model.model.layers[layer_idx].mlp.up_proj.weight.detach().clone()
        return weights
    except Exception as e:
        print(f"Warning: Could not load weights from {model_name}: {e}")
        print("Using placeholder weights for benchmarking.")
        return torch.randn(3200, 2304)


def calibrate_scales(weights: torch.Tensor, calib_data: list[torch.Tensor]) -> torch.Tensor:
    """Initialize block scales using activation statistics from calibration set."""
    num_blocks = weights.shape[-1] // BLOCK_SIZE
    scales = torch.ones(num_blocks)
    
    for activation in calib_data:
        act_per_block = torch.mean(activation.view(-1, BLOCK_SIZE), dim=-1)
        block_scales = torch.std(act_per_block, keepdim=True).clamp(min=1e-5)
        scales = torch.maximum(scales, block_scales.repeat(num_blocks))
    
    return scales


def quantize_block_learnable(
    weights: torch.Tensor,
    bit_width: int,
    block_size: int,
    steps: int,
    lr: float,
    optimizer: str = "adam",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Block quantization with learnable scales using straight-through estimator.
    Trains scales for given number of steps to minimize reconstruction error.
    """
    num_blocks = weights.shape[-1] // block_size
    scales = torch.ones(num_blocks, requires_grad=True)
    quantizer = BlockQuantizer(bit_width, block_size)
    quantizer.scales = nn.Parameter(scales)
    
    if optimizer == "adam":
        opt = torch.optim.Adam([scales], lr=lr)
    else:
        opt = torch.optim.SGD([scales], lr=lr)
    
    for step in range(steps):
        opt.zero_grad()
        dequantized, _ = quantizer(weights)
        loss = F.mse_loss(dequantized, weights)
        loss.backward()
        opt.step()
    
    return dequantized, scales.detach()


def quantize_block_static(weights: torch.Tensor, bit_width: int, block_size: int) -> torch.Tensor:
    """
    Static min-max block quantization.
    Scales computed once from min-max statistics per block, then fixed.
    """
    num_blocks = weights.shape[-1] // block_size
    reshaped = weights.view(-1, block_size)
    min_vals = torch.min(reshaped, dim=-1, keepdim=True).values
    max_vals = torch.max(reshaped, dim=-1, keepdim=True)
    
    scales = (max_vals - min_vals).clamp(min=1e-5)
    
    block = BlockQuantizer(bit_width, block_size)
    block.scales = nn.Parameter(scales.view(-1))
    
    dequantized, _ = block(weights)
    return dequantized


def compute_reconstruction_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Compute Frobenius norm of reconstruction error."""
    return torch.norm(original - quantized, p="fro").item()


def run_benchmark():
    """Run full benchmark comparing block-quant vs static quantization."""
    print(f"Loading weights from {MODEL_NAME}, layer {LAYER_INDEX}...")
    weights = load_layer_weights(MODEL_NAME, LAYER_INDEX)
    print(f"Weight shape: {weights.shape}")
    
    # Generate calibration data (mock activations)
    calibration_set = [torch.randn(weights.shape[0], 1024) for _ in range(CALIB_SAMPLES)]
    
    # Initialize static scales from calibration data
    static_scales = calibrate_scales(weights, calibration_set)
    
    results = {}
    bits_list = [2, 4, 8]
    
    for bit in bits_list:
        print(f"Testing {bit}-bit quantization...")
        
        # Block-quant (learnable scales)
        dequant_block, _ = quantize_block_learnable(
            weights, bit, BLOCK_SIZE, STEPS, LEARNING_RATE
        )
        error_block = compute_reconstruction_error(weights, dequant_block)
        
        # Static min-max quantization
        dequant_static = quantize_block_static(weights, bit, BLOCK_SIZE)
        error_static = compute_reconstruction_error(weights, dequant_static)
        
        results[bit] = {
            "block_quant_error": error_block,
            "static_quant_error": error_static,
            "improvement": (error_static - error_block) / error_static * 100,
        }
        print(f"  Block-quant error: {error_block:.6f}")
        print(f"  Static error:      {error_static:.6f}")
    
    # Save results
    output_path = Path(__file__).parent / "results" / "benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
