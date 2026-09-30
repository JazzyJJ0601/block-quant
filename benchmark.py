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

from block_quant import BlockQuantizer

# Model and data settings
MODEL_NAME = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
LAYER_INDEX = 10
BLOCK_SIZE = 64
LEARNING_RATE = 1e-3
STEPS = 100


def load_layer_weights(model_name: str, layer_idx: int) -> torch.Tensor:
    """Load mlp.up_proj weights from specified transformer layer."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    w = model.model.layers[layer_idx].mlp.up_proj.weight.detach().clone()
    print(f"Weight shape: {w.shape}")
    return w


def compute_rel_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Frobenius norm of reconstruction error relative to original norm."""
    denom = torch.norm(original, p="fro").item()
    if denom < 1e-10:
        return 0.0
    return torch.norm(original - quantized, p="fro").item() / denom


def run_benchmark():
    """Run full benchmark comparing block-quant vs static quantization."""
    print(f"Loading weights from {MODEL_NAME}, layer {LAYER_INDEX}...")
    weights = load_layer_weights(MODEL_NAME, LAYER_INDEX)
    print(f"Weight shape: {weights.shape}")

    results = {}
    bits_list = [2, 4, 8]

    for bit in bits_list:
        print(f"\nTesting {bit}-bit quantization...")

        # --- Static min-max quantization ---
        # Quantizer initialises scales from data automatically
        q_static = BlockQuantizer(bit, BLOCK_SIZE)
        dequant_static, s_static = q_static(weights)
        err_static = compute_rel_error(weights, dequant_static)

        # --- Learnable scales (STE gradient descent) ---
        q_learn = BlockQuantizer(bit, BLOCK_SIZE)

        # First forward pass to initialise scales from data
        dequant_initial, _ = q_learn(weights)
        err_initial = compute_rel_error(weights, dequant_initial)

        # Optimise scales via gradient descent
        opt = torch.optim.Adam([q_learn._scales], lr=LEARNING_RATE)
        for step in range(STEPS):
            opt.zero_grad()
            dequant, _ = q_learn(weights)
            loss = F.mse_loss(dequant, weights)
            loss.backward()
            opt.step()

        # Final eval with trained scales
        dequant_learn, s_learn = q_learn(weights)
        err_learn = compute_rel_error(weights, dequant_learn)

        improvement = (err_static - err_learn) / max(err_static, 1e-10) * 100
        results[bit] = {
            "block_quant_error": round(err_learn, 6),
            "static_quant_error": round(err_static, 6),
            "improvement_pct": round(improvement, 2),
        }
        print(f"  Static error:       {err_static:.6f}")
        print(f"  Block-quant error:  {err_learn:.6f}  (after {STEPS} steps)")
        print(f"  Improvement:        {improvement:.1f}%")

    # Save results
    output_path = Path(__file__).parent / "results" / "benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    run_benchmark()