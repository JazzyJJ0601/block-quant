# Block-Quant

**Learnable block-wise quantization with straight-through estimator.**

Post-training quantization that optimises per-block scale factors instead of fixing them from calibration statistics. The scale factors are learned via gradient descent, minimising reconstruction error through a straight-through estimator (STE).

## The Idea

Standard block-wise quantization computes scale factors once from min/max statistics. Block-Quant treats those scales as trainable parameters — you initialise from calibration data, then take a few gradient steps to minimise `||W - Q(W)||_2`. Because each block gets its own scale, the method adapts to local weight structure without adding inference cost.

## Implementation

- **BlockQuantizer** — PyTorch module with learnable `nn.Parameter` scales per block
- **Straight-through estimator** — gradient passes through the rounding step as identity
- **Supports 2, 4, and 8 bit** symmetric quantization
- **Convenience function** `quantize_block(x, bit_width, block_size)` for single-shot use

## Usage

```python
from block_quant import BlockQuantizer, quantize_block

# Direct use
q = BlockQuantizer(bit_width=4, block_size=64)
dequantized, scales = q(weight_tensor)

# Convenience
dequantized, scales = quantize_block(weight_tensor, bit_width=4, block_size=64)
```

## Benchmark results

*Not yet run on real model weights.* The `benchmark.py` script loads Qwen3-8B layer 10 weights and compares learnable scales vs static min-max scales:

```bash
python benchmark.py
```

## Design

See [DESIGN.md](DESIGN.md) for the full architecture, loss formulation, and experimental plan.