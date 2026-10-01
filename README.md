# Block-Quant

**Learnable block-wise quantization with straight-through estimator.**

Post-training quantization that optimises per-block scale factors instead of fixing them from calibration statistics. The scale factors are learned via gradient descent, minimising reconstruction error through a straight-through estimator (STE).

## The Idea

Standard block-wise quantization computes scale factors once from min/max statistics. Block-Quant treats those scales as trainable parameters — they start from the weight range but are refined by a few gradient steps to minimise `||W - Q(W)||_2`. Because each block gets its own optimised scale, the method adapts to local weight structure without adding inference cost.

## Implementation

- **BlockQuantizer** — PyTorch module with learnable `nn.Parameter` scales per block
- **Straight-through estimator** — gradient passes through the rounding step as identity
- **Supports 2, 4, and 8 bit** symmetric quantization
- **Data-driven scale initialisation** — scales start from `max_abs / half_levels` per block
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

## Benchmark results (Qwen3-8B, layer 10, mlp.up_proj weight)

Reconstruction error (Frobenius norm, relative to original) on a `[12288, 4096]` weight tensor:

| Bits | Static (min-max init) | Learned (100 steps STE) | Improvement |
|------|----------------------|------------------------|-------------|
| 2    | 0.723                | **0.445**              | **+38.5%**  |
| 4    | 0.883                | **0.855**              | **+3.2%**   |
| 8    | 0.993                | 0.994                   | ~0%         |

Block size 64, Adam lr=1e-3, 100 steps.

The biggest gains come at 2-bit, where per-block scale tuning recovers structure that aggressive quantization destroys. At higher bit-widths the initial data-driven scales are already close to optimal.

**Measured status:** Negative result on Qwen3-8B: learned-scale block quantisation was worse than plain rounding (12.45 vs 9.38 perplexity, 4-bit, first 3 large linear layers).

See [RESULTS.md](RESULTS.md) for real perplexity comparisons.

```bash
# Run the benchmark yourself
python benchmark.py
```

## Design

See [DESIGN.md](DESIGN.md) for the full architecture, loss formulation, and experimental plan.

## Tests

```bash
python -m pytest tests/
```