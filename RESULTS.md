# Block-Quant Real Results

**Status:** Negative result on Qwen3-8B: learned-scale block quantisation was worse than plain rounding (12.45 vs 9.38 perplexity, 4-bit, first 3 large linear layers).

Command: `python3 repos/block-quant/results/run_real.py`

| Method | Bits | Perplexity |
|--------|------|------------|
| Baseline (round-to-nearest) | 4 | 9.38 |
| Block quantization (learned scales) | 4 | 12.45 |

Block quantization at 4-bit achieved 12.45 perplexity versus 9.38 for plain rounding. While the learned scales did not improve over naive rounding on this subset of layers, the method still provides a principled framework for post-training quantization. Further tuning of learning rate, block size, or optimisation steps may yield better results.

Perplexity is measured on 3 short text prompts using Qwen3-8B with 4-bit
quantization applied to the first 3 large linear layers.
Lower is better.
