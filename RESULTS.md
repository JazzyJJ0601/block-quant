# Block-Quant: measured results

Qwen3-8B, 4-bit weights (levels -7..7), all 252 decoder linear layers, bf16 fake quantisation on an RTX 3090 Ti.
Perplexity on WikiText-2 test, 40 x 512 tokens. Calibration for the activation-weighted method: WikiText-2 train, 8 x 512 tokens.
Command: `python results/run_real.py` (writes `results/real.json`).

| Method | Perplexity |
|---|---|
| FP16 | 12.03 |
| Per-channel round-to-nearest | 16.45 |
| Block-64 absmax | 12.61 |
| Block-64 MSE clip search | 12.66 |
| Block-64 activation-weighted clip search | **12.26** |

Earlier results in this file (3 prompts, first 3 layers) came from a quantiser bug that used 3 levels instead of 15; they are withdrawn. See the README for details.
