# Block-Quant

**4-bit block-wise weight quantisation where each 64-weight block picks its own clipping scale, weighted by how much the layer's inputs actually use those weights.**

Measured on Qwen3-8B with every decoder linear layer (252 of them) quantised to 4 bits: activation-weighted block scales reach **12.26** perplexity on WikiText-2, against **12.03** for the unquantised model and **16.45** for the standard per-channel round-to-nearest baseline.

## Results (Qwen3-8B, 4-bit weights, all 252 decoder linear layers)

| Method | Scale per | How the scale is chosen | WikiText-2 perplexity |
|---|---|---|---|
| FP16 (no quantisation) | n/a | n/a | **12.03** |
| Round-to-nearest | output channel | max \|w\| / 7 | 16.45 |
| Block absmax | 64 weights | max \|w\| / 7 | 12.61 |
| Block MSE search | 64 weights | clip ratio minimising weight error | 12.66 |
| **Block activation-weighted search** | 64 weights | clip ratio minimising activation-weighted error | **12.26** |

Activation-weighted search closes about 60% of the gap between plain block quantisation and FP16 (0.58 → 0.22 perplexity points), at no inference cost: the output format is identical to block absmax, only the chosen scales differ.

The middle row matters too. Minimising plain weight error made things slightly *worse* than not searching at all. Weight error treats every weight as equally important; the layer's output does not. Weighting each weight's error by the mean squared activation of its input channel (8 × 512 tokens of WikiText-2 *train*) is what makes the search pay off.

Setup: WikiText-2 *test*, 40 non-overlapping windows of 512 tokens (20,480 tokens), bf16 on one RTX 3090 Ti, symmetric levels −7..7, one run with a fixed seed. Full numbers: [`results/real.json`](results/real.json).

## How it works

For each weight matrix, split every row into blocks of 64 consecutive weights (they share input channels, so they see the same activations). For each block, try 20 clipping ratios r from 0.5 to 1.0, with scale `s = r · max|w| / 7`, and keep the one that minimises

```
Σ_j  E[x_j²] · (w_j − s · clamp(round(w_j / s), −7, 7))²
```

where `E[x_j²]` is the mean squared input activation for the weight's input channel, collected with forward hooks on a few calibration sequences. Weighting quantisation error by input activation statistics is the same insight behind AWQ and GPTQ; here it is applied to the choice of a per-block clipping scale, which needs no extra stored parameters and no per-channel rescaling of the model.

## Honest notes

- **This is fake quantisation**: weights are quantised and dequantised back to bf16 to measure accuracy. Memory savings and speed of a packed 4-bit kernel are not measured here.
- **An earlier version of this repo was wrong.** Its quantiser divided by `scale × 7` but multiplied back by `scale` only, so "4-bit" blocks used just 3 levels and every weight shrank 7×. That produced the old negative result (12.45 vs 9.38 on 3 short prompts) and the old reconstruction table. Both were artefacts of the bug and have been removed. A regression test now checks that a 4-bit block uses more than 8 levels and keeps the weights' magnitude.
- The gradient-descent (straight-through estimator) path in `BlockQuantizer` is kept, but the measured results above use the grid search, which is deterministic and needs no learning rate.
- 20k evaluation tokens and a single seed: differences of a few hundredths are noise; the 12.26 vs 12.61 vs 16.45 gaps are not.

## Usage

```python
from block_quant import search_scales, quantize_with_scales

# act: mean squared input activation per input channel, shape (in_features,)
scales = search_scales(W, bit_width=4, block_size=64, weight=act.view(1, -1))
W_q = quantize_with_scales(W, scales, bit_width=4, block_size=64)
```

Reproduce the table (needs a local Qwen3-8B and about 17 GB of GPU memory):

```bash
python results/run_real.py
```

## Tests

```bash
python -m pytest tests/
```
