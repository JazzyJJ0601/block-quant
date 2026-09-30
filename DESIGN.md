# Block-Quant: Learnable Block-Wise Quantization

## Abstract

Block-Quant introduces a post-training quantization scheme that improves weight precision by learning scale factors for fixed-size blocks. Unlike static min-max or block-wise quantization methods that rely on activation statistics from a calibration set, Block-Quant fine-tunes block-scale parameters using a straight-through estimator on a small set of activations. This document outlines the motivation, method, mathematical formulation, and experimental plan for this approach.

## Motivation

### Limitations of Static Quantization

Static quantization methods, including min-max normalization and block-wise scaling, determine quantization parameters (such as scale and zero-point) from activation statistics collected over a calibration dataset. These parameters are then fixed during inference. While effective at reducing memory and compute costs, static methods have several drawbacks:

1. **Outlier Sensitivity:** Extreme values in activation distributions can compress the representational range for the majority of values, leading to information loss.
2. **Suboptimal Fit:** Statistics computed offline may not generalize well to downstream tasks or specific layers.
3. **No Task Adaptation:** Static scales do not adapt to the objective of the downstream model (e.g., perplexity on a language modeling task).

### Why Learnable Scales?

Learnable quantization addresses these issues by treating quantization parameters as trainable variables. Instead of fixing scales post-calibration, we optimize them jointly with a downstream objective. This allows the quantization scheme to adapt to the actual distribution of activations encountered during the task, preserving more signal and reducing quantization noise.

### Block-Wise Approach

Learning a scale for every individual weight parameter is expensive and can lead to overfitting. Instead, Block-Quant groups weights into fixed-size blocks (e.g., 64 elements) and learns a single scale factor per block. This balances expressivity with parameter efficiency, adding only one learnable parameter per 64 weights (approximately 1.5% overhead in parameters).

## Method

### Block Structure

We partition the weight tensor into non-overlapping blocks of 64 elements. For a weight matrix $W \in \mathbb{R}^{m \times n}$, the number of blocks is $\lceil \frac{m \cdot n}{64} \rceil$. Each block $b$ has an associated learnable scale factor $s_b \in \mathbb{R}^+$.

### Quantization Function

For a weight $w$ belonging to block $b$, the quantized value $\hat{w}$ is computed as:

$$ \hat{w} = s_b \cdot \text{round}\left( \frac{w}{s_b} \right) $$

This is a symmetric 4-bit quantization scheme where the integer representation ranges from $-8$ to $7$. The scale $s_b$ determines the range $[-8s_b, 7s_b]$ for each block.

### Straight-Through Estimator (STE)

During backpropagation, the rounding operation is non-differentiable. To overcome this, we use the Straight-Through Estimator (STE), which passes the gradient through the rounding function as if it were an identity function:

$$ \frac{\partial \text{round}(x)}{\partial x} \approx 1 $$

In practice, during the forward pass, we round $w/s_b$ to the nearest integer. During the backward pass, we treat the rounding operation as identity, allowing gradients to flow to both the weights $w$ and the scales $s_b$.

### Loss Function

We minimize the reconstruction loss between the original weights and the quantized weights:

$$ \mathcal{L}_{\text{recon}} = \sum_{b} \sum_{w \in b} \| w - \hat{w} \|_2^2 $$

Optionally, we can include a task-specific loss (e.g., cross-entropy on a calibration set) to jointly optimize the scales for downstream performance.

## Mathematical Formulation

Let $W \in \mathbb{R}^{m \times n}$ be the weight matrix. We reshape $W$ into blocks of size $B=64$, denoted $W_b \in \mathbb{R}^{64}$. Let $s \in \mathbb{R}^{N_{\text{blocks}}}$ be the vector of learnable scale factors.

The quantized block is:

$$ \hat{W}_b = s_b \cdot \text{round}\left( \frac{W_b}{s_b} \right) $$

The total loss is:

$$ \mathcal{L}(s) = \sum_{b} \| W_b - \hat{W}_b \|_2^2 $$

The gradient with respect to $s_b$ is computed via STE:

$$ \frac{\partial \mathcal{L}}{\partial s_b} = \sum_{i=1}^{64} 2 (W_{b,i} - \hat{W}_{b,i}) \cdot \frac{\partial \hat{W}_{b,i}}{\partial s_b} $$

Where:

$$ \frac{\partial \hat{W}_{b,i}}{\partial s_b} = \text{round}\left( \frac{W_{b,i}}{s_b} \right) - \frac{W_{b,i}}{s_b} \cdot 1 $$

This gradient allows us to update $s_b$ to minimize the quantization error.

## Calibration Setup

To initialize and fine-tune the scales, we use a small calibration set of activations from the Qwen3-8B model. Specifically, we collect 20 activation samples for each layer. This dataset is sufficiently small to keep calibration fast (minutes on a single GPU) while providing enough statistics to avoid overfitting.

### Data Collection

1. Forward pass Qwen3-8B over 20 input sequences (e.g., from a validation subset of CommonCrawl or a curated text corpus).
2. Record activation tensors at target layers (e.g., output projection or specific transformer blocks).
3. Use these activations to initialize scale factors (e.g., using min-max statistics as starting points).

### Fine-tuning

After initialization, we perform a few steps of gradient descent to optimize the scales using the reconstruction loss or a task-specific objective. The optimization uses Adam with a learning rate of $10^{-3}$ for 100 iterations.

## Experimental Plan

### Baseline

We compare Block-Quant against a static 4-bit block-wise quantization baseline. In the baseline, scale factors are computed once using min-max statistics from the same calibration set and remain fixed during fine-tuning.

### Metrics

The primary evaluation metric is perplexity (PPL) on a held-out validation set. We also measure reconstruction error ($\ell_2$ distance between original and quantized weights) to isolate quantization quality from downstream task performance.

### Setup

- **Model:** One layer of Qwen3-8B (e.g., output projection layer).
- **Blocks:** 64-element blocks.
- **Calibration:** 20 activation samples.
- **Fine-tuning:** 100 steps with Adam ($\text{lr} = 10^{-3}$).
- **Baseline:** Static min-max quantization (4-bit, block-wise).

### Expected Results

We expect Block-Quant to achieve lower perplexity and reconstruction error than the static baseline. By adapting scales to the local weight distribution and downstream objective, the learnable approach should preserve more information per bit.

## Implementation Notes

- **Memory:** Store scale factors as 16-bit floats to minimize overhead.
- **Inference:** No additional computation is needed; quantization is offline.
- **Compatibility:** Works with existing PyTorch models by replacing weight tensors with quantized wrappers.

## Conclusion

Block-Quant offers a practical path to higher-fidelity quantization by combining block-wise efficiency with learnable parameters. Using a straight-through estimator, we can optimize scale factors for minimal error without requiring changes to model architecture. Future work will explore joint optimization across multiple layers and integration with activation quantization.

## References

- Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.
- Zhang, P., et al. (2023). QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.
