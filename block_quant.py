import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundSTE(torch.autograd.Function):
    """Straight-through estimator for rounding.

    Forward: round(x) to nearest integer.
    Backward: pass gradient through as if rounding was identity.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _to_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """Flatten x and split it into (num_blocks, block_size), zero-padding the tail."""
    x_flat = x.flatten()
    n = x_flat.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        x_flat = F.pad(x_flat, (0, pad))
    return x_flat.view(-1, block_size), n


def _qmax(bit_width: int) -> int:
    """Largest symmetric integer level, e.g. 7 for 4-bit (levels -7..7)."""
    return 2 ** (bit_width - 1) - 1


def fake_quant(blocks: torch.Tensor, scales: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Quantise each row of `blocks` to integers in [-qmax, qmax] with its own scale,
    then dequantise. `scales` has shape (num_blocks,)."""
    qmax = _qmax(bit_width)
    s = scales.view(-1, 1).clamp_min(1e-12)
    levels = torch.clamp(RoundSTE.apply(blocks / s), -qmax, qmax)
    return levels * s


class BlockQuantizer(nn.Module):
    """Block quantizer with learnable scales using straight-through estimator.

    Each block of `block_size` weights gets its own scale. A weight w in a block
    with scale s becomes clamp(round(w / s), -qmax, qmax) * s, so a 4-bit block
    really uses 15 levels. Scales start at max|w| / qmax (absmax) and can be
    refined by gradient descent through the STE.
    """

    def __init__(self, bit_width: int, block_size: int):
        super().__init__()
        if bit_width not in (2, 4, 8):
            raise ValueError(f"bit_width must be 2, 4, or 8, got {bit_width}")
        self.bit_width = bit_width
        self.block_size = block_size
        self.num_levels = 2 ** bit_width
        self._scales = None

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, value):
        self._scales = value

    def _init_scales(self, num_blocks: int, init_values: torch.Tensor = None):
        if self._scales is None or self._scales.numel() != num_blocks:
            if init_values is not None:
                self._scales = nn.Parameter(init_values.detach().clone().view(-1))
            else:
                self._scales = nn.Parameter(torch.ones(num_blocks))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and dequantize x block by block. Returns (dequantized, scales)."""
        blocks, n = _to_blocks(x, self.block_size)
        self._init_scales(blocks.shape[0], init_values=blocks.abs().max(dim=1).values / _qmax(self.bit_width))
        deq = fake_quant(blocks, self.scales, self.bit_width)
        return deq.reshape(-1)[:n].view(x.shape), self.scales


def search_scales(
    x: torch.Tensor,
    bit_width: int,
    block_size: int,
    weight: torch.Tensor = None,
    grid: int = 20,
    min_ratio: float = 0.5,
) -> torch.Tensor:
    """Pick each block's scale by grid search over clipping ratios.

    For every block, scale = ratio * max|w| / qmax for ratio in [min_ratio, 1].
    The ratio that minimises the block's squared error is kept. If `weight` is
    given (same shape as x, e.g. the mean squared input activation of each
    weight's input channel), the error is weighted by it, so blocks protect the
    weights that matter most for the layer's output rather than all equally.

    Returns the per-block scales (num_blocks,).
    """
    blocks, n = _to_blocks(x, block_size)
    if weight is not None:
        wblocks, _ = _to_blocks(weight.expand_as(x), block_size)
    absmax = blocks.abs().max(dim=1).values / _qmax(bit_width)
    best_err = torch.full_like(absmax, float("inf"))
    best_scale = absmax.clone()
    for ratio in torch.linspace(min_ratio, 1.0, grid).tolist():
        s = absmax * ratio
        err = (fake_quant(blocks, s, bit_width) - blocks) ** 2
        if weight is not None:
            err = err * wblocks
        err = err.sum(dim=1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, s, best_scale)
    return best_scale


def quantize_with_scales(x: torch.Tensor, scales: torch.Tensor, bit_width: int, block_size: int) -> torch.Tensor:
    """Dequantised copy of x using the given per-block scales."""
    blocks, n = _to_blocks(x, block_size)
    return fake_quant(blocks, scales, bit_width).reshape(-1)[:n].view(x.shape)


def quantize_block(x: torch.Tensor, bit_width: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for block quantization with STE."""
    quantizer = BlockQuantizer(bit_width, block_size)
    return quantizer(x)
