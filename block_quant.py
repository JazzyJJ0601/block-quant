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


class BlockQuantizer(nn.Module):
    """Block quantizer with learnable scales using straight-through estimator.

    Each block of `block_size` weights gets its own learnable scale factor.
    Scales are optimised via gradient descent to minimise reconstruction error.
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
        """
        Quantize and dequantize weight tensor using block quantization with STE.

        Tensor is flattened, split into blocks of block_size, each block gets
        a learnable scale. The STE wraps only the rounding step so gradients
        flow through to both the input and the scale factors.

        Returns (dequantized, scales).
        """
        orig_shape = x.shape
        x_flat = x.flatten()
        n = x_flat.numel()

        # Pad to multiple of block_size
        pad = (self.block_size - n % self.block_size) % self.block_size
        if pad:
            x_padded = F.pad(x_flat, (0, pad))
        else:
            x_padded = x_flat

        # Reshape into blocks: (num_blocks, block_size)
        num_blocks = x_padded.numel() // self.block_size
        blocks = x_padded.view(num_blocks, self.block_size)

        # Quantization range: symmetric [-1, 1], mapped to N levels
        half_levels = (self.num_levels / 2) - 1  # e.g. 7 for 4-bit (levels 0-15 centered at 0)

        # Initialise scales based on data range (per-block max_abs / half_levels)
        self._init_scales(num_blocks, init_values=blocks.abs().max(dim=1).values / half_levels)

        # Scales: (num_blocks,) -> (num_blocks, 1) for broadcasting over block_size
        scale_expanded = self.scales.view(num_blocks, 1)

        # Normalise to [-1, 1], round via STE, then denormalise by scale
        # Forward: quantized = round(blocks / (scale * half_levels)) * scale
        # Backward: STE through rounding means grad flows to both blocks and scales
        normalised = blocks / (scale_expanded * half_levels)
        quantized_levels = RoundSTE.apply(normalised)
        quantized_levels = torch.clamp(quantized_levels, -half_levels, half_levels)

        # Multiply by scale to get dequantized values
        dequantized = quantized_levels * scale_expanded

        # Reshape back to original, removing padding
        result = dequantized.view(-1)[:n].view(orig_shape)
        return result, self.scales


def quantize_block(x: torch.Tensor, bit_width: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function for block quantization with STE."""
    quantizer = BlockQuantizer(bit_width, block_size)
    return quantizer(x)