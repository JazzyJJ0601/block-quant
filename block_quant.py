import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughSTE(torch.autograd.Function):
    """Straight-through estimator (STE) for quantization.
    
    straight_through: The gradient passes through unchanged during backward pass.
    """
    
    @staticmethod
    def forward(ctx, x, quantized):
        ctx.save_for_backward(x, quantized)
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        # straight_through: pass gradient unchanged as if x was used
        return grad_output, None


class BlockQuantizer(nn.Module):
    """Block quantizer with learnable scales using straight-through estimator."""
    
    def __init__(self, bit_width: int, block_size: int):
        super().__init__()
        if bit_width not in (2, 4, 8):
            raise ValueError(f"bit_width must be 2, 4, or 8, got {bit_width}")
        self.bit_width = bit_width
        self.block_size = block_size
        
        # Number of quantization levels
        self.num_levels = 2 ** bit_width
        
        # Learnable scale factor per block (initialized to 1.0)
        self._scales = None
    
    @property
    def scales(self):
        return self._scales
    
    @scales.setter
    def scales(self, value):
        self._scales = value
    
    def _init_scales(self, num_blocks: int):
        """Initialize learnable scales."""
        if self._scales is None:
            self._scales = nn.Parameter(torch.ones(num_blocks))
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize and dequantize weight tensor using block quantization with STE.
        
        Args:
            x: Input weight tensor of shape (..., num_blocks, block_size)
        
        Returns:
            Tuple of (quantized/dequantized weights, scales)
        """
        # Ensure x has the right shape
        if x.dim() < 2:
            raise ValueError(f"x must be at least 2D, got {x.dim()}D")
        
        # Calculate number of blocks
        num_blocks = x.shape[-2]
        
        # Initialize scales if needed
        self._init_scales(num_blocks)
        
        # Expand scales to match input
        # Shape: (1, ..., 1, num_blocks) -> broadcast to (..., num_blocks, block_size)
        shape = [1] * (x.dim() - 1) + [num_blocks]
        scale_expanded = self.scales.view(shape)
        
        # Quantization range: [-1, 1] mapped to levels
        max_val = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        half_levels = (self.num_levels / 2) - 1
        
        # Quantize: round to nearest integer level
        # q_levels = round(x / (scale * half_levels)) * half_levels
        # Then map back to [-1, 1]
        quantized_levels = torch.round(x / (scale_expanded * half_levels))
        quantized_levels = torch.clamp(quantized_levels, -half_levels, half_levels)
        quantized = quantized_levels * (scale_expanded * half_levels) / half_levels
        
        # Apply straight-through estimator
        # Forward uses quantized, backward passes gradient as if using x
        dequantized = StraightThroughSTE.apply(x, quantized)
        
        return dequantized, self.scales


def quantize_block(x: torch.Tensor, bit_width: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for block quantization with STE.
    
    Args:
        x: Input weight tensor
        bit_width: Quantization bits (2, 4, or 8)
        block_size: Size of each quantization block
    
    Returns:
        Tuple of (quantized/dequantized weights, scales)
    """
    quantizer = BlockQuantizer(bit_width, block_size)
    return quantizer(x)
