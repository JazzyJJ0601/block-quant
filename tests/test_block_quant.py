import torch
from block_quant import BlockQuantizer, quantize_block


def test_quantizer_initialization():
    """Test that quantizer initializes correctly."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    assert q.bit_width == 4
    assert q.block_size == 8
    assert q.num_levels == 16


def test_quantizer_invalid_bit_width():
    """Test that invalid bit width raises error."""
    try:
        BlockQuantizer(bit_width=3, block_size=8)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_forward_pass_shape():
    """Test that forward pass preserves shape."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    x = torch.randn(10, 32, 8)  # batch=10, 32 blocks, block_size=8
    out, scales = q(x)
    assert out.shape == x.shape
    assert scales.shape == (32,)


def test_quantizer_2bit():
    """Test 2-bit quantization."""
    q = BlockQuantizer(bit_width=2, block_size=8)
    x = torch.randn(5, 16, 8)
    out, scales = q(x)
    assert out.shape == x.shape
    assert scales.shape == (16,)


def test_quantizer_8bit():
    """Test 8-bit quantization."""
    q = BlockQuantizer(bit_width=8, block_size=8)
    x = torch.randn(5, 16, 8)
    out, scales = q(x)
    assert out.shape == x.shape
    assert scales.shape == (16,)


def test_backward_pass_stm():
    """Test that gradient flows through quantizer."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    x = torch.randn(5, 16, 8, requires_grad=True)
    out, scales = q(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_scales_are_parameter():
    """Test that scales are learnable parameters."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    x = torch.randn(5, 16, 8)
    out, scales = q(x)
    # Check that scales is a Parameter
    assert isinstance(scales, torch.nn.Parameter)
    # Check that scales requires grad
    assert scales.requires_grad


def test_quantize_block_function():
    """Test the convenience quantize_block function."""
    x = torch.randn(5, 16, 8)
    out, scales = quantize_block(x, bit_width=4, block_size=8)
    assert out.shape == x.shape
    assert scales.shape == (16,)
