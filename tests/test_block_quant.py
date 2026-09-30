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


def test_forward_any_shape():
    """Test that forward works with any tensor shape and preserves shape."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    shapes = [
        (64,),          # 1D
        (32, 64),       # 2D matrix
        (16, 32, 64),   # 3D batch
        (12288, 4096),  # large 2D (like real weights)
    ]
    for s in shapes:
        x = torch.randn(s)
        out, scales = q(x)
        assert out.shape == s, f"Expected {s}, got {out.shape}"
        assert scales.numel() == (x.numel() + 7) // 8  # ceiling division


def test_quantizer_2bit():
    """Test 2-bit quantization preserves shape."""
    q = BlockQuantizer(bit_width=2, block_size=8)
    x = torch.randn(5, 16, 8)
    out, scales = q(x)
    assert out.shape == x.shape


def test_quantizer_8bit():
    """Test 8-bit quantization preserves shape."""
    q = BlockQuantizer(bit_width=8, block_size=8)
    x = torch.randn(5, 16, 8)
    out, scales = q(x)
    assert out.shape == x.shape


def test_backward_pass_stm():
    """Test that gradient flows through quantizer."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    x = torch.randn(64, requires_grad=True)
    out, scales = q(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_scales_are_parameter():
    """Test that scales are learnable parameters."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    x = torch.randn(64)
    out, scales = q(x)
    assert isinstance(scales, torch.nn.Parameter)
    assert scales.requires_grad


def test_scales_trained():
    """Test that scales actually move during training."""
    q = BlockQuantizer(bit_width=4, block_size=8)
    x = torch.randn(128)
    scales_before = q._scales.clone().detach() if q._scales is not None else 1

    # Train for a few steps
    opt = torch.optim.Adam([q._scales], lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        out, _ = q(x)
        loss = torch.nn.functional.mse_loss(out, x)
        loss.backward()
        opt.step()

    scales_after = q._scales
    assert not torch.allclose(scales_before, scales_after), "Scales should have changed"


def test_quantize_block_function():
    """Test the convenience quantize_block function."""
    x = torch.randn(5, 16, 8)
    out, scales = quantize_block(x, bit_width=4, block_size=8)
    assert out.shape == x.shape