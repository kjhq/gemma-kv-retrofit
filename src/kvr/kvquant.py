"""Simulated ("fake") KV-cache quantization.

Each function quantizes then dequantizes a K or V tensor shaped (batch, kv_heads, seq, head_dim),
in blocks along head_dim only, so every token/head row is quantized independently. That makes
"quantize the whole cache at every step" identical to "quantize each token once when written",
which is what real runtimes do. Keys arrive here after RoPE, matching llama.cpp and DeepSeek V4.1.

The same functions are used later for quantization-aware training via a straight-through
estimator (forward = quantized, backward = identity).
"""
from __future__ import annotations

import torch

# E2M1 (FP4) representable magnitudes.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    *lead, d = x.shape
    assert d % block == 0, f"head_dim {d} not divisible by block {block}"
    return x.reshape(*lead, d // block, block)


def q8_0(x: torch.Tensor) -> torch.Tensor:
    """llama.cpp q8_0: blocks of 32, fp16 scale = absmax/127, int8 values."""
    xb = _blocks(x.float(), 32)
    d = (xb.abs().amax(-1, keepdim=True) / 127.0).to(torch.float16).float()
    q = torch.where(d > 0, torch.round(xb / d.clamp_min(1e-30)), 0.0).clamp(-127, 127)
    return (q * d).reshape(x.shape).to(x.dtype)


def q4_0(x: torch.Tensor) -> torch.Tensor:
    """llama.cpp q4_0: blocks of 32, d = (signed value with largest magnitude) / -8, 4-bit codes 0..15."""
    xb = _blocks(x.float(), 32)
    idx = xb.abs().argmax(-1, keepdim=True)
    mx = torch.gather(xb, -1, idx)
    d = (mx / -8.0).to(torch.float16).float()
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.trunc(xb * inv + 8.5), 0, 15)
    return ((q - 8.0) * d).reshape(x.shape).to(x.dtype)


def fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """vLLM-style fp8 KV with unit scale (no calibration)."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def fp4_e2m1_b16(x: torch.Tensor) -> torch.Tensor:
    """DeepSeek V4.1 main-KV format: E2M1 values, one E4M3 scale per 16 channels, no global scale."""
    if x.dim() >= 2 and x.shape[-2] > 16384:  # rows are independent: slice tokens to cap fp32 temporaries
        return torch.cat([fp4_e2m1_b16(c) for c in x.split(16384, dim=-2)], dim=-2)
    xb = _blocks(x.float(), 16)
    s = (xb.abs().amax(-1, keepdim=True) / 6.0).clamp_min(2.0**-9)
    s = s.to(torch.float8_e4m3fn).float()
    y = (xb / s).clamp(-6, 6)
    grid = _E2M1.to(y.device)
    # nearest grid value via midpoints (ties go to the lower value); no 8x temporary, which OOMed at 128K
    mag = grid[torch.bucketize(y.abs(), (grid[1:] + grid[:-1]) / 2)]
    return (torch.sign(y) * mag * s).reshape(x.shape).to(x.dtype)


FORMATS = {"bf16": None, "q8_0": q8_0, "q4_0": q4_0, "fp8": fp8_e4m3, "fp4b16": fp4_e2m1_b16}

# Effective bits per cached value, including block scales.
BITS = {"bf16": 16.0, "q8_0": 8.5, "q4_0": 4.5, "fp8": 8.0, "fp4b16": 4.5}


def ste(fn, x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: quantized forward, identity gradient."""
    return x + (fn(x) - x).detach()
