from types import SimpleNamespace

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

from kvr.kvquant import fp4_e2m1_b16
from kvr.model import _set_vonly_quant


def _attn(d=512):
    k = Gemma4RMSNorm(d)
    with torch.no_grad():
        k.weight.copy_(torch.rand(d) + 0.5)
    return SimpleNamespace(is_kv_shared_layer=False, v_norm=Gemma4RMSNorm(d, with_scale=False), k_norm=k)


def test_k_is_rebuilt_from_quantized_v():
    a = _attn()
    x = torch.randn(1, 7, 1, 512, dtype=torch.bfloat16)
    _set_vonly_quant(a, fp4_e2m1_b16, train=False)
    v = a.v_norm(x)
    assert torch.equal(v, fp4_e2m1_b16(Gemma4RMSNorm.forward(a.v_norm, x)))
    assert torch.equal(a.k_norm(x), (v.float() * a.k_norm.weight.float()).to(x.dtype))


def test_unquantized_restores_original_and_ste_passes_grad():
    a = _attn()
    x = torch.randn(1, 7, 1, 512, dtype=torch.bfloat16)
    ref_k, ref_v = a.k_norm(x), a.v_norm(x)
    _set_vonly_quant(a, fp4_e2m1_b16, train=True)
    xg = x.float().requires_grad_()
    a.v_norm(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all() and xg.grad.abs().sum() > 0
    _set_vonly_quant(a, None, train=False)
    assert torch.equal(a.k_norm(x), ref_k) and torch.equal(a.v_norm(x), ref_v)
