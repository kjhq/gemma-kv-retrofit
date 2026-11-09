import torch
from torch import nn

from kvr.lora import LoRALinear


def test_lora_starts_as_noop_and_merge_is_exact():
    torch.manual_seed(0)
    base = nn.Linear(64, 48, bias=False)
    x = torch.randn(5, 64)
    ref = base(x).detach()
    m = LoRALinear(base, r=8, alpha=8.0)
    assert torch.equal(m(x), ref)
    with torch.no_grad():
        m.B.normal_()
    y = m(x).detach()
    assert not torch.allclose(y, ref)
    merged = m.merged()
    assert torch.allclose(merged(x), y, atol=1e-5)


def test_lora_grads_reach_A_and_B_only():
    base = nn.Linear(16, 16, bias=False)
    base.weight.requires_grad_(False)
    m = LoRALinear(base, r=4, alpha=4.0)
    m(torch.randn(3, 16)).sum().backward()
    assert m.B.grad is not None and m.B.grad.abs().sum() > 0
    assert m.A.grad is not None  # zero at init because B = 0
    assert base.weight.grad is None
