import torch

from kvr.kvquant import _E2M1, fp4_e2m1_b16


def _fp4_reference(x):
    """Original argmin version (8x temporary). Kept as the spec for the nearest-value lookup."""
    xb = x.float().reshape(*x.shape[:-1], -1, 16)
    s = (xb.abs().amax(-1, keepdim=True) / 6.0).clamp_min(2.0**-9).to(torch.float8_e4m3fn).float()
    y = (xb / s).clamp(-6, 6)
    grid = _E2M1.to(y.device)
    mag = grid[(y.abs().unsqueeze(-1) - grid).abs().argmin(-1)]
    return (torch.sign(y) * mag * s).reshape(x.shape).to(x.dtype)


def test_fp4_matches_argmin_reference():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(2, 2, 257, 512, generator=g) * 3).to(torch.bfloat16)
    assert torch.equal(fp4_e2m1_b16(x), _fp4_reference(x))


def test_fp4_token_slicing_is_exact():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 1, 16384 * 2 + 5, 32, generator=g).to(torch.bfloat16)
    assert torch.equal(fp4_e2m1_b16(x), _fp4_reference(x))


def test_fp4_ties_round_down_like_reference():
    # each block: max 6 -> scale 1, then exact midpoints between grid values
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    row = torch.cat([torch.tensor([6.0]), mids, -mids, torch.tensor([0.0])])  # 16 values
    x = row.repeat(32).reshape(1, 1, 1, 512)
    assert torch.equal(fp4_e2m1_b16(x), _fp4_reference(x))
    assert fp4_e2m1_b16(x)[0, 0, 0, 1] == 0.0  # 0.25 -> 0
