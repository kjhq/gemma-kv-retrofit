from types import SimpleNamespace

import torch

from kvr.logits import kl_loss_backward, nll_and_kl


def _fake_model(V=300, H=16, cap=30.0):
    g = torch.Generator().manual_seed(0)
    return SimpleNamespace(lm_head=SimpleNamespace(weight=torch.randn(V, H, generator=g) * 0.5),
                           config=SimpleNamespace(final_logit_softcapping=cap))


def _teacher(m, N, k=32):
    g = torch.Generator().manual_seed(1)
    lg = torch.randn(N, m.lm_head.weight.shape[0], generator=g) * 3
    lp = lg.log_softmax(-1)
    v, i = lp.topk(k, -1)
    return v.half(), i.int()


def test_kl_value_matches_eval_metric_and_grad_matches_unchunked():
    m = _fake_model()
    N = 50
    h0 = torch.randn(N, 16)
    t_lp, t_idx = _teacher(m, N)

    h = h0.clone().requires_grad_()
    total = kl_loss_backward(m, h, t_lp, t_idx, scale=1.0, chunk=16)  # several chunks

    _, kl_eval = nll_and_kl(m, h0, torch.zeros(N, dtype=torch.long), t_lp, t_idx, chunk=16)
    assert abs(total - kl_eval.sum().item()) < 1e-3 * max(1.0, abs(total))

    # unchunked reference with full autograd
    h2 = h0.clone().requires_grad_()
    lg = torch.tanh((h2 @ m.lm_head.weight.T) / 30.0) * 30.0
    slp = lg.log_softmax(-1)
    sl = slp.gather(-1, t_idx.long())
    tl = t_lp.float(); tp = tl.exp()
    t_rest = (1 - tp.sum(-1)).clamp_min(1e-6)
    s_rest = torch.log1p(-sl.exp().sum(-1).clamp(max=1 - 1e-7))
    ((tp * (tl - sl)).sum(-1) + t_rest * (t_rest.log() - s_rest)).sum().backward()
    assert torch.allclose(h.grad, h2.grad, rtol=1e-3, atol=1e-5)
