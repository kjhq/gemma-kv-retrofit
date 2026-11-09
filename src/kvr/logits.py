"""Chunked LM head: top-k log-probs, target NLL and KL-to-teacher without materialising full-vocab logits."""
from __future__ import annotations

import torch


def _logits(model, h: torch.Tensor) -> torch.Tensor:
    logits = (h @ model.lm_head.weight.T).float()
    cap = model.config.final_logit_softcapping
    if cap:
        logits.div_(cap).tanh_().mul_(cap)
    return logits


@torch.no_grad()
def hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Final normed hidden states (B, T, H)."""
    return model.model(input_ids=input_ids, use_cache=False).last_hidden_state


@torch.no_grad()
def topk_and_nll(model, h: torch.Tensor, targets: torch.Tensor | None, k: int = 32, chunk: int = 2048):
    """h: (N, H) flattened hidden states. Returns (topk_logprob fp16 (N,k), topk_idx int32 (N,k), nll fp32 (N,) or None)."""
    lps, idxs, nlls = [], [], []
    for s in range(0, h.shape[0], chunk):
        lg = _logits(model, h[s : s + chunk])
        lse = torch.logsumexp(lg, -1, keepdim=True)
        v, i = lg.topk(k, -1)
        lps.append((v - lse).to(torch.float16))
        idxs.append(i.to(torch.int32))
        if targets is not None:
            nlls.append((lse.squeeze(-1) - lg.gather(-1, targets[s : s + chunk, None].long()).squeeze(-1)))
        del lg
    return torch.cat(lps), torch.cat(idxs), (torch.cat(nlls) if targets is not None else None)


@torch.no_grad()
def nll_and_kl(model, h: torch.Tensor, targets: torch.Tensor, t_lp: torch.Tensor, t_idx: torch.Tensor, chunk: int = 2048):
    """Student NLL of targets and top-k KL(teacher || student) per position.

    Teacher mass outside its top-k is lumped into one "rest" bucket on both sides, so the KL is a
    lower bound on the full KL that is tight when the top-k holds almost all teacher mass.
    """
    nlls, kls = [], []
    for s in range(0, h.shape[0], chunk):
        lg = _logits(model, h[s : s + chunk])
        lse = torch.logsumexp(lg, -1, keepdim=True)
        s_lp_all = lg.sub_(lse)
        nlls.append(-s_lp_all.gather(-1, targets[s : s + chunk, None].long()).squeeze(-1))
        tl = t_lp[s : s + chunk].float()
        sl = s_lp_all.gather(-1, t_idx[s : s + chunk].long())
        tp = tl.exp()
        t_rest = (1 - tp.sum(-1)).clamp_min(1e-12)
        s_rest = (1 - sl.exp().sum(-1)).clamp_min(1e-12)
        kl = (tp * (tl - sl)).sum(-1) + t_rest * (t_rest.log() - s_rest.log())
        kls.append(kl)
        del lg, s_lp_all
    return torch.cat(nlls), torch.cat(kls)


def kl_loss_backward(model, h: torch.Tensor, t_lp: torch.Tensor, t_idx: torch.Tensor, scale: float, chunk: int = 512) -> float:
    """Training loss: top-k KL(teacher || student) + exact "rest" bucket, summed over positions times `scale`.

    h: (N, H) final hidden states with grad. Logits are built per chunk and backpropagated into h
    chunk by chunk, so the full (N, vocab) logits never exist. Returns the summed KL (float).
    """
    hd = h.detach().requires_grad_()
    cap = model.config.final_logit_softcapping
    W = model.lm_head.weight
    total = 0.0
    for s in range(0, h.shape[0], chunk):
        lg = (hd[s : s + chunk] @ W.T).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        lse = torch.logsumexp(lg, -1)
        idx = t_idx[s : s + chunk].long()
        sl = lg.gather(-1, idx) - lse[:, None]
        s_rest = torch.logsumexp(lg.scatter(-1, idx, float("-inf")), -1) - lse
        tl = t_lp[s : s + chunk].float()
        tp = tl.exp()
        t_rest = (1 - tp.sum(-1)).clamp_min(1e-6)
        kl = (tp * (tl - sl)).sum(-1) + t_rest * (t_rest.log() - s_rest)
        (kl.sum() * scale).backward()
        total += kl.sum().item()
        del lg, lse, sl, s_rest, kl
    h.backward(hd.grad)
    return total
