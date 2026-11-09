"""Perplexity and KL-to-teacher on the fixed eval sets, with position buckets for long books."""
from __future__ import annotations

import json, math, time
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from .logits import hidden, nll_and_kl, topk_and_nll

CACHE = Path.home() / ".cache/gemma-kv-retrofit"
EVAL = CACHE / "data/eval"
SETS = {  # name: (batch size, prefill chunk or None, position bucket edges)
    "web_4k": (8, None, [0, 4096]),
    "code_4k": (8, None, [0, 4096]),
    "book_32k": (1, None, [0, 4096, 16384, 32768]),
    "book_128k": (1, 4096, [0, 4096, 32768, 65536, 131072]),
}


@torch.no_grad()
def hidden_chunked(model, ids: torch.Tensor, chunk: int) -> torch.Tensor:
    """Prefill a long sequence in chunks through the KV cache; returns (B, T, H) final hidden states."""
    cache = DynamicCache(config=model.config)
    outs = []
    for s in range(0, ids.shape[1], chunk):
        o = model.model(input_ids=ids[:, s : s + chunk], past_key_values=cache, use_cache=True)
        outs.append(o.last_hidden_state)
    return torch.cat(outs, 1)


def _forward(model, x, chunk):
    return hidden_chunked(model, x, chunk) if chunk else hidden(model, x)


def teacher_labels(model, name: str, k: int = 32) -> Path:
    """Cache the teacher's top-k log-probs for an eval set (used to score students' KL)."""
    out = EVAL / f"teacher_{name}.npz"
    if out.exists():
        return out
    arr = np.load(EVAL / f"{name}.npy")
    B, chunk, _ = SETS[name]
    lps, idxs = [], []
    for s in range(0, len(arr), B):
        x = torch.from_numpy(arr[s : s + B]).cuda()
        h = _forward(model, x, chunk).reshape(-1, model.config.hidden_size)
        lp, idx, _ = topk_and_nll(model, h, None, k)
        lps.append(lp.cpu().numpy()); idxs.append(idx.cpu().numpy())
    np.savez(out, lp=np.concatenate(lps), idx=np.concatenate(idxs))
    return out


def evaluate(model, name: str, limit: int | None = None) -> dict:
    """limit: only the first `limit` sequences (quick checks during training)."""
    arr = np.load(EVAL / f"{name}.npy")[:limit]
    B, chunk, edges = SETS[name]
    T = arr.shape[1]
    tl = np.load(EVAL / f"teacher_{name}.npz")
    t_lp, t_idx = torch.from_numpy(tl["lp"]), torch.from_numpy(tl["idx"])
    nll_sum = np.zeros(len(edges) - 1); kl_sum = np.zeros(len(edges) - 1); cnt = np.zeros(len(edges) - 1)
    t0 = time.time()
    for s in range(0, len(arr), B):
        x = torch.from_numpy(arr[s : s + B]).cuda()
        b = x.shape[0]
        h = _forward(model, x, chunk).reshape(-1, model.config.hidden_size)
        tgt = torch.roll(x, -1, 1).reshape(-1)
        rows = slice(s * T, (s + b) * T)
        nll, kl = nll_and_kl(model, h, tgt, t_lp[rows].cuda(), t_idx[rows].cuda())
        nll, kl = nll.view(b, T)[:, :-1].cpu().numpy(), kl.view(b, T)[:, :-1].cpu().numpy()  # last pos has no target
        for j in range(len(edges) - 1):
            a, e = edges[j], min(edges[j + 1], T - 1)
            nll_sum[j] += nll[:, a:e].sum(); kl_sum[j] += kl[:, a:e].sum(); cnt[j] += nll[:, a:e].size
    res = {
        "ppl": math.exp(nll_sum.sum() / cnt.sum()),
        "kl": float(kl_sum.sum() / cnt.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    if len(edges) > 2:
        res["buckets"] = {f"{edges[j]//1024}-{edges[j+1]//1024}K": {"ppl": math.exp(nll_sum[j] / cnt[j]), "kl": float(kl_sum[j] / cnt[j])}
                          for j in range(len(edges) - 1)}
    return res


def save(result: dict, tag: str) -> None:
    d = CACHE / "results/ppl"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.json").write_text(json.dumps(result, indent=1))
