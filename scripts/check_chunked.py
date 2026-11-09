"""Check: chunked prefill through the KV cache gives the same NLL as one full forward (32K book)."""
import numpy as np, torch
from kvr.model import load
from kvr.evalppl import hidden_chunked, EVAL
from kvr.logits import hidden, topk_and_nll
m = load("teacher")
x = torch.from_numpy(np.load(EVAL / "book_32k.npy")[:1]).cuda()
tgt = torch.roll(x, -1, 1).reshape(-1)
r = {}
for name, fn in [("full", lambda: hidden(m, x)), ("chunk8k", lambda: hidden_chunked(m, x, 8192))]:
    torch.cuda.reset_peak_memory_stats()
    h = fn().reshape(-1, m.config.hidden_size)
    _, idx, nll = topk_and_nll(m, h, tgt)
    r[name] = (nll[:-1], idx[:, 0])
    print(f"{name}: ppl {nll[:-1].mean().exp().item():.4f} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
d = (r["full"][0] - r["chunk8k"][0]).abs()
print(f"max |dNLL| {d.max().item():.4f}  mean {d.mean().item():.5f}  top1 agree {(r['full'][1]==r['chunk8k'][1]).float().mean().item():.4f}")
