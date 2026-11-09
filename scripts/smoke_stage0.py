import torch, numpy as np
from kvr.kvquant import FORMATS
from kvr.model import load, set_kv_quant, kv_bytes_per_token, tokenizer
from kvr import niah
from kvr.evalppl import EVAL
from kvr.logits import hidden, topk_and_nll
x = torch.randn(2, 2, 64, 512, device="cuda", dtype=torch.bfloat16) * 3
for n, f in FORMATS.items():
    if f is None: continue
    y = f(x); rel = ((y.float()-x.float()).norm()/x.float().norm()).item()
    print(f"{n:8s} rel_err {rel:.4f} idempotent {torch.equal(f(y), y)}")
tok = tokenizer()
ids = torch.from_numpy(np.load(EVAL / "web_4k.npy")[:2]).cuda()
tgt = torch.roll(ids, -1, 1).reshape(-1)
for v in ["teacher", "g1", "g1kv", "s30"]:
    m = load(v)
    for fmt in (["bf16", "q4_0"] if v == "teacher" else ["bf16"]):
        set_kv_quant(m, full=fmt, sliding=fmt)
        _, _, nll = topk_and_nll(m, hidden(m, ids).reshape(-1, m.config.hidden_size), tgt)
        print(f"{v:8s} kv={fmt:5s} bytes/tok={kv_bytes_per_token(m, fmt):7.0f} ppl={nll.view(2,-1)[:, :-1].mean().exp().item():.3f}")
    if v == "teacher":
        set_kv_quant(m)
        print("niah", niah.run(m, tok, lengths=[4096, 32768], depths=[0.1, 0.5, 0.9]))
    del m; torch.cuda.empty_cache()
