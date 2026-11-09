"""Labelling speed test, transformers engine.

Packs ~1M FineWeb-Edu tokens (shard 000) into 8K chunks, saves them so the vLLM test uses the
exact same input, then times teacher forward + top-32 log-probs + copy to CPU at a few batch sizes.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from kvr.logits import hidden, topk_and_nll
from kvr.model import load, tokenizer

CACHE = Path.home() / ".cache/gemma-kv-retrofit"
SRC = CACHE / "data/raw/fineweb-edu/sample/10BT/000_00000.parquet"
OUT = CACHE / "data/bench_8k.npy"

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=8192)
ap.add_argument("--chunks", type=int, default=128)
ap.add_argument("--batches", default="1,2,4")
ap.add_argument("--compile", action="store_true")
args = ap.parse_args()

tok = tokenizer()
if not OUT.exists():
    need = args.seq * args.chunks
    ids: list[int] = []
    pf = pq.ParquetFile(SRC)
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for enc in tok(texts, add_special_tokens=False)["input_ids"]:
            ids += [tok.bos_token_id, *enc, tok.eos_token_id]
        if len(ids) >= need:
            break
    arr = np.asarray(ids[:need], dtype=np.int32).reshape(args.chunks, args.seq)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, arr)
arr = np.load(OUT)
print(f"input {arr.shape} = {arr.size/1e6:.2f}M tokens")

model = load("teacher")
if args.compile:
    model.model = torch.compile(model.model)
data = torch.from_numpy(arr)
res = {}
for B in map(int, args.batches.split(",")):
    torch.cuda.reset_peak_memory_stats()
    n_tok = 0
    for step, s in enumerate(range(0, data.shape[0], B)):
        if step == 1:  # first batch = warmup
            torch.cuda.synchronize(); t0 = time.time(); n_tok = 0
        x = data[s : s + B].cuda(non_blocking=True)
        h = hidden(model, x).reshape(-1, model.config.hidden_size)
        tgt = torch.roll(x, -1, 1).reshape(-1)
        lp, idx, nll = topk_and_nll(model, h, tgt)
        lp, idx = lp.cpu(), idx.cpu()
        n_tok += x.numel()
    torch.cuda.synchronize()
    dt = time.time() - t0
    res[B] = {"tok_per_s": n_tok / dt, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    print(f"B={B}: {n_tok/dt:,.0f} tok/s  peak {res[B]['peak_gib']:.1f} GiB  (ppl check {nll.mean().exp().item():.2f})")
(CACHE / "results").mkdir(exist_ok=True)
(CACHE / f"results/bench_label_hf{'_compile' if args.compile else ''}.json").write_text(json.dumps(res, indent=1))
