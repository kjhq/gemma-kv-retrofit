"""Label training text with the teacher's top-32 next-token log-probs (resumable).

Packs FineWeb-Edu shard 000 (eval uses shard 013) into 8K sequences of bos+doc+eos, then writes
shards of 64 sequences to data/labels/<name>/shard_NNNNN.npz with:
  ids int32 (64, T), lp fp16 (64, T, 32), idx int32 (64, T, 32)
A crash loses at most one shard (~70 s). Rerun to resume.

  uv run python scripts/label.py --tokens 10_000_000
"""
import argparse, os, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from kvr.logits import hidden, topk_and_nll
from kvr.model import load, tokenizer

CACHE = Path.home() / ".cache/gemma-kv-retrofit"
SRC = CACHE / "data/raw/fineweb-edu/sample/10BT/000_00000.parquet"
PER_SHARD = 64

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=float, default=10e6)
ap.add_argument("--seq", type=int, default=8192)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--name", default="fwe_8k")
args = ap.parse_args()

out = CACHE / "data/labels" / args.name
out.mkdir(parents=True, exist_ok=True)
n_shards = -(-int(args.tokens) // (args.seq * PER_SHARD))
n_seq = n_shards * PER_SHARD


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# 1. deterministic packing (cached; extended if a later run asks for more tokens)
tok_file = out / "tokens.npy"
if not tok_file.exists() or np.load(tok_file, mmap_mode="r").shape[0] < n_seq:
    tok = tokenizer()
    need, ids = n_seq * args.seq, []
    pf = pq.ParquetFile(SRC)
    for rg in range(pf.num_row_groups):
        for enc in tok(pf.read_row_group(rg, columns=["text"]).column("text").to_pylist(), add_special_tokens=False)["input_ids"]:
            ids += [tok.bos_token_id, *enc, tok.eos_token_id]
        if len(ids) >= need:
            break
    assert len(ids) >= need, f"shard 000 has only {len(ids)} tokens"
    np.save(tok_file, np.asarray(ids[:need], np.int32).reshape(n_seq, args.seq))
    log(f"packed {n_seq} x {args.seq} = {need/1e6:.1f}M tokens")
tokens = np.load(tok_file, mmap_mode="r")

# 2. label shard by shard
todo = [s for s in range(n_shards) if not (out / f"shard_{s:05d}.npz").exists()]
log(f"{n_shards - len(todo)}/{n_shards} shards done, {len(todo)} to go")
if todo:
    model = load("teacher")
    for s in todo:
        t0 = time.time()
        x_all = torch.from_numpy(np.array(tokens[s * PER_SHARD : (s + 1) * PER_SHARD]))
        lps, idxs, nll_sum = [], [], 0.0
        for b in range(0, PER_SHARD, args.batch):
            x = x_all[b : b + args.batch].cuda()
            h = hidden(model, x).reshape(-1, model.config.hidden_size)
            lp, idx, nll = topk_and_nll(model, h, torch.roll(x, -1, 1).reshape(-1))
            lps.append(lp.view(*x.shape, -1).cpu()); idxs.append(idx.view(*x.shape, -1).cpu())
            nll_sum += nll.view(*x.shape)[:, :-1].sum().item()
        tmp = out / f"shard_{s:05d}.tmp.npz"
        np.savez(tmp, ids=x_all.numpy(), lp=torch.cat(lps).numpy(), idx=torch.cat(idxs).numpy())
        os.replace(tmp, out / f"shard_{s:05d}.npz")
        dt = time.time() - t0
        ppl = np.exp(nll_sum / (PER_SHARD * (args.seq - 1)))
        log(f"shard {s}: {PER_SHARD * args.seq / dt:,.0f} tok/s, teacher ppl {ppl:.2f}")
(out / "DONE").write_text(f"{n_shards} shards, {n_seq * args.seq} tokens\n")
log("labelling done")
