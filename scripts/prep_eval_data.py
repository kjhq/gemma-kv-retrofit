"""Build fixed, held-out eval token sets (never used for training).

web_4k   FineWeb-Edu shard 013 (training uses other shards), 256 x 4096
code_4k  codeparrot-clean-valid (Python), 128 x 4096
book_32k PG19 test books, 32 x 32768 (one window per book, from the start)
book_128k PG19 test books, 8 x 131072 (only books that long)
"""
import gzip, json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kvr.model import tokenizer

R = Path.home() / ".cache/gemma-kv-retrofit/data/raw"
OUT = Path.home() / ".cache/gemma-kv-retrofit/data/eval"
OUT.mkdir(parents=True, exist_ok=True)
tok = tokenizer()
BOS, EOS = tok.bos_token_id, tok.eos_token_id


def pack(texts, n, L):
    ids = []
    for enc in tok(texts, add_special_tokens=False)["input_ids"]:
        ids += [BOS, *enc, EOS]
        if len(ids) >= n * L:
            break
    assert len(ids) >= n * L, f"only {len(ids)} tokens"
    return np.asarray(ids[: n * L], np.int32).reshape(n, L)


def save(name, arr):
    np.save(OUT / f"{name}.npy", arr)
    print(f"{name}: {arr.shape} = {arr.size/1e6:.2f}M tokens")


web = pq.read_table(R / "fineweb-edu/sample/10BT/013_00000.parquet", columns=["text"]).column("text")
save("web_4k", pack(web.slice(0, 20000).to_pylist(), 256, 4096))

code_file = next((R / "codeparrot-valid").glob("*.json.gz"))
code = []
with gzip.open(code_file, "rt") as f:
    for line in f:
        code.append(json.loads(line)["content"])
        if len(code) >= 20000:
            break
save("code_4k", pack(code, 128, 4096))

books = pq.read_table(next((R / "pg19/data").glob("test-*.parquet")), columns=["text"]).column("text").to_pylist()
enc = [[BOS, *e] for e in tok(books, add_special_tokens=False)["input_ids"]]
lens = sorted((len(e) for e in enc), reverse=True)
print("pg19 test books:", len(enc), "longest", lens[:10])
b32 = [e[:32768] for e in enc if len(e) >= 32768][:32]
b128 = [e[:131072] for e in enc if len(e) >= 131072][:8]
save("book_32k", np.asarray(b32, np.int32))
if len(b128) < 8:  # top up from validation books if test has too few long ones
    vb = pq.read_table(next((R / "pg19/data").glob("validation-*.parquet")), columns=["text"]).column("text").to_pylist()
    b128 += [[BOS, *e][:131072] for e in tok(vb, add_special_tokens=False)["input_ids"] if len(e) + 1 >= 131072][: 8 - len(b128)]
save("book_128k", np.asarray(b128, np.int32))
