"""Needle-in-a-haystack recall for a base model (completion style, greedy decode).

single : one "special magic number" hidden in book text, asked at the end
multi4 : four needles with different keys, asked for one of them
Lengths 4K..128K, needle depth 5..95%. Score = fraction of answers containing the exact number.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import DynamicCache

from .logits import _logits

R = Path.home() / ".cache/gemma-kv-retrofit/data/raw"
LENGTHS = [4096, 16384, 32768, 65536, 126976]
DEPTHS = [0.05, 0.25, 0.5, 0.75, 0.95]
KEYS = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Hobart", "Recife", "Tromso"]


def _haystack_ids(tok, n: int, seed: int) -> list[int]:
    books = pq.read_table(next((R / "pg19/data").glob("validation-*.parquet")), columns=["text"]).column("text").to_pylist()
    rng = random.Random(seed)
    ids: list[int] = []
    while len(ids) < n:
        ids += tok(books[rng.randrange(len(books))][:600_000], add_special_tokens=False)["input_ids"]
    return ids[:n]


@torch.no_grad()
def greedy(model, ids: torch.Tensor, n_new: int = 12, chunk: int = 4096) -> list[int]:
    cache = DynamicCache(config=model.config)
    for s in range(0, ids.shape[1], chunk):
        o = model.model(input_ids=ids[:, s : s + chunk], past_key_values=cache, use_cache=True)
    h, out = o.last_hidden_state[:, -1], []
    for _ in range(n_new):
        nxt = _logits(model, h).argmax(-1)
        out.append(int(nxt))
        h = model.model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
    return out


def run(model, tok, lengths=LENGTHS, depths=DEPTHS, seed: int = 0) -> dict:
    rng = random.Random(seed)
    hay = _haystack_ids(tok, max(lengths), seed)
    res = {}
    for L in lengths:
        for task in ("single", "multi4"):
            hits = 0
            for d in depths:
                keys = rng.sample(KEYS, 4 if task == "multi4" else 1)
                nums = [str(rng.randrange(1_000_000, 9_999_999)) for _ in keys]
                needles = [tok(f" The special magic number for {k} is {n}. ", add_special_tokens=False)["input_ids"] for k, n in zip(keys, nums)]
                ask = rng.randrange(len(keys))
                q = tok(f"\n\nQuestion: What is the special magic number for {keys[ask]}?\nAnswer: The special magic number for {keys[ask]} is",
                        add_special_tokens=False)["input_ids"]
                body = hay[: L - 1 - len(q) - sum(map(len, needles))]
                # place the asked needle at depth d, others spread evenly
                spots = sorted([(d if i == ask else (i + 0.5) / len(keys)) for i in range(len(keys))])
                order = sorted(range(len(keys)), key=lambda i: d if i == ask else (i + 0.5) / len(keys))
                seq, prev = [tok.bos_token_id], 0
                for pos, i in zip(spots, order):
                    cut = int(pos * len(body))
                    seq += body[prev:cut] + needles[i]
                    prev = cut
                seq += body[prev:] + q
                out = tok.decode(greedy(model, torch.tensor([seq], device="cuda")))
                hits += nums[ask] in out.replace(",", "")
            res[f"{task}@{L//1024}K"] = hits / len(depths)
    res["mean"] = float(np.mean([v for k, v in res.items()]))
    return res
