"""Stage 0: baselines. Resumable: each (config, eval) result is a JSON file; existing ones are skipped.

Configs:
  teacher                    original E4B, bf16 KV
  kvq_*                      original weights, simulated KV quantization (no training)
  var_*                      untrained student variants (shows how much healing each change needs)
"""
import gc, json, sys, time
from pathlib import Path

import torch

from kvr import evalppl, niah
from kvr.model import kv_bytes_per_token, load, set_kv_quant, tokenizer

CACHE = Path.home() / ".cache/gemma-kv-retrofit"
OUT = CACHE / "results/stage0"
OUT.mkdir(parents=True, exist_ok=True)
ALL = ["web_4k", "code_4k", "book_32k", "book_128k"]
CONFIGS = [  # tag, variant, full-layer KV fmt, sliding KV fmt, ppl sets, run niah
    ("teacher", "teacher", "bf16", "bf16", ALL, True),
    ("kvq_q8_0", "teacher", "q8_0", "q8_0", ALL, True),
    ("kvq_q4_0", "teacher", "q4_0", "q4_0", ALL, True),
    ("kvq_fp4full_q8slide", "teacher", "fp4b16", "q8_0", ALL, True),
    ("kvq_fp8", "teacher", "fp8", "fp8", ["web_4k", "book_32k"], False),
    ("var_g1", "g1", "bf16", "bf16", ["web_4k", "book_32k"], False),
    ("var_g1kv", "g1kv", "bf16", "bf16", ["web_4k", "book_32k"], False),
    ("var_s24", "s24", "bf16", "bf16", ["web_4k", "book_32k"], False),
    ("var_s30", "s30", "bf16", "bf16", ["web_4k", "book_32k"], False),
    ("var_s36", "s36", "bf16", "bf16", ["web_4k", "book_32k"], False),
]
only = set(sys.argv[1:])
tok = tokenizer()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


for tag, variant, full, slide, sets, do_niah in CONFIGS:
    if only and tag not in only:
        continue
    todo = [s for s in sets if not (OUT / f"{tag}__{s}.json").exists()] + (["niah"] if do_niah and not (OUT / f"{tag}__niah.json").exists() else [])
    if not todo:
        continue
    log(f"== {tag}: {todo}")
    model = load(variant)
    set_kv_quant(model, full=full, sliding=slide)
    meta = {"tag": tag, "variant": variant, "kv_full": full, "kv_sliding": slide,
            "kv_bytes_per_token": kv_bytes_per_token(model, full)}
    for s in todo:
        if s == "niah":
            r = niah.run(model, tok)
        else:
            if tag == "teacher":
                evalppl.teacher_labels(model, s)
            r = evalppl.evaluate(model, s)
        (OUT / f"{tag}__{s}.json").write_text(json.dumps({**meta, "eval": s, **r}, indent=1))
        log(f"{tag} {s}: " + json.dumps({k: v for k, v in r.items() if k != 'buckets'}))
    del model; gc.collect(); torch.cuda.empty_cache()
(CACHE / "stage0.DONE").write_text(time.strftime("%F %T"))
log("stage0 done")
