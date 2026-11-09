# Stage 0 results (2026-11-04, no training)

Model: `google/gemma-4-E4B` base, transformers 5.22.1, torch 2.11.
Raw JSON: `~/.cache/gemma-kv-retrofit/results/stage0/`. Script: `scripts/run_stage0.py`.

- **ppl**: perplexity on held-out text; lower is better.
- **KL**: mean per-token KL(teacher ‖ model) over the teacher's top-32 tokens plus a "rest" bucket; 0 = identical predictions.
- **NIAH**: needle recall, single and 4-needle, 4K–124K, 5 depths each; 1.0 = all found.
- **Growing KV**: bytes per token for the 4 full-attention KV-owner layers (sliding-window KV is a fixed ~20 MB, not counted).
  Quantized formats count block scales.

## KV quantization on the original weights

| Config | Growing KV B/tok | web 4K ppl / KL | code 4K | book 32K | book 128K | NIAH |
|---|---|---|---|---|---|---|
| teacher (bf16) | 16,384 | 13.57 / 0 | 2.908 / 0 | 8.837 / 0 | 8.724 / 0 | 1.0 |
| q8_0 all layers | 8,704 | 13.59 / 0.009 | 2.906 / 0.009 | 8.839 / 0.001 | 8.726 / 0.001 | 1.0 |
| fp8 e4m3, unit scale | 8,192 | 13.54 / 0.019 | – | 8.873 / 0.005 | – | – |
| fp4b16 full + q8_0 sliding | 4,608 | 13.49 / 0.027 | 2.893 / 0.028 | 8.865 / 0.004 | 8.757 / 0.004 | 1.0 |
| q4_0 all layers | 4,608 | 13.32 / 0.057 | 2.879 / 0.051 | 9.082 / 0.027 | 8.966 / 0.026 | 1.0 |

128K book ppl by position:

| Config | 0–4K | 4–32K | 32–64K | 64–128K |
|---|---|---|---|---|
| teacher | 9.495 | 8.962 | 8.639 | 8.619 |
| fp4b16 full + q8_0 sliding | 9.517 | 8.989 | 8.671 | 8.655 |
| q4_0 all layers | 9.771 | 9.196 | 8.876 | 8.864 |

Takeaways:
- 8-bit KV is free.
- DeepSeek-style fp4 (E2M1, E4M3 scale per 16) on the global layers, with q8_0 on sliding layers, costs ~0.4% ppl
  at 128K with no training: 3.6x less growing KV, recall intact.
- llama.cpp q4_0 on every layer costs ~2.8% on long books. Not yet isolated whether that comes from 4-bit
  sliding layers or from q4_0 vs fp4 on global layers (needs fp4-global + bf16-sliding and q4_0-global +
  q8_0-sliding runs). Lower web/code ppl under q4 is not an improvement: KL shows the predictions moved.

## Untrained structural variants (bf16 KV)

| Variant | Growing KV B/tok | web 4K ppl / KL | book 32K ppl / KL |
|---|---|---|---|
| g1: 1 KV head on global layers (mean-pooled) | 8,192 | 74.5 / 1.72 | 93.0 / 2.02 |
| g1kv: g1 + K=V | 8,192 (4,096 with a K=V-aware cache) | 2,691 / 5.26 | 1,905 / 4.95 |
| s24: 24 shared layers | 12,288 | 2.9e6 / 11.4 | 4.4e5 / 9.7 |
| s30: 30 shared layers | 8,192 | 4.9e8 / 16.2 | 7.4e7 / 14.5 |
| s36: 36 shared layers | 4,096 | 9.5e8 / 16.7 | 1.0e8 / 14.7 |

Takeaways:
- g1 is a mild break (KL ~2): a good first healing target.
- K=V without training is much worse than g1 alone; heal it as a separate step after g1.
- Extra layer sharing via `num_kv_shared_layers` destroys the model (KL 10–17). That config also shares sliding
  layers, which saves no memory but adds damage. Next: a custom sharing that only redirects global layers.

## Incidents

- book_128k at 16K prefill chunks OOMed (transformers builds a chunk x 128K mask); fixed with 4K chunks.
- fp4 at 128K OOMed: nearest-value lookup built an 8x fp32 temporary. Replaced with `bucketize` plus 16K-token
  slices (bit-identical output, `tests/test_kvquant.py`); peak extra memory 4 GB+ → 0.54 GB.
