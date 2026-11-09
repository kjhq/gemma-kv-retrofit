# Healing on 10M tokens (2026-11-04)

Target cache: 1 KV head on the 4 global KV-owner layers, K=V (V-only cache, K rebuilt as rope(V * k_norm.weight)),
fp4b16 on that V, q8_0 on sliding layers. Growing KV 1,152 B/token (original 16,384; DeepSeek V4.1 Flash ~890).
Training simulates exactly that cache (QAT, straight-through estimator). Distillation target: teacher top-32
log-probs on 10.5M FineWeb-Edu tokens (shard 000, 8K sequences, `scripts/label.py`).

| Run | What trains | Steps (x 65K tokens) | Held-out web_4k[:64] ppl | KL to teacher |
|---|---|---|---|---|
| untrained | – | 0 | 2,717 | 5.29 |
| B | global-attn q/k/o + norms (152M), lr 1e-4 cosine | 40 | 14.74 | 0.498 |
| B | same | 80 | 14.48 | 0.430 |
| C (from B@80) | + LoRA r32 on MLPs and sliding attn (66M), lr 5e-5 / 2e-4 | 40 | 13.03 | 0.357 |
| C | same | 80 | 13.46 | 0.323 |

For reference (full web_4k, no training): original ppl 13.57; fp4 global + q8_0 sliding ppl 13.49, KL 0.027.

- Perplexity on this 64-sequence slice is not directly comparable with the full-set 13.57; KL is the trustworthy
  "how close to the original" number. C's ppl rising 13.03 → 13.46 while KL fell is the student moving toward
  the teacher's own uncertainty.
- B plateaued at training KL ~0.375 by step 60. Letting the rest of the network adapt (C) lowered KL another ~25%.
- C was stopped at step ~105/160; the last checkpoint is step 80. Full eval (code, 128K books, NIAH) not run.
- Closing the remaining gap (KL ~0.3 vs 0.03 for fp4 alone) would need far more tokens (likely 100M+), which
  was judged not worth it for now.

Checkpoints (`~/.cache/gemma-kv-retrofit/runs/heal/`):
- `B_g1kv_fp4/ckpt.pt`: step 80, fp32 master copies of the trained attention weights + optimizer.
- `C_g1kv_fp4_lora/ckpt.pt`: step 80, same plus LoRA A/B (rank 32). Resume:
  `scripts/heal.py --tag C_g1kv_fp4_lora --variant g1kv --kv-full fp4b16 --kv-sliding q8_0 --init-from B_g1kv_fp4 --lora-rank 32 --lr 5e-5 --lora-lr 2e-4 --shard-offset 10`

Speed: training runs at 1,600–1,700 tok/s at 8K sequences with gradient checkpointing.
Run C was interrupted once and resumed from its step-40 checkpoint with identical KL.
