# gemma-kv-retrofit

Gemma 4 E4B keeps 16 KB of memory for every token of context (its KV cache). At 128K tokens of context that is
about 2 GB, on a model meant for small devices. This project shrinks that memory **14 times, to 1.15 KB per
token**, and then retrains the changed model against the original to win back the quality it lost.

It uses only switches the Gemma 4 code already has, plus 4-bit storage that the model is trained to tolerate.
One change is made at a time, and each is measured before the next.

## Result

How close each version is to the original model, on held-out web text:

| Version | KV memory per token | KL to the original | Perplexity |
|---|---|---|---|
| Original (bf16) | 16,384 B | 0 | 13.57 |
| 4-bit KV only, no training | 4,608 B | 0.027 | 13.49 |
| All changes, no training | 1,152 B | 5.29 | 2,717 |
| All changes, attention layers retrained | 1,152 B | 0.430 | 14.48 |
| All changes, plus LoRA on the rest of the network | 1,152 B | **0.323** | 13.46 |

- **KL to the original** measures how far the model's next-token predictions have moved from the original's.
  0 means identical. Lower is better, and this is the number to trust.
- **Perplexity** is how surprised the model is by real text. Lower is better. The first two rows use the full
  evaluation set and the last three a 64-sequence slice of it, so compare them only roughly.

"All changes" means: one key-value head instead of eight on the four global-attention layers, keys rebuilt from
the values (K=V, so only V is stored), that V stored in 4 bits, and the sliding-window layers stored in 8 bits.

Retraining used 10.5M tokens of FineWeb-Edu and the original model's top-32 predictions as the target.

### What is not done

- The retrained model still sits at KL 0.32, against 0.03 for 4-bit storage alone. Closing that gap would
  probably take 100M tokens or more.
- The last run was stopped at step 80 of 160.
- The retrained model was scored on web text only. The code, long-book and needle-recall tests were run for the
  untrained versions (below) but not after retraining.

Full tables: [`docs/stage0-results.md`](docs/stage0-results.md) (no training) and
[`docs/heal-10m-results.md`](docs/heal-10m-results.md) (retraining).

### What needs no training at all

From the untrained measurements:

- **8-bit KV is free.** KL 0.009, recall unchanged.
- **4-bit on the global layers with 8-bit on the sliding layers** costs about 0.4% perplexity at 128K tokens and
  keeps needle recall at 100%, for 3.6 times less memory.
- **Sharing KV across more layers through the built-in setting destroys the model** (KL 10 to 17).

## How it works

- `src/kvr/model.py` loads E4B as a text-only model, builds the changed variants and simulates the quantized
  cache during the forward pass.
- `src/kvr/kvquant.py` holds the storage formats: llama.cpp's `q8_0` and `q4_0`, `fp8`, and a 4-bit float format
  with one 8-bit scale per 16 values (`fp4b16`).
- `src/kvr/logits.py` computes log-probabilities, loss and the KL to the teacher in chunks.
- `src/kvr/evalppl.py` and `src/kvr/niah.py` run perplexity and KL by position, and needle recall from 4K to 124K
  tokens.
- `scripts/run_stage0.py` runs the untrained measurements. It can be stopped and resumed.
- `scripts/label.py` stores the original model's top-32 predictions for the training text.
- `scripts/heal.py` retrains a changed model against those predictions. `--kv-full fp4b16` trains with the 4-bit
  cache in the loop. It checkpoints every 40 steps and resumes.

## Running it

You need an NVIDIA GPU with 24 GB of memory and [uv](https://docs.astral.sh/uv/). The LoRA stage needs more
memory than that; lower the sequence length to fit.

```bash
uv sync
hf download google/gemma-4-E4B
scripts/fetch_data.sh                     # FineWeb-Edu shards, PG19, codeparrot validation
uv run python scripts/prep_eval_data.py   # fixed evaluation sets
uv run pytest -q tests
uv run python scripts/run_stage0.py       # untrained measurements
uv run python scripts/label.py            # teacher predictions
uv run python scripts/heal.py --tag B_g1kv_fp4 --variant g1kv --kv-full fp4b16 --kv-sliding q8_0
```

Data, labels, checkpoints and logs go to `~/.cache/gemma-kv-retrofit/`.

## Things that tripped me up

- `Gemma4ForCausalLM.from_pretrained` needs `key_mapping={r"^model\.language_model\.": "model."}`. Without it
  every weight silently loads as random.
- In transformers, `num_global_key_value_heads` only applies when `attention_k_eq_v` is on, so the variants set
  `per_layer_config` directly.
- `attention_k_eq_v` removes the V projection, but standard caches still store K and V separately. It saves no
  memory without a custom cache.
- Gemma 4's 512-dimension global-attention heads ran only on PyTorch's memory-efficient attention backend here.
- 128K evaluations prefill in 4K chunks. With 16K chunks transformers builds a chunk x 128K mask and runs out of
  memory. Against a one-shot pass on a 32K book the perplexity is 6.9235 against 6.9213, which is rounding noise.
- For writing the training labels, vLLM was only 1.09 times faster than plain transformers (8,606 against 7,870
  tokens a second), so labelling uses transformers and shares its code path with training. The vLLM benchmark is
  in `bench/vllm/`.

## Licence

MIT for the code. Gemma 4 carries its own licence.
