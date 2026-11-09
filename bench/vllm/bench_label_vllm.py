"""Labelling speed test, vLLM engine: prompt_logprobs=32 on the same packed 8K chunks as the transformers test.
Timing includes turning vLLM's per-token Python dicts into arrays, since the labeller would have to do that."""
import json, math, sys, time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

CACHE = Path.home() / ".cache/gemma-kv-retrofit"
arr = np.load(CACHE / "data/bench_8k.npy")
K = 32
llm = LLM(model="google/gemma-4-E4B", max_model_len=arr.shape[1] + 8, max_logprobs=K,
          gpu_memory_utilization=float(sys.argv[1]) if len(sys.argv) > 1 else 0.80,
          limit_mm_per_prompt={"image": 0, "audio": 0}, enable_prefix_caching=False,
          max_num_batched_tokens=int(sys.argv[2]) if len(sys.argv) > 2 else 4096)
sp = SamplingParams(max_tokens=1, prompt_logprobs=K, temperature=0)
prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in arr]

llm.generate(prompts[:2], sp, use_tqdm=False)  # warmup
t0 = time.time()
outs = llm.generate(prompts[2:], sp, use_tqdm=False)
t_gen = time.time() - t0
nll_tail = []
for j, o in enumerate(outs):
    pl = o.prompt_logprobs  # list[None | dict[int, Logprob]], one per prompt token
    T = len(pl)
    idx = np.zeros((T, K), np.int32); lp = np.zeros((T, K), np.float16)
    row = arr[j + 2]
    for t in range(1, T):
        d = pl[t]
        top = sorted(((v.rank, k, v.logprob) for k, v in d.items() if v.rank <= K))[:K]
        idx[t] = [k for _, k, _ in top]; lp[t] = [p for _, _, p in top]
        if j >= len(outs) - 4:
            nll_tail.append(-d[int(row[t])].logprob)
t_all = time.time() - t0
n_tok = arr[2:].size
res = {"tok_per_s_engine": n_tok / t_gen, "tok_per_s_with_convert": n_tok / t_all,
       "ppl_last4": math.exp(sum(nll_tail) / len(nll_tail))}
print(json.dumps(res, indent=1))
(CACHE / "results/bench_label_vllm.json").write_text(json.dumps(res, indent=1))
