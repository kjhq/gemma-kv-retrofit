import time, torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
M = "google/gemma-4-E4B"
cfg = AutoConfig.from_pretrained(M).text_config
t = time.time()
model = Gemma4ForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map="cuda",
                                          key_mapping={r"^model\.language_model\.": "model."})
print(f"loaded {time.time()-t:.0f}s vram {torch.cuda.memory_allocated()/2**30:.1f} GiB params {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
tok = AutoTokenizer.from_pretrained(M)
for p in ["The capital of France is", "def fibonacci(n):"]:
    ids = tok(p, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=20, do_sample=False)
    print(repr(tok.decode(out[0][ids.input_ids.shape[1]:])))
