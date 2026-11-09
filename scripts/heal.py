"""Heal a KV-reduced student by distilling the teacher's offline top-32 log-probs (resumable).

Only full-attention layers train (q/k/v/o projections and q/k norm scales); everything else is frozen.
Weights stay bf16 in the model; AdamW runs on fp32 master copies. With --kv-full fp4b16 the KV cache
is fake-quantized in the forward pass with a straight-through estimator (QAT).

  uv run python scripts/heal.py --tag A_g1kv --variant g1kv
  uv run python scripts/heal.py --tag B_g1kv_fp4 --variant g1kv --kv-full fp4b16 --kv-sliding q8_0

Checkpoint (trainable params + optimizer) every --ckpt-every steps under runs/heal/<tag>/; a rerun resumes.
At the end: full Stage-0 eval set, results in results/heal/<tag>__<set>.json.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch

from kvr import evalppl, lora, niah
from kvr.logits import kl_loss_backward
from kvr.model import full_layers, kv_bytes_per_token, load, set_kv_quant, tokenizer

CACHE = Path.home() / ".cache/gemma-kv-retrofit"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--variant", default="g1kv")
ap.add_argument("--kv-full", default="bf16")
ap.add_argument("--kv-sliding", default="bf16")
ap.add_argument("--labels", default="fwe_8k")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8, help="8K sequences per optimizer step")
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--ckpt-every", type=int, default=40)
ap.add_argument("--eval-every", type=int, default=40)
ap.add_argument("--max-steps", type=int, default=0, help="0 = one pass over the labels")
ap.add_argument("--no-final-eval", action="store_true")
ap.add_argument("--init-from", default="", help="start from another run's ckpt.pt (trained attention weights only)")
ap.add_argument("--lora-rank", type=int, default=0, help="LoRA on MLPs + sliding attention (0 = off)")
ap.add_argument("--lora-lr", type=float, default=2e-4)
ap.add_argument("--shard-offset", type=int, default=0, help="start the data pass at this label shard (wraps around)")
args = ap.parse_args()

run_dir = CACHE / "runs/heal" / args.tag
run_dir.mkdir(parents=True, exist_ok=True)
res_dir = CACHE / "results/heal"
res_dir.mkdir(parents=True, exist_ok=True)
logf = open(run_dir / "train.jsonl", "a")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- model
model = load(args.variant)
set_kv_quant(model, full=args.kv_full, sliding=args.kv_sliding, train=True)
cfg = model.config
for p in model.parameters():
    p.requires_grad_(False)
train_named = []
for i in full_layers(cfg):
    a = model.model.layers[i].self_attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        m = getattr(a, name, None)
        if m is not None and getattr(m, "weight", None) is not None:
            m.weight.requires_grad_(True)
            train_named.append((f"layers.{i}.self_attn.{name}.weight", m.weight))
masters = [p.detach().float().clone().requires_grad_(True) for _, p in train_named]
log(f"{args.tag}: variant {args.variant}, KV full={args.kv_full} sliding={args.kv_sliding}, "
    f"{sum(p.numel() for p in masters)/1e6:.0f}M trainable params, growing KV {kv_bytes_per_token(model, args.kv_full):.0f} B/token")
if args.init_from:
    st = torch.load(CACHE / "runs/heal" / args.init_from / "ckpt.pt", map_location="cuda")
    with torch.no_grad():
        for m, (_, p), w in zip(masters, train_named, st["masters"]):
            m.copy_(w); p.copy_(w.to(p.dtype))
    log(f"initialised attention weights from {args.init_from} step {st['step']}")
lora_params = lora.apply(model, args.lora_rank) if args.lora_rank else []
if lora_params:
    log(f"LoRA rank {args.lora_rank}: {sum(p.numel() for p in lora_params)/1e6:.0f}M params")
groups = [{"params": masters, "lr": args.lr, "base_lr": args.lr}]
if lora_params:
    groups.append({"params": lora_params, "lr": args.lora_lr, "base_lr": args.lora_lr})
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
all_trainable = masters + lora_params
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.train()

# ---------------------------------------------------------------- data
shards = sorted((CACHE / "data/labels" / args.labels).glob("shard_*.npz"))
shards = shards[args.shard_offset:] + shards[:args.shard_offset]
assert shards, "no labels"
seqs_per_shard = np.load(shards[0])["ids"].shape[0]
total_steps = args.max_steps or (len(shards) * seqs_per_shard) // args.accum


def batches():
    for f in shards:
        z = np.load(f)
        for j in range(len(z["ids"])):
            yield z["ids"][j], z["lp"][j], z["idx"][j]


def lr_at(step, base=None):
    base = args.lr if base is None else base
    if step < args.warmup:
        return base * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, total_steps - args.warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def quick_eval(step):
    model.eval()
    set_kv_quant(model, args.kv_full, args.kv_sliding, train=False)
    with torch.no_grad():
        r = evalppl.evaluate(model, "web_4k", limit=64)
    set_kv_quant(model, args.kv_full, args.kv_sliding, train=True)
    model.train()
    log(f"step {step} eval web_4k[:64]: ppl {r['ppl']:.3f} KL {r['kl']:.4f}")
    logf.write(json.dumps({"step": step, "eval_web64": r}) + "\n"); logf.flush()


# ---------------------------------------------------------------- resume
step = 0
ck = run_dir / "ckpt.pt"
if ck.exists():
    st = torch.load(ck, map_location="cuda")
    for m, (_, p), w in zip(masters, train_named, st["masters"]):
        m.data.copy_(w); p.data.copy_(w.to(p.dtype))
    for p, w in zip(lora_params, st.get("lora", [])):
        p.data.copy_(w)
    opt.load_state_dict(st["opt"])
    step = st["step"]
    log(f"resumed at step {step}")
if step == 0:
    quick_eval(0)

# ---------------------------------------------------------------- train
it = batches()
for _ in range(step * args.accum):  # skip consumed sequences
    next(it)
t0, tok = time.time(), 0
while step < total_steps:
    for g in opt.param_groups:
        g["lr"] = lr_at(step, g["base_lr"])
    for m in masters:
        m.grad = torch.zeros_like(m)
    for p in lora_params:
        p.grad = None
    kl_sum, n = 0.0, 0
    for _ in range(args.accum):
        ids, lp, idx = next(it)
        x = torch.from_numpy(ids[None].astype(np.int64)).cuda()
        h = model.model(input_ids=x, use_cache=False).last_hidden_state.reshape(-1, cfg.hidden_size)
        T = h.shape[0]
        kl_sum += kl_loss_backward(model, h, torch.from_numpy(lp).cuda(), torch.from_numpy(idx).cuda(), 1.0 / (T * args.accum))
        n += T
        for m, (_, p) in zip(masters, train_named):
            if p.grad is not None:
                m.grad += p.grad.float()
                p.grad = None
    gnorm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0).item()
    opt.step()
    with torch.no_grad():
        for m, (_, p) in zip(masters, train_named):
            p.copy_(m.to(p.dtype))
    step += 1
    tok += n
    rate = tok / (time.time() - t0)
    rec = {"step": step, "kl": kl_sum / n, "gnorm": gnorm, "lr": lr_at(step - 1), "tok_s": rate}
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    if step % 5 == 0 or step == 1:
        eta = (total_steps - step) * args.accum * 8192 / rate / 60
        log(f"step {step}/{total_steps} KL {rec['kl']:.4f} gnorm {gnorm:.2f} lr {rec['lr']:.2e} {rate:,.0f} tok/s, ETA {eta:.0f} min")
    if step % args.ckpt_every == 0 or step == total_steps:
        torch.save({"step": step, "masters": [m.detach() for m in masters], "lora": [p.detach() for p in lora_params],
                    "opt": opt.state_dict()}, run_dir / "ckpt.tmp")
        (run_dir / "ckpt.tmp").replace(ck)
    if step % args.eval_every == 0 and step < total_steps:
        quick_eval(step)

# ---------------------------------------------------------------- final: weights + full eval
torch.save({n: p.detach().cpu() for n, p in train_named}, run_dir / "trained_attn.pt")
if lora_params:
    torch.save({"rank": args.lora_rank, "state": lora.lora_state(model)}, run_dir / "lora.pt")
if args.no_final_eval:
    raise SystemExit
model.eval()
model.gradient_checkpointing_disable()
set_kv_quant(model, args.kv_full, args.kv_sliding, train=False)
meta = {"tag": args.tag, "variant": args.variant, "init_from": args.init_from, "lora_rank": args.lora_rank, "kv_full": args.kv_full, "kv_sliding": args.kv_sliding,
        "kv_bytes_per_token": kv_bytes_per_token(model, args.kv_full), "steps": step, "tokens": step * args.accum * 8192}
with torch.no_grad():
    for s in ["web_4k", "code_4k", "book_32k", "book_128k", "niah"]:
        out = res_dir / f"{args.tag}__{s}.json"
        if out.exists():
            continue
        r = niah.run(model, tokenizer()) if s == "niah" else evalppl.evaluate(model, s)
        out.write_text(json.dumps({**meta, "eval": s, **r}, indent=1))
        log(f"final {s}: " + json.dumps({k: v for k, v in r.items() if k != 'buckets'}))
(run_dir / "DONE").write_text(time.strftime("%F %T"))
log("heal done")
