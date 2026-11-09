"""Load Gemma 4 E4B (text only), build KV-reduced student variants, and hook simulated KV quantization.

Variants only use switches the Gemma 4 code already has:
  * fewer KV heads on the full-attention layers (per_layer_config num_key_value_heads, as in Gemma 4 12B/31B)
  * attention_k_eq_v (keys reused as values on full-attention layers, as in Gemma 4 12B/31B)
  * num_kv_shared_layers (how many trailing layers reuse earlier layers' KV)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_utils import AttentionInterface
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from .kvquant import FORMATS, ste

MODEL_ID = "google/gemma-4-E4B"
KEY_MAP = {r"^model\.language_model\.": "model."}
CKPT_PREFIX = "model.language_model."


def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def base_text_config():
    return AutoConfig.from_pretrained(MODEL_ID).text_config


def full_layers(cfg) -> list[int]:
    return [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]


def kv_owner_layers(cfg) -> list[int]:
    """Layers that compute and store their own KV (everything before the shared tail)."""
    return list(range(cfg.num_hidden_layers - cfg.num_kv_shared_layers))


@dataclass(frozen=True)
class Variant:
    name: str = "teacher"
    global_kv_heads: int | None = None  # None = keep 2
    k_eq_v: bool = False
    kv_shared: int | None = None  # None = keep 18

    def is_teacher(self) -> bool:
        return self.global_kv_heads is None and not self.k_eq_v and self.kv_shared is None


VARIANTS = {
    "teacher": Variant(),
    "g1": Variant("g1", global_kv_heads=1),
    "g1kv": Variant("g1kv", global_kv_heads=1, k_eq_v=True),
    "s24": Variant("s24", kv_shared=24),
    "s30": Variant("s30", kv_shared=30),
    "s36": Variant("s36", kv_shared=36),
    "g1_s30": Variant("g1_s30", global_kv_heads=1, kv_shared=30),
}


def variant_config(v: Variant):
    cfg = base_text_config()
    if v.is_teacher():
        return cfg
    d = cfg.to_dict()
    if v.kv_shared is not None:
        d["num_kv_shared_layers"] = v.kv_shared
    d["attention_k_eq_v"] = v.k_eq_v
    heads = v.global_kv_heads or cfg.num_key_value_heads
    gdim = cfg.per_layer_config[full_layers(cfg)[0]].head_dim if cfg.per_layer_config else 512
    d["per_layer_config"] = {i: {"head_dim": gdim, "num_key_value_heads": heads} for i in full_layers(cfg)}
    d.pop("num_global_key_value_heads", None)
    d.pop("global_head_dim", None)
    return type(cfg)(**d)


def _teacher_tensor(name: str) -> torch.Tensor:
    path = hf_hub_download(MODEL_ID, "model.safetensors")
    with safe_open(path, "pt") as f:
        return f.get_tensor(CKPT_PREFIX + name)


def load(variant: str | Variant = "teacher", device: str = "cuda", dtype=torch.bfloat16) -> Gemma4ForCausalLM:
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    cfg = variant_config(v)
    model = Gemma4ForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, dtype=dtype, device_map=device, key_mapping=KEY_MAP,
        ignore_mismatched_sizes=not v.is_teacher(),
    )
    if v.global_kv_heads is not None:
        _merge_global_kv_heads(model, cfg, v)
    model.eval()
    model.kvr_variant = v
    return model


@torch.no_grad()
def _merge_global_kv_heads(model, cfg, v: Variant) -> None:
    """GQA-style mean-pool init: average the teacher's KV heads into v.global_kv_heads heads."""
    owners = set(kv_owner_layers(cfg))
    for i in full_layers(cfg):
        if i not in owners:
            continue
        attn = model.model.layers[i].self_attn
        for proj in ("k_proj", "v_proj"):
            mod = getattr(attn, proj, None)
            if mod is None:  # k_eq_v drops v_proj
                continue
            w = _teacher_tensor(f"layers.{i}.self_attn.{proj}.weight")  # (H*D, hidden)
            hd = attn.head_dim
            h_old = w.shape[0] // hd
            w = w.view(h_old, hd, -1).float()
            w = w.view(v.global_kv_heads, h_old // v.global_kv_heads, hd, -1).mean(1).reshape(-1, w.shape[-1])
            mod.weight.copy_(w.to(mod.weight.dtype))


# ---------------------------------------------------------------- simulated KV quantization
_ORIG_SDPA = AttentionInterface._global_mapping["sdpa"]


def _kvq_sdpa(module, query, key, value, attention_mask, **kwargs):
    fn = getattr(module, "_kvq_fn", None)
    if fn is not None:
        if getattr(module, "_kvq_ste", False):
            key, value = ste(fn, key), ste(fn, value)
        else:
            key, value = fn(key), fn(value)
    return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("sdpa", _kvq_sdpa)


def set_kv_quant(model, full: str = "bf16", sliding: str = "bf16", train: bool = False) -> None:
    """Quantize cached K/V on full-attention and sliding layers with the given formats (see kvquant.FORMATS).

    K=V layers simulate a V-only cache: V = v_norm(k_proj x) is quantized once, before the cache, and
    K = rope(V * k_norm.weight) is rebuilt from the quantized V (exact, since k_norm = v_norm * weight).
    Their full-attention layers (and the layers sharing their KV) then skip the SDPA-level quantization.
    """
    kveq = any(getattr(l.self_attn, "use_alternative_attention", False) for l in model.model.layers)
    for layer in model.model.layers:
        a = layer.self_attn
        fn = FORMATS[sliding if a.is_sliding else full]
        _set_vonly_quant(a, fn if (a.use_alternative_attention and fn is not None) else None, train)
        a._kvq_fn = None if (kveq and not a.is_sliding) else fn
        a._kvq_ste = train


def _set_vonly_quant(a, fn, train: bool) -> None:
    if getattr(a, "is_kv_shared_layer", False) or not hasattr(a, "v_norm"):
        return
    for m in (a.v_norm, a.k_norm):
        m.__dict__.pop("forward", None)  # back to the class method
    if fn is None:
        return
    vn, kn = a.v_norm, a.k_norm
    q = (lambda x: ste(fn, x)) if train else fn
    vn.forward = lambda x: q(type(vn).forward(vn, x))
    kn.forward = lambda x: (q(type(vn).forward(vn, x)).float() * kn.weight.float()).type_as(x)


def kv_bytes_per_token(model, full: str = "bf16") -> float:
    """Growing (full-attention) KV bytes per token. Sliding layers are a fixed cost and excluded."""
    from .kvquant import BITS
    cfg = model.config
    owners = set(kv_owner_layers(cfg))
    total = 0.0
    for i in full_layers(cfg):
        if i in owners:
            a = model.model.layers[i].self_attn
            heads = a.k_proj.weight.shape[0] // a.head_dim
            n = 1 if a.v_proj is None else 2  # K=V: a V-only cache (K is rebuilt from V)
            total += n * heads * a.head_dim * BITS[full] / 8
    return total


def clone_variant(v: Variant, **kw) -> Variant:
    return copy.replace(v, **kw)
