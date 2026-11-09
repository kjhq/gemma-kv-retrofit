"""Minimal LoRA for healing: y = W x + scale * B(A x). A, B live in fp32; B starts at zero (no-op at init).

Targets: MLP gate/up/down in every layer, and q/k/v/o of sliding-window attention layers (full-attention
layers train their projections directly). merge() folds the adapters back into W for deployment.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + ((x @ self.A.to(x.dtype).T) @ self.B.to(x.dtype).T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight += (self.scale * self.B @ self.A).to(self.base.weight.dtype)
        return self.base


def targets(model):
    """(parent module, attribute name) pairs to wrap."""
    out = []
    for layer in model.model.layers:
        for n in ("gate_proj", "up_proj", "down_proj"):
            out.append((layer.mlp, n))
        a = layer.self_attn
        if a.is_sliding:
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if isinstance(getattr(a, n, None), nn.Linear):
                    out.append((a, n))
    return out


def apply(model, r: int = 32, alpha: float = 32.0) -> list[nn.Parameter]:
    params = []
    for parent, n in targets(model):
        m = LoRALinear(getattr(parent, n), r, alpha)
        setattr(parent, n, m)
        params += [m.A, m.B]
    return params


def lora_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if k.endswith((".A", ".B")) and v.dtype == torch.float32}


def merge(model) -> None:
    for mod in list(model.modules()):
        for n, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, n, child.merged())
