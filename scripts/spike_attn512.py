"""Spike: which PyTorch SDPA backends run head_dim=512 causal attention on this GPU, and how fast/memory-hungry."""
import time, torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F
dev = "cuda"
for hd, H, Hkv in [(256, 8, 2), (512, 8, 2)]:
    for N in [8192, 32768]:
        q = torch.randn(1, H, N, hd, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, H, N, hd, device=dev, dtype=torch.bfloat16)  # pre-expanded GQA
        v = torch.randn_like(k)
        for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            try:
                with sdpa_kernel([be]):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.cuda.synchronize()
                    t = time.time(); F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.cuda.synchronize()
                print(f"hd={hd} N={N:>6} {be.name:<20} ok  {1000*(time.time()-t):7.1f} ms  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
            except Exception as e:
                print(f"hd={hd} N={N:>6} {be.name:<20} FAIL {str(e).splitlines()[0][:90]}")
        del q, k, v; torch.cuda.empty_cache()
