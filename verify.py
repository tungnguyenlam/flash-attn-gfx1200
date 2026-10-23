#!/usr/bin/env python
"""Correctness + speed check for the gfx1200 flash-attn wheel.

Compares flash_attn_func (CK backend) against PyTorch's math SDPA reference for
a LFM2-shaped GQA workload, then benchmarks forward/backward.

    python verify.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import flash_attn
from flash_attn import flash_attn_func


def reference(q, k, v, causal=True):
    """Float32 SDPA reference. q/k/v are (B, S, H, D)."""
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            is_causal=causal,
            enable_gqa=True,
        )
    return out.transpose(1, 2)


def make_qkv(batch, seqlen, nheads, nheads_kv, headdim, dtype=torch.bfloat16):
    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen, nheads_kv, headdim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, nheads_kv, headdim, device="cuda", dtype=dtype)
    return q, k, v


def check(batch, seqlen, nheads, nheads_kv, headdim, causal=True):
    q, k, v = make_qkv(batch, seqlen, nheads, nheads_kv, headdim)
    q1, k1, v1 = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    q2, k2, v2 = (x.detach().clone().requires_grad_(True) for x in (q, k, v))

    out = flash_attn_func(q1, k1, v1, causal=causal)
    ref = reference(q2, k2, v2, causal=causal)

    fwd_err = (out.float() - ref).abs().max().item()
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())

    grad_errs = {
        name: (a.grad.float() - b.grad.float()).abs().max().item()
        for name, a, b in (("dq", q1, q2), ("dk", k1, k2), ("dv", v1, v2))
    }
    print(
        f"B={batch} S={seqlen} H={nheads}/{nheads_kv} D={headdim} causal={causal}: "
        f"fwd maxerr {fwd_err:.4f} | "
        + " ".join(f"{k} maxerr {v:.4f}" for k, v in grad_errs.items())
    )
    assert fwd_err < 0.08, fwd_err
    assert max(grad_errs.values()) < 0.25, grad_errs


def bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def speed(batch, seqlen, nheads, nheads_kv, headdim, dtype=torch.bfloat16):
    q, k, v = make_qkv(batch, seqlen, nheads, nheads_kv, headdim, dtype)
    qk = q.transpose(1, 2)
    kk = k.transpose(1, 2)
    vk = v.transpose(1, 2)

    fa_fwd = lambda: flash_attn_func(q, k, v, causal=True)
    sdpa_fwd = lambda: F.scaled_dot_product_attention(qk, kk, vk, is_causal=True, enable_gqa=True)
    fa_ms = bench(fa_fwd)
    sdpa_ms = bench(sdpa_fwd)

    q1, k1, v1 = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    q2, k2, v2 = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    g = torch.randn_like(q)

    def fa_step():
        flash_attn_func(q1, k1, v1, causal=True).backward(g)
        q1.grad = k1.grad = v1.grad = None

    def sdpa_step():
        F.scaled_dot_product_attention(
            q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2),
            is_causal=True, enable_gqa=True,
        ).backward(g.transpose(1, 2))
        q2.grad = k2.grad = v2.grad = None

    fa_bwd_ms = bench(fa_step, warmup=5, rep=20)
    sdpa_bwd_ms = bench(sdpa_step, warmup=5, rep=20)

    print(
        f"B={batch} S={seqlen} H={nheads}/{nheads_kv} D={headdim}: "
        f"fwd FA {fa_ms:.3f} ms vs SDPA {sdpa_ms:.3f} ms ({sdpa_ms / fa_ms:.2f}x) | "
        f"fwd+bwd FA {fa_bwd_ms:.3f} ms vs SDPA {sdpa_bwd_ms:.3f} ms ({sdpa_bwd_ms / fa_bwd_ms:.2f}x)"
    )


def main():
    print(f"flash-attn {flash_attn.__version__}")
    print(f"torch {torch.__version__} | hip {torch.version.hip}")
    print(f"GPU {torch.cuda.get_device_name(0)} | arch {torch.cuda.get_device_properties(0).gcnArchName}")

    check(2, 512, 16, 8, 64)
    check(2, 1024, 16, 8, 64)
    check(2, 512, 16, 16, 64)

    for seqlen in (512, 2048, 8192):
        speed(1, seqlen, 16, 8, 64)


if __name__ == "__main__":
    main()
