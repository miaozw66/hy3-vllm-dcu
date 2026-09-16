#!/usr/bin/env python3
"""Benchmark: zth M16 fast-path gemm_out vs vLLM Triton scaled_mm, per shape.
Usage: python3 bench_m16_fastpath.py [iters]
"""
import os
import sys
import time

os.environ.setdefault(
    "ZTH_W8A8_CACHE", "/home/hy3-vllm-dcu/算子优化/deliver/so_cache")
sys.path.insert(0, "/home/hy3-vllm-dcu/算子优化/deliver/python")

import torch

NAMES = {
    "o_proj_m16": (16, 4096, 1024),
    "qkv_proj_m16": (16, 1280, 4096),
    "shared_down_proj_m16": (16, 4096, 192),
    "shared_gate_up_proj_m16": (16, 384, 4096),
}


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    import zth_w8a8_ext
    assert not zth_w8a8_ext.load_all(verbose=False,
                                     include_standalone_only=True)
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )

    torch.manual_seed(0)
    print(f"{'shape':34s} {'zth_us':>10} {'triton_us':>12} {'speedup':>8}")
    for name, (m, n, k) in NAMES.items():
        ops = getattr(torch.ops, f"zth_{name}")
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 3.0
        x_q, x_s, _ = vops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True)
        w_q = torch.randint(-127, 128, (k, n), dtype=torch.int8,
                            device="cuda")
        w_s = (torch.rand(n, 1, device="cuda") * 0.5 + 0.05).float()
        packed, pscale = ops.pack_weight(w_q.contiguous(), w_s.contiguous())
        # replicate routing workspace allocation
        bpp = m * n * 4
        cap = min(16, (16 * 1024 * 1024) // bpp, k // 32)
        ws = torch.empty(max(256, cap * bpp), dtype=torch.uint8,
                         device="cuda")

        def zth_call():
            return ops.gemm_out(x_q, packed, x_s, pscale, ws)

        def tri_call():
            return triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s.view(-1),
                                    out_dtype=torch.bfloat16, bias=None)

        for _ in range(20):
            zth_call()
            tri_call()
        torch.cuda.synchronize()

        def bench(fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e6

        zt = bench(zth_call)
        tt = bench(tri_call)
        # verify equal output
        md = (zth_call().float() - tri_call().float()).abs().max().item()
        print(f"{name:34s} {zt:10.1f} {tt:12.1f} {tt / zt:8.2f}x  "
              f"(max|zth-tri|={md:.4f})")


if __name__ == "__main__":
    print("torch", torch.__version__)
    main()
