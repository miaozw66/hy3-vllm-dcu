#!/usr/bin/env python3
"""Correctness check: for the 4 TP8 shapes x m in {1,2,4,8,32}, run the m4096
.so's m<=32 split-k DUMMA arm via gemm_out and verify numerics match an exact
fp64 reference (int32 dot exact in fp64, then per-row x_scale x per-col
weight_scale -> bf16). Also profiles one call to confirm the
w8a8_dumma_m32_splitk_partial/combine kernels actually launch.
Usage: python3 verify_m32_fastpath.py
"""
import os
import sys

os.environ.setdefault(
    "ZTH_W8A8_CACHE", "/home/hy3-vllm-dcu/算子优化/deliver/so_cache")
sys.path.insert(0, "/home/hy3-vllm-dcu/算子优化/deliver/python")

import torch

# (name, n, k) per TP8 shape; the m<=32 arm lives in the m4096 extension.
SHAPES = [
    ("qkv_proj_m4096", 1280, 4096),
    ("o_proj_m4096", 4096, 1024),
    ("shared_gate_up_proj_m4096", 384, 4096),
    ("shared_down_proj_m4096", 4096, 192),
]
M_VALUES = [1, 2, 4, 8, 32]


def workspace_bytes(n, k):
    bpp = 32 * n * 4
    cap = min(16, (16 * 1024 * 1024) // bpp, k // 32)
    return max(256, cap * bpp)


def main():
    import zth_w8a8_ext
    failed = zth_w8a8_ext.load_all(verbose=False, include_standalone_only=True)
    assert not failed, f"failed to load: {failed}"
    from vllm import _custom_ops as vops

    torch.manual_seed(0)
    all_ok = True
    for name, n, k in SHAPES:
        ops = getattr(torch.ops, f"zth_{name}")
        w_q = torch.randint(-127, 128, (k, n), dtype=torch.int8,
                            device="cuda")
        w_s = (torch.rand(n, 1, device="cuda") * 0.5 + 0.05).float()
        packed, pscale = ops.pack_weight(w_q.contiguous(), w_s.contiguous())
        ws = torch.empty(workspace_bytes(n, k), dtype=torch.uint8,
                         device="cuda")
        for m in M_VALUES:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 3.0
            x_q, x_s, _ = vops.scaled_int8_quant(
                x.contiguous(), None, None, symmetric=True)
            out = ops.gemm_out(x_q, packed, x_s, pscale, ws)
            torch.cuda.synchronize()

            # exact int32 dot in fp64, then per-row / per-col scales
            ref = (x_q.to(torch.float64) @ w_q.to(torch.float64))
            ref = (ref.to(torch.float32) * x_s * w_s.t()).to(torch.bfloat16)
            md = (out.float() - ref.float()).abs().max().item()
            ok = md < 1.0
            all_ok = all_ok and ok
            print(f"{name:28s} m={m:3d} n={n:5d} k={k:5d} "
                  f"max|hip-fp64|={md:10.4f}  {'OK' if ok else 'FAIL'}")

    # profile one m=32 qkv call to confirm the m32 kernels launch
    ops = getattr(torch.ops, "zth_qkv_proj_m4096")
    w_q = torch.randint(-127, 128, (4096, 1280), dtype=torch.int8,
                        device="cuda")
    w_s = torch.rand(1280, 1, device="cuda").float()
    packed, pscale = ops.pack_weight(w_q.contiguous(), w_s.contiguous())
    x = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda") * 3.0
    x_q, x_s, _ = vops.scaled_int8_quant(
        x.contiguous(), None, None, symmetric=True)
    ws = torch.empty(workspace_bytes(1280, 4096), dtype=torch.uint8,
                     device="cuda")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            ops.gemm_out(x_q, packed, x_s, pscale, ws)
        torch.cuda.synchronize()
    knames = sorted({e.key for e in prof.key_averages()})
    print("\nlaunched kernels (m=32 qkv):")
    for kn in knames:
        print("   ", kn[:110])

    status = "PASS" if all_ok else "FAIL"
    print(f"\n{status}: m32 split-k kernels correct")


if __name__ == "__main__":
    print("torch", torch.__version__)
    main()
