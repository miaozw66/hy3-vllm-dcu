#!/usr/bin/env python3
"""Kernel-level microbenchmark of the zth m16 .so dispatch.

Directly calls torch.ops.zth_*.gemm_out at M=16 and M=8 for the four TP8
shapes, bypassing the Python routing in zth_w8a8_integrate.py, to confirm:
  - M=16 (the value the current routing forwards) -> fast M16 DUMMA
  - M=8  (would only occur if a non-16 M slipped through routing) -> scalar
Also compares bf16 dequant reference for correctness on one shape.
"""
import os
import sys
import time

os.environ["ZTH_W8A8_CACHE"] = "/home/hy3-vllm-dcu/算子优化/deliver/so_cache"
sys.path.insert(0, "/home/hy3-vllm-dcu/算子优化/deliver/python")

import torch

torch.manual_seed(0)
dev = "cuda"

import zth_w8a8_ext

failed = zth_w8a8_ext.load_all(verbose=True)
assert not failed, f"load failed: {failed}"

# (name, k, n)  TP8 shapes, weight is [k, n]
SHAPES = [
    ("qkv_proj_m16", 4096, 1280),
    ("o_proj_m16", 1024, 4096),
    ("shared_gate_up_proj_m16", 4096, 384),
    ("shared_down_proj_m16", 192, 4096),
]


def pack(name, k, n):
    mod = getattr(torch.ops, f"zth_{name}")
    raw = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=dev)
    ws = torch.rand(n, dtype=torch.float32, device=dev) * 0.01 + 0.001
    packed, pscale = mod.pack_weight(raw, ws)
    return mod, packed, pscale, raw, ws


def run(mod, packed, pscale, k, n, m, iters=20):
    x = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=dev)
    xs = (torch.rand(m, dtype=torch.float32, device=dev) * 0.01 + 0.001)
    ws_bytes = m * n * 4
    budget = 16 * 1024 * 1024
    cap = min(16, budget // ws_bytes, k // 32)
    ws_alloc = max(256, cap * ws_bytes)
    workspace = torch.empty(ws_alloc, dtype=torch.uint8, device=dev)
    # warmup
    for _ in range(3):
        out = mod.gemm_out(x, packed, xs, pscale, workspace)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = mod.gemm_out(x, packed, xs, pscale, workspace)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, out


print(f"{'shape':22s} {'M':>4s} {'ms/call':>10s}")
for name, k, n in SHAPES:
    mod, packed, pscale, raw, ws = pack(name, k, n)
    for m in (16, 8, 4, 1):
        ms, out = run(mod, packed, pscale, k, n, m)
        print(f"{name:22s} {m:4d} {ms:10.4f}")
    # correctness check at M=16 vs float reference
    mod, packed, pscale, raw, ws = pack(name, k, n)
    ms, out = run(mod, packed, pscale, k, n, 16, iters=1)
    m = 16
    x = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=dev).float()
    xs = (torch.rand(m, dtype=torch.float32, device=dev) * 0.01 + 0.001)
    ref = (x.unsqueeze(1) * raw.T.unsqueeze(0)).sum(-1) * xs.unsqueeze(1) * ws.unsqueeze(0)
    ref_bf = ref.to(torch.bfloat16)
    err = (out.float() - ref_bf.float()).abs().max().item()
    print(f"{name:22s}    max_abs_err@M16={err:.3f}")
    print()

print("DONE")
