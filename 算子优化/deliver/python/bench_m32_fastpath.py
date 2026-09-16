#!/usr/bin/env python3
"""Benchmark: zth m<=32 split-k DUMMA (m4096 extension arm) vs vLLM Triton
scaled_mm heuristic, per shape x m in {1,2,4,8,32}. Graph-replay timing
(warmup + CUDA-graph capture + replay), matching the launch-bound regime.
Also reports max|zth - triton| for a first-order correctness gate.
Usage: python3 bench_m32_fastpath.py
"""
import os
import statistics
import sys
import time

os.environ.setdefault(
    "ZTH_W8A8_CACHE", "/home/hy3-vllm-dcu/算子优化/deliver/so_cache")
sys.path.insert(0, "/home/hy3-vllm-dcu/算子优化/deliver/python")

import torch

SHAPES = [
    ("qkv_proj_m4096", "qkv_proj", 1280, 4096),
    ("o_proj_m4096", "o_proj", 4096, 1024),
    ("shared_gate_up_proj_m4096", "gate_up", 384, 4096),
    ("shared_down_proj_m4096", "down", 4096, 192),
]
M_VALUES = [1, 2, 4, 8, 32]
WARMUP, ITERS, ROUNDS = 20, 200, 3

# NOTE: on this DCU/torch build, torch.cuda.Event.elapsed_time reports a value
# ~1000x too small (verified against wall-clock on a 1.5ms matmul: event 1.46
# vs wall 1544us). All timings below therefore use CUDA-graph replay + wall
# clock, which also removes Python launch overhead (the launch-bound regime
# that matters in production).


def workspace_bytes(n, k):
    bpp = 32 * n * 4
    cap = min(16, (16 * 1024 * 1024) // bpp, k // 32)
    return max(256, cap * bpp)


def measure(call, try_graph=True):
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    if try_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
        except Exception as exc:  # noqa: BLE001
            print(f"  (graph capture failed, using raw wall clock: {exc})")
            try_graph = False
    if not try_graph:
        samples = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(ITERS):
                call()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6 / ITERS)
        return statistics.median(samples)
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            graph.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6 / ITERS)
    return statistics.median(samples)


def main():
    import zth_w8a8_ext
    assert not zth_w8a8_ext.load_all(verbose=False,
                                     include_standalone_only=True)
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )

    torch.manual_seed(0)
    print(f"{'shape':30s} {'m':>3} {'zth_us':>8} {'tri_us':>8} {'speedup':>7} "
          f"{'maxerr':>8}")
    for name, label, n, k in SHAPES:
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

            zth_call = lambda: ops.gemm_out(x_q, packed, x_s, pscale, ws)  # noqa: E731
            tri_call = lambda: triton_scaled_mm(  # noqa: E731
                x_q, w_q, scale_a=x_s, scale_b=w_s.view(-1),
                out_dtype=torch.bfloat16, bias=None)

            md = (zth_call().float() - tri_call().float()).abs().max().item()
            zt = measure(zth_call)
            tt = measure(tri_call)
            print(f"{label:30s} {m:3d} {zt:8.1f} {tt:8.1f} {tt / zt:7.2f}x "
                  f"{md:8.3f}")


if __name__ == "__main__":
    print("torch", torch.__version__)
    main()
