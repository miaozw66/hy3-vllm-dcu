#!/usr/bin/env python3
"""Validate and time the HY3 routed w13 HIP fast path at M=4."""

import argparse
import os
import pathlib

import torch

os.environ.setdefault(
    "ZTH_ROUTED_W13_SO",
    "/root/.cache/torch_extensions/py310_cpu/routed_w13/routed_w13.so",
)

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.triton_utils import tl


E, N, K, M, TOP_K = 192, 384, 4096, 4, 8


def elapsed_us(fn, warmups: int, iters: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A DCU is required for this validation")
    so_path = pathlib.Path(os.environ["ZTH_ROUTED_W13_SO"])
    if not so_path.is_file():
        raise FileNotFoundError(so_path)
    torch.ops.load_library(str(so_path))
    torch.manual_seed(0)

    qhidden = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    w1 = torch.randint(-127, 128, (E, N, K), device="cuda", dtype=torch.int8)
    a_scale = (
        torch.rand(M, 1, device="cuda", dtype=torch.float32) + 0.01
    ).contiguous()
    w1_scale = (
        torch.rand(E, N, 1, device="cuda", dtype=torch.float32) + 0.01
    ).contiguous()
    topk_ids = torch.randint(E, (M, TOP_K), device="cuda", dtype=torch.int32)
    token_ids = torch.arange(M, device="cuda", dtype=torch.int32).repeat_interleave(TOP_K)
    packed = torch.ops.routed_w13.w1_pack(w1)
    unpacked = torch.ops.routed_w13.w1_unpack(packed, E, N, K)
    if not torch.equal(w1, unpacked):
        raise AssertionError("w1 pack/unpack is not bitwise exact")

    config = try_get_optimal_moe_config(
        w1.size(), (E, K, N // 2), TOP_K, "int8_w8a8", M
    )
    padded = torch.full(
        (1,), M * TOP_K * config["BLOCK_SIZE_M"], device="cuda", dtype=torch.int32
    )
    triton_out = torch.empty((M, TOP_K, N), device="cuda", dtype=torch.bfloat16)
    hip_out = torch.empty((M * TOP_K, N), device="cuda", dtype=torch.bfloat16)

    def run_triton() -> None:
        invoke_fused_moe_triton_kernel(
            qhidden, w1, triton_out, a_scale, w1_scale, None, None,
            topk_ids.view(-1), padded, False, TOP_K, config,
            compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=True,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=True,
        )

    def run_hip() -> None:
        torch.ops.routed_w13.out(
            qhidden, packed, a_scale.reshape(-1), w1_scale.squeeze(-1), token_ids,
            topk_ids.view(-1), hip_out,
        )

    run_triton()
    run_hip()
    torch.cuda.synchronize()
    reference = triton_out.view(-1, N)
    exact_out = torch.empty_like(hip_out)
    for pair in range(M * TOP_K):
        token = pair // TOP_K
        expert = topk_ids.view(-1)[pair]
        accum = torch.matmul(
            qhidden[token].float(), w1[expert].float().transpose(0, 1)
        )
        exact_out[pair] = (accum.float() * a_scale[token, 0] * w1_scale[expert, :, 0]).to(
            torch.bfloat16
        )

    def error_metrics(actual: torch.Tensor) -> tuple[float, float]:
        diff = (exact_out.float() - actual.float()).abs()
        return (
            diff.max().item(),
            (diff / exact_out.float().abs().clamp_min(1e-6)).max().item(),
        )

    hip_abs, hip_rel = error_metrics(hip_out)
    triton_abs, triton_rel = error_metrics(reference)
    finite = torch.isfinite(hip_out).all().item()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run_hip()
    graph.replay()
    torch.cuda.synchronize()
    graph_abs, graph_rel = error_metrics(hip_out)
    triton_us = elapsed_us(run_triton, args.warmups, args.iters)
    hip_us = elapsed_us(run_hip, args.warmups, args.iters)
    print(f"M={M} P={M * TOP_K} config={config}")
    print(f"pack_roundtrip=exact finite={finite}")
    print(f"hip_vs_fp32 max_abs={hip_abs:.6g} max_rel={hip_rel:.6g}")
    print(f"triton_vs_fp32 max_abs={triton_abs:.6g} max_rel={triton_rel:.6g}")
    print(f"graph_replay_vs_fp32 max_abs={graph_abs:.6g} max_rel={graph_rel:.6g}")
    print(f"triton_w13_us={triton_us:.2f} hip_w13_us={hip_us:.2f} speedup={triton_us / hip_us:.2f}x")


if __name__ == "__main__":
    main()
