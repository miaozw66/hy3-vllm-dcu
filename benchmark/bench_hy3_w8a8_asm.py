#!/usr/bin/env python3
"""Graph-replay latency and effective-bandwidth benchmark for HY3 W8A8 asm."""

import argparse
import statistics
from pathlib import Path

import torch

K = 4096
N = 384
WEIGHT_BYTES = K * N
TOTAL_BYTES = 1 * K + WEIGHT_BYTES + 2 * N


def percentile(samples: list[float], p: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * p)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--split-k", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--a-resident", action="store_true")
    parser.add_argument("--persistent-workspace", action="store_true")
    args = parser.parse_args()
    if args.a_resident and args.split_k != 16:
        raise ValueError("--a-resident currently requires --split-k 16")
    if args.persistent_workspace and not args.a_resident:
        raise ValueError("--persistent-workspace currently requires --a-resident")

    torch.cuda.set_device(args.gpu)
    torch.ops.load_library(str(args.extension))
    op = torch.ops._rocm_C
    scaled_mm = op.hy3_w8a8_scaled_mm_splitk16_aresident if args.a_resident else {
        1: op.hy3_w8a8_scaled_mm,
        4: op.hy3_w8a8_scaled_mm_splitk4,
        8: op.hy3_w8a8_scaled_mm_splitk8,
        16: op.hy3_w8a8_scaled_mm_splitk16,
    }[args.split_k]
    activation = torch.randint(-128, 128, (1, K), device="cuda", dtype=torch.int8)
    weight = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)
    activation_scale = torch.tensor([[0.03125]], device="cuda", dtype=torch.float32)
    weight_scale = torch.linspace(0.001, 0.02, N, device="cuda", dtype=torch.float32).reshape(-1, 1)
    packed = op.hy3_w8a8_pack(weight)
    partial = torch.empty((16, N), device="cuda", dtype=torch.int32)
    tickets = torch.zeros((N // 16,), device="cuda", dtype=torch.int32)
    output = torch.empty((1, N), device="cuda", dtype=torch.bfloat16)

    def call():
        if args.persistent_workspace:
            op.hy3_w8a8_scaled_mm_splitk16_aresident_out(
                activation, packed, activation_scale, weight_scale,
                partial, tickets, output
            )
            return output
        return scaled_mm(activation, packed, activation_scale, weight_scale)

    for _ in range(args.warmup):
        output = call()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    all_samples = []
    round_medians = []
    for _ in range(args.rounds):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        for start, end in zip(starts, ends):
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        all_samples.extend(samples)
        round_medians.append(statistics.median(samples))

    p50_ms = statistics.median(all_samples)
    p95_ms = percentile(all_samples, 0.95)
    p50_s = p50_ms / 1e3
    weight_gbps = WEIGHT_BYTES / p50_s / 1e9
    total_gbps = TOTAL_BYTES / p50_s / 1e9
    cv = 100 * statistics.pstdev(round_medians) / statistics.mean(round_medians)
    print(f"p50: {p50_ms * 1e3:.3f} us")
    print(f"p95: {p95_ms * 1e3:.3f} us")
    print(f"round median CV: {cv:.2f}%")
    print(f"effective weight bandwidth (K*N/time): {weight_gbps:.2f} GB/s")
    print(f"effective total no-reuse bandwidth: {total_gbps:.2f} GB/s")
    print(f"output checksum: {output.float().sum().item():.6f}")


if __name__ == "__main__":
    main()
