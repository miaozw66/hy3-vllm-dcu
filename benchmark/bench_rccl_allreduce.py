#!/usr/bin/env python3
"""Measure single-node TP8 RCCL all-reduce latency for decode-sized payloads."""
import argparse
import json
import os
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def percentile(values, q):
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def worker(rank, world_size, port, sizes, warmup, iterations, output, algorithm, protocol, channels):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_ALGO"] = algorithm
    os.environ["NCCL_PROTO"] = protocol
    os.environ["NCCL_MIN_NCHANNELS"] = str(channels)
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    results = []
    for size in sizes:
        if size % 2:
            raise ValueError(f"payload size must be even for bfloat16: {size}")
        tensor = torch.ones(size // 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(tensor)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        if rank == 0:
            p50 = percentile(samples, 0.5)
            p95 = percentile(samples, 0.95)
            results.append({
                "payload_bytes": size,
                "world_size": world_size,
                "dtype": "bfloat16",
                "p50_ms": p50,
                "p95_ms": p95,
                "mean_ms": statistics.mean(samples),
                "algorithmic_gbps_p50": size / (p50 * 1e6),
                "algorithmic_gbps_p95": size / (p95 * 1e6),
            })
    if rank == 0:
        with open(output, "w") as f:
            json.dump({"config": {"algorithm": algorithm, "protocol": protocol, "channels": channels}, "results": results}, f, indent=2)
        print(json.dumps(results, indent=2), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--sizes", default="8192,16384,32768,65536,262144,1048576")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--algorithm", default="Tree")
    parser.add_argument("--protocol", default="LL")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sizes = [int(size) for size in args.sizes.split(",")]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        worker,
        args=(
            args.world_size,
            port,
            sizes,
            args.warmup,
            args.iterations,
            args.output,
            args.algorithm,
            args.protocol,
            args.channels,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
