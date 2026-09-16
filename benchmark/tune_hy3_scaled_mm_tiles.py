#!/usr/bin/env python3
"""Sweep explicit Triton scaled-mm tiles for exact HY3 TP shapes."""

import argparse
import hashlib
import importlib.util
import json
from itertools import product
from pathlib import Path

import torch

from bench_hy3_scaled_mm_shapes import (
    load_cases,
    measure,
    scaled_mm_reference,
    tensor_metadata,
)
from vllm._custom_ops import scaled_int8_quant

TILES = (
    (16, 32, 64),
    (16, 32, 128),
    (16, 64, 128),
    (16, 64, 256),
    (32, 32, 128),
    (32, 64, 128),
    (32, 64, 256),
    (32, 128, 128),
    (32, 128, 256),
    (64, 64, 256),
    (64, 128, 256),
)


def load_source_kernel(path: str):
    spec = importlib.util.spec_from_file_location("hy3_tuned_scaled_mm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scaled-mm source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.triton_scaled_mm


def parse_tiles(values: list[str] | None):
    if not values:
        return TILES
    tiles = []
    for value in values:
        parts = tuple(int(part) for part in value.split(","))
        if len(parts) != 3:
            raise ValueError(f"Tile must be BM,BN,BK: {value}")
        tiles.append(parts)
    return tuple(tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--timing-mode",
        choices=("sync", "batched", "graph-replay"),
        default="graph-replay",
    )
    parser.add_argument("--model-config")
    parser.add_argument("--case-manifest")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--m-values", default="1")
    parser.add_argument("--case", action="append")
    parser.add_argument("--tile", action="append")
    parser.add_argument("--num-warps", action="append", type=int)
    parser.add_argument("--num-stages", action="append", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    scaled_mm = load_source_kernel(args.source)
    cases, case_source = load_cases(args.model_config, args.case_manifest, args.tp_size)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["name"] in selected]
    if not cases:
        raise ValueError("No tuning cases selected")

    tiles = parse_tiles(args.tile)
    warp_values = [None, *args.num_warps] if args.num_warps else [None]
    stage_values = [None, *args.num_stages] if args.num_stages else [None]
    m_values = [int(value) for value in args.m_values.split(",")]
    results = []
    for m in m_values:
        for case in cases:
            k, n = int(case["k"]), int(case["n"])
            activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            weight_source = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            weight_q, weight_scale, _ = scaled_int8_quant(weight_source)
            weight_q = weight_q.t().contiguous()
            activation_q, activation_scale, _ = scaled_int8_quant(activation)
            reference = scaled_mm_reference(
                activation_q,
                weight_q,
                activation_scale,
                weight_scale,
                torch.bfloat16,
            )

            heuristic_output, heuristic_stats = measure(
                lambda: scaled_mm(
                    activation_q,
                    weight_q,
                    activation_scale,
                    weight_scale,
                    torch.bfloat16,
                    use_heuristic=True,
                ),
                args.warmup,
                args.iterations,
                args.rounds,
                args.timing_mode,
            )
            heuristic_error = (
                heuristic_output.float() - reference.float()
            ).abs()

            for tile, num_warps, num_stages in product(
                tiles, warp_values, stage_values
            ):
                base = {
                    **case,
                    "m": m,
                    "tile": list(tile),
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "timing_mode": args.timing_mode,
                    "heuristic": heuristic_stats,
                    "heuristic_correctness": {
                        "max_abs_error_vs_independent_reference": heuristic_error.max().item(),
                        "mean_abs_error_vs_independent_reference": heuristic_error.mean().item(),
                    },
                    "abi": {
                        "activation_q": tensor_metadata(activation_q),
                        "activation_scale": tensor_metadata(activation_scale),
                        "weight_q_kernel": tensor_metadata(weight_q),
                        "weight_scale": tensor_metadata(weight_scale),
                        "output_dtype": str(torch.bfloat16),
                        "bias": None,
                    },
                }
                try:
                    output, stats = measure(
                        lambda tile=tile, num_warps=num_warps, num_stages=num_stages: scaled_mm(
                            activation_q,
                            weight_q,
                            activation_scale,
                            weight_scale,
                            torch.bfloat16,
                            block_size_m=tile[0],
                            block_size_n=tile[1],
                            block_size_k=tile[2],
                            use_heuristic=False,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        ),
                        args.warmup,
                        args.iterations,
                        args.rounds,
                        args.timing_mode,
                    )
                    error = (output.float() - reference.float()).abs()
                    result = {
                        **base,
                        "status": "ok",
                        "candidate": stats,
                        "speedup_vs_heuristic": (
                            heuristic_stats["p50_ms"] / stats["p50_ms"]
                        ),
                        "correctness": {
                            "max_abs_error_vs_independent_reference": error.max().item(),
                            "mean_abs_error_vs_independent_reference": error.mean().item(),
                            "max_abs_error_vs_heuristic": (
                                output.float() - heuristic_output.float()
                            )
                            .abs()
                            .max()
                            .item(),
                        },
                    }
                except Exception as error:
                    result = {
                        **base,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                results.append(result)
                print(json.dumps(result), flush=True)

    source_path = Path(args.source)
    payload = {
        "schema_version": 2,
        "device": torch.cuda.get_device_name(args.gpu),
        "device_capability": list(torch.cuda.get_device_capability(args.gpu)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "tp_size": args.tp_size,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "timing_mode": args.timing_mode,
        "case_source": case_source,
        "cases": cases,
        "tiles": [list(tile) for tile in tiles],
        "num_warps": warp_values,
        "num_stages": stage_values,
        "source": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "successful_candidates": sum(result["status"] == "ok" for result in results),
        "failed_candidates": sum(result["status"] == "error" for result in results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    if payload["successful_candidates"] == 0:
        raise SystemExit("All requested tile candidates failed")


if __name__ == "__main__":
    main()
