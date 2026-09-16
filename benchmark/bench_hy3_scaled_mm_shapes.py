#!/usr/bin/env python3
"""Benchmark exact HY3 TP scaled-mm shapes with comparable timing modes."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from vllm._custom_ops import cutlass_scaled_mm, scaled_int8_quant

DEFAULT_HY3_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 13312,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "expert_hidden_dim": 1536,
    "num_shared_experts": 1,
}


def derive_hy3_tp_cases(config: dict[str, Any], tp_size: int) -> list[dict[str, Any]]:
    hidden_size = config["hidden_size"]
    head_dim = config.get("head_dim", hidden_size // config["num_attention_heads"])
    num_q_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]
    expert_hidden_dim = config["expert_hidden_dim"]
    num_shared_experts = config.get("num_shared_experts", 1)

    if num_q_heads % tp_size:
        raise ValueError(f"num_attention_heads={num_q_heads} is not divisible by TP={tp_size}")
    if intermediate_size % tp_size:
        raise ValueError(f"intermediate_size={intermediate_size} is not divisible by TP={tp_size}")
    shared_intermediate = expert_hidden_dim * num_shared_experts
    if shared_intermediate % tp_size:
        raise ValueError(
            f"shared intermediate={shared_intermediate} is not divisible by TP={tp_size}"
        )

    local_q = num_q_heads // tp_size
    local_kv = max(1, num_kv_heads // tp_size)
    qkv_n = (local_q + 2 * local_kv) * head_dim
    attention_k = num_q_heads * head_dim // tp_size
    dense_local = intermediate_size // tp_size
    shared_local = shared_intermediate // tp_size

    return [
        {
            "name": "qkv_proj",
            "path": "attention",
            "k": hidden_size,
            "n": qkv_n,
            "derivation": "(Q_heads/TP + 2*max(1, KV_heads/TP))*head_dim",
        },
        {
            "name": "o_proj",
            "path": "attention",
            "k": attention_k,
            "n": hidden_size,
            "derivation": "Q_heads*head_dim/TP -> hidden_size",
        },
        {
            "name": "dense_gate_up_proj",
            "path": "dense_mlp",
            "k": hidden_size,
            "n": 2 * dense_local,
            "derivation": "hidden_size -> 2*intermediate_size/TP",
        },
        {
            "name": "dense_down_proj",
            "path": "dense_mlp",
            "k": dense_local,
            "n": hidden_size,
            "derivation": "intermediate_size/TP -> hidden_size",
        },
        {
            "name": "shared_mlp_gate_up_proj",
            "path": "shared_mlp",
            "k": hidden_size,
            "n": 2 * shared_local,
            "derivation": "hidden_size -> 2*shared_intermediate/TP",
        },
        {
            "name": "shared_mlp_down_proj",
            "path": "shared_mlp",
            "k": shared_local,
            "n": hidden_size,
            "derivation": "shared_intermediate/TP -> hidden_size",
        },
    ]


def _abi_case_name(record: dict[str, Any]) -> str:
    prefix = record.get("layer_prefix", "scaled_mm")
    if prefix.endswith("self_attn.qkv_proj") or "qkv" in prefix:
        return "qkv_proj"
    if prefix.endswith("self_attn.o_proj") or "o_proj" in prefix:
        return "o_proj"
    if prefix.endswith("mlp.gate_up_proj"):
        return (
            "dense_gate_up_proj"
            if ".layers.0." in prefix
            else "shared_mlp_gate_up_proj"
        )
    if prefix.endswith("mlp.down_proj"):
        return (
            "dense_down_proj"
            if ".layers.0." in prefix
            else "shared_mlp_down_proj"
        )
    return prefix


def _cases_from_abi_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = {}
    for record in records:
        if not all(field in record for field in ("m", "k", "n")):
            raise ValueError("ABI record must contain m, k, and n")
        name = _abi_case_name(record)
        key = (name, int(record["k"]), int(record["n"]))
        cases[key] = {
            "name": name,
            "path": "production_abi",
            "k": int(record["k"]),
            "n": int(record["n"]),
            "observed_m": int(record["m"]),
            "derivation": "production ABI dump",
            "abi_record": record,
        }
    return list(cases.values())


def load_cases(
    model_config: str | None, case_manifest: str | None, tp_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if case_manifest:
        manifest_path = Path(case_manifest)
        text = manifest_path.read_text()
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                cases = payload
                manifest_format = "case-list"
            elif "cases" in payload:
                cases = payload["cases"]
                manifest_format = "case-manifest"
            elif all(field in payload for field in ("m", "k", "n")):
                cases = _cases_from_abi_records([payload])
                manifest_format = "production-abi-json"
            else:
                raise ValueError("Manifest object must contain cases or ABI m/k/n fields")
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            cases = _cases_from_abi_records(records)
            manifest_format = "production-abi-jsonl"
        return cases, {
            "source": manifest_format,
            "path": str(manifest_path.resolve()),
        }

    if model_config:
        config_path = Path(model_config)
        config = json.loads(config_path.read_text())
        source = {"source": "model-config", "path": str(config_path.resolve())}
    else:
        config = DEFAULT_HY3_CONFIG.copy()
        source = {"source": "built-in-hy3-defaults", "warning": "verify against checkpoint"}
    return derive_hy3_tp_cases(config, tp_size), source


def tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_contiguous": tensor.is_contiguous(),
    }


def summarize(samples: list[float], round_medians: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    return {
        "samples": len(ordered),
        "p50_ms": median,
        "p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
        "mean_ms": statistics.mean(ordered),
        "mad_ms": statistics.median(abs(value - median) for value in ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "round_medians_ms": round_medians,
        "round_median_cv_pct": (
            100 * statistics.pstdev(round_medians) / statistics.mean(round_medians)
            if len(round_medians) > 1 and statistics.mean(round_medians)
            else 0.0
        ),
    }


def _event_samples(call: Callable[[], torch.Tensor], iterations: int, sync_each: bool):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    outputs = []
    for start, end in zip(starts, ends):
        start.record()
        outputs.append(call())
        end.record()
        if sync_each:
            end.synchronize()
    if not sync_each:
        torch.cuda.synchronize()
    return outputs[-1], [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _capture_graph(call: Callable[[], torch.Tensor], warmup: int):
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        output = None
        for _ in range(warmup):
            output = call()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def measure(
    call: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    rounds: int,
    timing_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize()

    graph = None
    capture_ms = None
    if timing_mode == "graph-replay":
        capture_start = torch.cuda.Event(enable_timing=True)
        capture_end = torch.cuda.Event(enable_timing=True)
        capture_start.record()
        graph, output = _capture_graph(call, warmup)
        capture_end.record()
        capture_end.synchronize()
        capture_ms = capture_start.elapsed_time(capture_end)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

    all_samples = []
    round_medians = []
    for _ in range(rounds):
        if graph is not None:
            _, samples = _event_samples(lambda: (graph.replay(), output)[1], iterations, False)
        else:
            output, samples = _event_samples(
                call, iterations, timing_mode == "sync"
            )
        all_samples.extend(samples)
        round_medians.append(statistics.median(samples))

    stats = summarize(all_samples, round_medians)
    stats.update(
        {
            "timing_mode": timing_mode,
            "sync_scope": "per-iteration" if timing_mode == "sync" else "after-batch",
            "capture_ms": capture_ms,
        }
    )
    return output, stats


def scaled_mm_reference(
    activation_q: torch.Tensor,
    weight_q: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.mm(activation_q.float(), weight_q.float())
    output = output * scale_a.float()
    output = output * scale_b.float().T
    return output.to(out_dtype)


def benchmark_case(
    case: dict[str, Any],
    m: int,
    warmup: int,
    iterations: int,
    rounds: int,
    timing_mode: str,
) -> dict[str, Any]:
    k, n = int(case["k"]), int(case["n"])
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight_source = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    weight_q, weight_scale, _ = scaled_int8_quant(weight_source)
    weight_q = weight_q.t().contiguous()
    activation_q, activation_scale, _ = scaled_int8_quant(activation)

    def quant():
        return scaled_int8_quant(activation)[0]

    def gemm():
        return cutlass_scaled_mm(
            activation_q,
            weight_q,
            activation_scale,
            weight_scale,
            torch.bfloat16,
        )

    def full():
        current_q, current_scale, _ = scaled_int8_quant(activation)
        return cutlass_scaled_mm(
            current_q, weight_q, current_scale, weight_scale, torch.bfloat16
        )

    quant_output, quant_stats = measure(
        quant, warmup, iterations, rounds, timing_mode
    )
    gemm_output, gemm_stats = measure(
        gemm, warmup, iterations, rounds, timing_mode
    )
    full_output, full_stats = measure(
        full, warmup, iterations, rounds, timing_mode
    )

    reference = scaled_mm_reference(
        activation_q, weight_q, activation_scale, weight_scale, torch.bfloat16
    )
    error = (gemm_output.float() - reference.float()).abs()
    operations = 2 * m * n * k
    gemm_tops = operations / (gemm_stats["p50_ms"] * 1e9)
    full_tops = operations / (full_stats["p50_ms"] * 1e9)

    return {
        **case,
        "m": m,
        "operations": operations,
        "abi": {
            "activation": tensor_metadata(activation),
            "activation_q": tensor_metadata(activation_q),
            "activation_scale": tensor_metadata(activation_scale),
            "weight_source": tensor_metadata(weight_source),
            "weight_q_kernel": tensor_metadata(weight_q),
            "weight_scale": tensor_metadata(weight_scale),
            "quant_output": tensor_metadata(quant_output),
            "output": tensor_metadata(gemm_output),
            "bias": None,
        },
        "dispatch": {
            "entrypoint": "vllm._custom_ops.cutlass_scaled_mm",
            "expected_rocm_backend": "triton",
            "vllm_rocm_use_aiter": os.getenv("VLLM_ROCM_USE_AITER"),
        },
        "quant": quant_stats,
        "gemm": gemm_stats,
        "quant_gemm": full_stats,
        "gemm_tops_p50": gemm_tops,
        "quant_gemm_effective_tops_p50": full_tops,
        "correctness": {
            "max_abs_error_vs_independent_reference": error.max().item(),
            "mean_abs_error_vs_independent_reference": error.mean().item(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--timing-mode",
        choices=("sync", "batched", "graph-replay"),
        default="sync",
    )
    parser.add_argument("--model-config")
    parser.add_argument("--case-manifest")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--m-values", default="1,64")
    parser.add_argument("--case", action="append")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    cases, case_source = load_cases(args.model_config, args.case_manifest, args.tp_size)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["name"] in selected]
    if not cases:
        raise ValueError("No benchmark cases selected")

    m_values = [int(value) for value in args.m_values.split(",")]
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = []
    for m in m_values:
        for case in cases:
            result = benchmark_case(
                case,
                m,
                args.warmup,
                args.iterations,
                args.rounds,
                args.timing_mode,
            )
            results.append(result)
            print(json.dumps(result), flush=True)

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
        "script_sha256": script_hash,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
