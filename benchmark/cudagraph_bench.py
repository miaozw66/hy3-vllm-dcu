#!/usr/bin/env python3
"""
CUDA Graph Benchmark for HY3 80-Layer Model on vLLM.

Tests CUDA-graph-enabled vLLM serving performance with a consistent Chinese prompt
requesting the full text of "满江红" (Man Jiang Hong) by Yue Fei.

Metrics collected:
  - TTFT (Time To First Token): latency from request to first token
  - TPOT (Time Per Output Token): average inter-token interval during decode
  - Decode throughput (tokens/s): tokens generated per second during decode phase
  - Total throughput (tokens/s): end-to-end token rate (prefill + decode)
  - Prefill time, total decode time, input/output token counts

Usage:
  python3 cudagraph_bench.py --endpoint http://localhost:8000
  python3 cudagraph_bench.py --endpoint http://localhost:8000 --concurrencies 1,4,8 --runs 3
"""

import argparse
import os
import json
import sys
import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# ── Configuration ──────────────────────────────────────────────────────────────
PROMPT = "输出岳飞满江红全文，怒发冲冠那篇"
MAX_TOKENS = 1024
TEMPERATURE = 0.0
MODEL_NAME = "hy3"


def call_vllm_streaming(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 600,
) -> dict:
    """Send a streaming completion request and measure timing metrics.

    Returns a dict with:
        status, text, ttft_s, prefill_time_s, total_decode_time_s,
        input_tokens, output_tokens, decode_throughput, total_throughput,
        token_times (list of (timestamp, text_delta) for detailed analysis),
        total_time_s
    """
    url = f"{endpoint.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    ttft = None
    token_times = []  # list of (timestamp, text_delta)
    full_text = ""
    input_tokens = 0
    output_tokens_from_usage = 0
    ttft_token_time = None  # timestamp when first token text arrived

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return {"status": "error", "http_status": resp.status_code, "text": resp.text[:500]}

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                choices = data.get("choices", [])
                if choices:
                    text_delta = choices[0].get("text", "")
                    if text_delta:
                        if ttft is None:
                            ttft = now - t_start
                            ttft_token_time = now
                        token_times.append((now, text_delta))
                        full_text += text_delta

                usage = data.get("usage", {})
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens_from_usage = usage.get("completion_tokens", output_tokens_from_usage)

    except requests.Timeout:
        return {"status": "timeout", "text": full_text}
    except requests.ConnectionError as e:
        return {"status": "connection_error", "text": str(e)}
    except Exception as e:
        return {"status": "error", "text": str(e)}

    t_end = time.perf_counter()
    total_time = t_end - t_start
    output_tokens = len(token_times)
    num_tokens = output_tokens_from_usage if output_tokens_from_usage > 0 else output_tokens

    # Compute prefill time and decode time
    # prefill = TTFT (time from request to first token)
    # decode = time from first token to last token
    if ttft is not None and len(token_times) >= 1:
        prefill_time = ttft
        t_last_token = token_times[-1][0]
        # We define decode time as: time from request start to last token minus prefill time
        # But more accurately: decode_time = time from first token to last token
        decode_time = t_last_token - ttft_token_time if ttft_token_time else 0
    else:
        prefill_time = total_time
        decode_time = 0

    # Throughput
    decode_throughput = output_tokens / decode_time if decode_time > 0 else 0
    total_throughput = (input_tokens + output_tokens) / total_time if total_time > 0 else 0

    # TPOT (Time Per Output Token): average time between consecutive decode tokens
    tpot_values = []
    if len(token_times) >= 2:
        for i in range(1, len(token_times)):
            interval = token_times[i][0] - token_times[i - 1][0]
            tpot_values.append(interval)
    avg_tpot = statistics.mean(tpot_values) if tpot_values else 0.0
    p50_tpot = statistics.median(tpot_values) if tpot_values else 0.0
    p95_tpot = 0.0
    p99_tpot = 0.0
    if len(tpot_values) >= 2:
        sorted_intervals = sorted(tpot_values)
        p95_idx = int(len(sorted_intervals) * 0.95)
        p99_idx = int(len(sorted_intervals) * 0.99)
        p95_tpot = sorted_intervals[min(p95_idx, len(sorted_intervals) - 1)]
        p99_tpot = sorted_intervals[min(p99_idx, len(sorted_intervals) - 1)]

    return {
        "status": "ok",
        "text": full_text.strip(),
        "ttft_s": ttft if ttft else total_time,
        "prefill_time_s": prefill_time,
        "total_decode_time_s": decode_time,
        "input_tokens": input_tokens,
        "output_tokens": num_tokens,
        "decode_throughput_tokens_per_s": round(decode_throughput, 3),
        "total_throughput_tokens_per_s": round(total_throughput, 3),
        "total_time_s": round(total_time, 3),
        "avg_tpot_s": round(avg_tpot, 6),
        "med_tpot_s": round(p50_tpot, 6),
        "p95_tpot_s": round(p95_tpot, 6),
        "p99_tpot_s": round(p99_tpot, 6),
        "num_token_intervals": len(tpot_values),
    }


def run_concurrent_benchmark(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    num_runs: int,
    timeout: int = 600,
) -> list[dict]:
    """Run benchmark at a given concurrency level for num_runs iterations.

    Returns list of per-request result dicts.
    """
    results = []

    for run_idx in range(num_runs):
        print(f"    Run {run_idx + 1}/{num_runs} (concurrency={concurrency}) ...", end=" ", flush=True)

        t_batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    call_vllm_streaming,
                    endpoint, model, prompt, max_tokens, temperature, timeout
                )
                for _ in range(concurrency)
            ]
            batch_results = []
            for future in as_completed(futures):
                batch_results.append(future.result())

        t_batch_end = time.perf_counter()
        batch_wall_time = t_batch_end - t_batch_start

        # Annotate with batch info
        ok_results = [r for r in batch_results if r["status"] == "ok"]
        total_output_tokens = sum(r["output_tokens"] for r in ok_results)
        total_input_tokens = sum(r["input_tokens"] for r in ok_results)

        for r in batch_results:
            r["run_index"] = run_idx
            r["concurrency"] = concurrency
            r["batch_wall_time_s"] = round(batch_wall_time, 3)

        results.extend(batch_results)

        if ok_results:
            avg_ttft = statistics.mean([r["ttft_s"] for r in ok_results])
            # Batch throughput = total output tokens across all concurrent reqs / batch wall time
            batch_decode_tput = total_output_tokens / batch_wall_time if batch_wall_time > 0 else 0
            print(
                f"OK. avg_ttft={avg_ttft:.2f}s, "
                f"total_out={total_output_tokens}, "
                f"batch_decode_tput={batch_decode_tput:.1f} tok/s"
            )
        else:
            failed = [r for r in batch_results if r["status"] != "ok"]
            print(f"WARN: {len(failed)} failures")

    return results


def print_summary(all_results: list[dict], args):
    """Print a human-readable summary of benchmark results."""
    print("\n" + "=" * 100)
    print("  CUDA GRAPH BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"  Endpoint:   {args.endpoint}")
    print(f"  Prompt:     {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Runs:       {args.runs}")
    print(f"  Concurrencies: {args.concurrencies}")
    print()

    # Group by concurrency
    by_conc = {}
    for r in all_results:
        c = r.get("concurrency", 1)
        if c not in by_conc:
            by_conc[c] = []
        by_conc[c].append(r)

    for concurrency in sorted(by_conc.keys()):
        results = by_conc[concurrency]
        ok_results = [r for r in results if r["status"] == "ok"]
        error_results = [r for r in results if r["status"] != "ok"]

        if not ok_results:
            print(f"  Concurrency {concurrency}: ALL {len(error_results)} requests FAILED")
            continue

        # Aggregate
        ttfts = [r["ttft_s"] for r in ok_results]
        prefill_times = [r["prefill_time_s"] for r in ok_results]
        decode_times = [r["total_decode_time_s"] for r in ok_results]
        output_tokens_list = [r["output_tokens"] for r in ok_results]
        input_tokens_list = [r["input_tokens"] for r in ok_results]
        decode_tputs = [r["decode_throughput_tokens_per_s"] for r in ok_results]
        total_tputs = [r["total_throughput_tokens_per_s"] for r in ok_results]
        tpot_values = [r["avg_tpot_s"] for r in ok_results]

        # Batch-level (average across runs)
        # For each run, compute batch_decode_tput as sum(output_tokens)/batch_wall_time
        batch_decode_tputs = []
        for run_idx in range(args.runs):
            run_results = [r for r in ok_results if r.get("run_index") == run_idx]
            if run_results:
                wall = run_results[0].get("batch_wall_time_s", 1)
                total_out = sum(r["output_tokens"] for r in run_results)
                if wall > 0:
                    batch_decode_tputs.append(total_out / wall)

        print(f"{'─' * 100}")
        print(f"  CONCURRENCY = {concurrency}")
        print(f"  {'─' * 50}")
        print(f"  Successful requests: {len(ok_results)} / {len(results)}")
        if error_results:
            print(f"  Failed requests:     {len(error_results)}")
        print()

        # Per-request averages
        print(f"  ── Per-Request Metrics (avg across {len(ok_results)} requests) ──")
        print(f"  {'TTFT (s)':<25s} {statistics.mean(ttfts):>10.3f}  (min={min(ttfts):.3f}, max={max(ttfts):.3f})")
        print(f"  {'Prefill time (s)':<25s} {statistics.mean(prefill_times):>10.3f}")
        print(f"  {'Decode time (s)':<25s} {statistics.mean(decode_times):>10.3f}")
        print(f"  {'Input tokens':<25s} {statistics.mean(input_tokens_list):>10.1f}")
        print(f"  {'Output tokens':<25s} {statistics.mean(output_tokens_list):>10.1f}")
        print(f"  {'TPOT avg (ms)':<25s} {statistics.mean(tpot_values) * 1000:>10.2f}  (per-token decode interval)")
        print(f"  {'Decode throughput':<25s} {statistics.mean(decode_tputs):>10.1f} tok/s (per-request)")
        print(f"  {'Total throughput':<25s} {statistics.mean(total_tputs):>10.1f} tok/s (per-request)")
        print()

        # Aggregate batch throughput
        if batch_decode_tputs:
            print(f"  ── Aggregate Batch Throughput (concurrency={concurrency}) ──")
            print(f"  {'Batch decode tput':<25s} {statistics.mean(batch_decode_tputs):>10.1f} tok/s")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="CUDA Graph Benchmark for HY3 80-Layer Model on vLLM"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:8000",
        help="vLLM API endpoint"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="Model name"
    )
    parser.add_argument(
        "--prompt", default=PROMPT,
        help="Prompt to send"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per concurrency level"
    )
    parser.add_argument(
        "--concurrencies", default="1,4,8",
        help="Comma-separated list of concurrency levels to test"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (default: cudagraph_results_<timestamp>.json)"
    )
    args = parser.parse_args()

    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]

    print("=" * 80)
    print("  CUDA GRAPH BENCHMARK — HY3 80-Layer Model")
    print("=" * 80)
    print(f"  Endpoint:      {args.endpoint}")
    print(f"  Model:         {args.model}")
    print(f"  Prompt:        {args.prompt}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Runs per conc: {args.runs}")
    print(f"  Concurrencies: {concurrencies}")
    print()

    # Check server health
    print("─" * 80)
    print("  Checking server health...")
    try:
        resp = requests.get(f"{args.endpoint.rstrip('/')}/health", timeout=10)
        if resp.status_code == 200:
            print("  Server is healthy.")
        else:
            print(f"  WARNING: Server returned HTTP {resp.status_code}")
    except requests.ConnectionError:
        print("  ERROR: Cannot connect to server. Is it running?")
        sys.exit(1)

    # Detect model
    try:
        resp = requests.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=5)
        if resp.status_code == 200:
            models_data = resp.json()
            if models_data.get("data"):
                detected_model = models_data["data"][0]["id"]
                max_len = models_data["data"][0].get("max_model_len", "?")
                print(f"  Detected model: {detected_model} (max_model_len={max_len})")
                args.model = detected_model
    except Exception:
        pass
    print()

    # ── Run benchmarks ──────────────────────────────────────────────────────────
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for concurrency in concurrencies:
        print(f"{'─' * 80}")
        print(f"  Testing concurrency = {concurrency}")
        print(f"{'─' * 80}")

        results = run_concurrent_benchmark(
            endpoint=args.endpoint,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=concurrency,
            num_runs=args.runs,
            timeout=args.timeout,
        )
        all_results.extend(results)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print_summary(all_results, args)

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = args.output or os.path.join(
        script_dir, f"cudagraph_results_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Detailed results saved to: {output_file}")

    # ── Sample output ──────────────────────────────────────────────────────────
    ok_results = [r for r in all_results if r["status"] == "ok"]
    if ok_results:
        print(f"\n{'─' * 80}")
        print(f"  Sample output (first successful request):")
        print(f"{'─' * 80}")
        sample = ok_results[0]
        print(f"  {sample['text'][:500]}")
        if len(sample['text']) > 500:
            print(f"  ... ({len(sample['text'])} chars total)")
        print()


if __name__ == "__main__":
    main()
