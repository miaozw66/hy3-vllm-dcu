#!/usr/bin/env python3
"""vLLM 对齐 benchmark —— 对齐 SGLang MTP 表（'Hy3-Channel-INT8-w8a8+mtp213 L20'）口径。

SGLang 侧使用 `sglang.bench_serving --backend sglang`，dataset=random-ids，
参数 --random-input-len / --random-output-len / --num-prompts / --max-concurrency。
本脚本对每个 (input_len, output_len, concurrency) 组合，向 vLLM OpenAI 兼容端点
发送等量并发 random-id 流式请求，采集每个 SSE token 块，并按 SGLang 口径计算：

  TTFT  = 首个 token 块到达耗时
  TPOT  = (e2e_latency - ttft) / output_tokens   # 数值逐 token 口径，对齐 SGLang
  ITL   = 相邻 SSE 块间隔（真实 token 间隔，SGLang 另有 mean_itl_ms）
  e2e   = 请求总耗时

输出 JSON 同时保留两种口径（tpot_sglang_ms 与 tpot_raw_itl_ms），并生成与
SGLang 表同列名的 markdown 报告（mean_ttft_ms / mean_tpot_ms / mean_itl_ms /
mean_e2e_latency_ms / output_token_throughput_tok_s ...）。

用法：
  python3 benchmark/bench_hy3.py \
      --endpoint http://localhost:8000 \
      --matrix "1024:1024,2048:1024,7168:2048" \
      --concurrency 1,2,4,8,16 \
      --runs 1 \
      --tag align_d3 \
      --out /tmp/sglang_align_d3.json

  # 复现 SGLang 表 MTP 关键行（每行单轮）
  python3 benchmark/bench_hy3.py --matrix "1024:1024,7168:2048,32768:1024,65536:1024,131072:1024"
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from cudagraph_bench import call_vllm_streaming  # noqa: E402

RANDOM_ID_RANGE = (1, 120000)
DEFAULT_MATRIX = "1024:1024,2048:1024,2048:2048,4096:1024,7168:2048,16384:1024,32768:1024,65536:1024,131072:1024"


def parse_matrix(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def sglang_metrics(r: dict) -> dict:
    """从 call_vllm_streaming 结果对齐 SGLang bench_serving 指标。

    SGLang 口径（已用 xlsx 1K_1K_c1 验证）：
      mean_tpot_ms = (e2e_ms - ttft_ms) / output_tokens
      mean_itl_ms = 相邻 token 块平均间隔（原始 inter-token latency）
    """
    e2e_ms = r["total_time_s"] * 1000.0
    ttft_ms = r["ttft_s"] * 1000.0
    out_tokens = r["output_tokens"]
    tpot_sglang_ms = (e2e_ms - ttft_ms) / out_tokens if out_tokens > 0 else 0.0
    raw_itl_ms = r["avg_tpot_s"] * 1000.0  # cudagraph_bench 的 avg_tpot 是真实块间隔
    return {
        "ttft_ms": ttft_ms,
        "tpot_sglang_ms": tpot_sglang_ms,
        "tpot_raw_itl_ms": raw_itl_ms,
        "itl_ms": raw_itl_ms,
        "e2e_ms": e2e_ms,
        "input_tokens": r["input_tokens"],
        "output_tokens": out_tokens,
        "status": r["status"],
        "total_time_s": r["total_time_s"],
    }


def run_cell(endpoint, model, in_len, out_len, conc, seed, timeout):
    rng = random.Random(seed)
    prompts = [
        [rng.randrange(*RANDOM_ID_RANGE) for _ in range(in_len)]
        for _ in range(conc)
    ]

    def worker(pidx):
        r = call_vllm_streaming(endpoint, model, prompts[pidx], out_len, 0.0,
                                timeout, True)
        r["prompt_id"] = pidx
        return sglang_metrics(r)

    t0 = time.time()
    ok = 0
    results = []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(worker, i) for i in range(conc)]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")

    # 聚合
    okr = [r for r in results if r["status"] == "ok"]
    total_out = sum(r["output_tokens"] for r in okr)
    agg = {
        "input_len": in_len,
        "output_len": out_len,
        "concurrency": conc,
        "num_prompts": conc,
        "benchmark_duration_s": round(wall, 3),
        "successful_requests": ok,
        "output_token_throughput_tok_s": round(total_out / wall, 3) if wall > 0 else 0,
        "mean_ttft_ms": round(statistics.mean([r["ttft_ms"] for r in okr]), 3) if okr else None,
        "mean_tpot_ms": round(statistics.mean([r["tpot_sglang_ms"] for r in okr]), 3) if okr else None,
        "mean_itl_ms": round(statistics.mean([r["itl_ms"] for r in okr]), 3) if okr else None,
        "mean_e2e_latency_ms": round(statistics.mean([r["e2e_ms"] for r in okr]), 3) if okr else None,
        "total_generated_tokens": total_out,
        "per_request": results,
    }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--matrix", default=DEFAULT_MATRIX)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--runs", type=int, default=1, help="每个 (len, conc) 组合重复轮数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="align")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    model = args.model
    try:
        resp = requests.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=10)
        if resp.status_code == 200 and resp.json()["data"]:
            model = resp.json()["data"][0]["id"]
    except Exception:
        pass

    matrix = parse_matrix(args.matrix)
    concs = [int(x.strip()) for x in args.concurrency.split(",")]

    print("=" * 88)
    print("  vLLM SGLang-aligned benchmark")
    print("=" * 88)
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Model:    {model}")
    print(f"  Matrix:   {matrix}")
    print(f"  Concurrency: {concs}")
    print(f"  Runs per cell: {args.runs}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"/tmp/sglang_align_{args.tag}_{timestamp}.json"

    cells = []
    for in_len, out_len in matrix:
        for conc in concs:
            for run in range(args.runs):
                seed = args.seed + in_len + out_len * 7 + conc * 13 + run * 1000
                print(f"[{args.tag}] cell in={in_len} out={out_len} conc={conc} "
                      f"run={run + 1}/{args.runs} ...", flush=True)
                try:
                    agg = run_cell(args.endpoint, model, in_len, out_len, conc,
                                   seed, args.timeout)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    agg = {"input_len": in_len, "output_len": out_len,
                           "concurrency": conc, "num_prompts": conc, "error": str(e),
                           "per_request": []}
                cells.append(agg)
                ok = agg.get("successful_requests", 0)
                print(f"  ok={ok} dur={agg.get('benchmark_duration_s')}s "
                      f"out_tput={agg.get('output_token_throughput_tok_s')} tok/s "
                      f"ttft={agg.get('mean_ttft_ms')}ms "
                      f"tpot(sglang)={agg.get('mean_tpot_ms')}ms "
                      f"e2e={agg.get('mean_e2e_latency_ms')}ms", flush=True)

    doc = {
        "tag": args.tag,
        "timestamp": timestamp,
        "endpoint": args.endpoint,
        "matrix": [[a, b] for a, b in matrix],
        "concurrency": concs,
        "runs": args.runs,
        "cells": cells,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"\n  wrote {out_path}")

    # Markdown 报告（对齐 SGLang 表列名）
    print("\n" + "#" * 88)
    print("#  SGLang-aligned report (tpot = (e2e-ttft)/out_tokens)")
    print("#" * 88)
    print("| input | output | conc | dur_s | ok | out_tput(tok/s) | mean_ttft_ms | mean_tpot_ms | mean_itl_ms | mean_e2e_ms |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        if "error" in c:
            print(f"| {c['input_len']} | {c['output_len']} | {c['concurrency']} "
                  f"| - | 0 | - | - | - | - | ERROR: {c['error'][:40]} |")
            continue
        print(f"| {c['input_len']} | {c['output_len']} | {c['concurrency']} "
              f"| {c['benchmark_duration_s']} | {c['successful_requests']} "
              f"| {c['output_token_throughput_tok_s']} "
              f"| {c['mean_ttft_ms']} | {c['mean_tpot_ms']} "
              f"| {c['mean_itl_ms']} | {c['mean_e2e_latency_ms']} |")
    print()
    print(f"  Full JSON: {out_path}")


if __name__ == "__main__":
    main()
