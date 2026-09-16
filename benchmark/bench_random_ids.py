#!/usr/bin/env python3
"""Random-ids online benchmark for HY3, aligned with the sglang reference table.

The reference table (K100AI 需求收集 & 基准性能 副本.xlsx, Hy3 sheets) was produced
with sglang bench_serving + `--dataset-name random-ids`: each row is one
(input_len, output_len) case at one max_concurrency, with num_prompts =
max_concurrency, input tokens drawn uniformly at random (no semantic content),
and metrics:
  - benchmark_duration_s          (wall time of the batch)
  - output_token_throughput_tok_s (= total_generated_tokens / duration)
  - request_throughput_req_s      (= successful_requests / duration)
  - input_token_throughput_tok_s  (= total_input_tokens / duration)
  - total_token_throughput_tok_s  (= (in+out) / duration)
  - mean_ttft_ms                  (time to first token, per request)
  - mean_tpot_ms                  (PER-TOKEN decode latency: SSE block interval
                                   / avg acceptance length for MTP, raw block
                                   interval for baseline)
  - concurrency                   (avg concurrent requests, Little's law)
  - peak_concurrent_requests      (= max_concurrency, since launched together)

We reproduce the same protocol against the vLLM API server by sending the prompt
as a list of token ids (exact input length, no tokenizer round-trip). MTP
acceptance / avg_len are additionally captured from /metrics counter deltas.

Usage:
  python3 benchmark/bench_random_ids.py \
      --cases "1024/1024,2048/1024,2048/2048,4096/1024,7168/2048,16384/1024" \
      --concurrencies 1,2,4,8,16 --runs 1 --ignore-eos \
      --output benchmark/results_online_rids/mtp_cases.json
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cudagraph_bench import call_vllm_streaming  # noqa: E402

VOCAB_SIZE = 120832
# HY3 special tokens are at the top of the vocab; avoid them so random input ids
# are plain tokens (bos=120000, pad=120002, eos=120025).
RANDOM_ID_RANGE = (1, 120000)

METRICS_PATTERNS = {
    "drafts": r'vllm:spec_decode_num_drafts_total\{[^}]*\} ([0-9.]+)',
    "accepted": r'vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} ([0-9.]+)',
}


def fetch_metrics(endpoint: str) -> dict | None:
    try:
        resp = requests.get(f"{endpoint.rstrip('/')}/metrics", timeout=10)
        if resp.status_code != 200:
            return None
        text = resp.text
        out = {}
        for key, pat in METRICS_PATTERNS.items():
            m = re.search(pat, text)
            out[key] = float(m.group(1)) if m else 0.0
        return out
    except Exception:
        return None


def delta_metrics(a, b) -> dict:
    if a is None or b is None:
        return {"drafts": 0.0, "accepted": 0.0}
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in METRICS_PATTERNS}


def random_token_ids(input_len: int, rng: random.Random) -> list[int]:
    lo, hi = RANDOM_ID_RANGE
    return [rng.randrange(lo, hi) for _ in range(input_len)]


def run_case(
    endpoint: str,
    model: str,
    input_len: int,
    output_len: int,
    conc: int,
    runs: int,
    ignore_eos: bool,
    temperature: float,
    timeout: int,
    seed: int,
) -> dict:
    """Run one (input_len, output_len) case at one concurrency level.

    num_prompts = conc (each request gets a distinct random prompt), matching
    the sglang reference table. Returns a per-concurrency aggregate row plus the
    raw per-request results.
    """
    rng = random.Random(seed)
    m_before = fetch_metrics(endpoint)

    batch_results = []
    batch_walls = []
    for run_idx in range(runs):
        prompts = [random_token_ids(input_len, rng) for _ in range(conc)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as executor:

            def _call(i):
                r = call_vllm_streaming(
                    endpoint, model, prompts[i],
                    output_len, temperature, timeout, ignore_eos,
                )
                return i, r

            futures = [executor.submit(_call, i) for i in range(conc)]
            run_results = []
            for future in as_completed(futures):
                i, r = future.result()
                r["run_index"] = run_idx
                r["concurrency"] = conc
                r["prompt_id"] = i
                run_results.append(r)
        wall = time.perf_counter() - t0
        batch_walls.append(wall)
        for r in run_results:
            r["batch_wall_time_s"] = round(wall, 3)
        batch_results.extend(run_results)

    m_after = fetch_metrics(endpoint)
    md = delta_metrics(m_before, m_after)
    acc_rate = avg_len = None
    if md["drafts"] > 0:
        acc_rate = md["accepted"] / md["drafts"]
        avg_len = 1 + acc_rate

    ok = [r for r in batch_results if r["status"] == "ok"]
    wall = max(batch_walls) if batch_walls else 1.0
    total_out = sum(r["output_tokens"] for r in ok)
    total_in = sum(r["input_tokens"] for r in ok)
    n_ok = len(ok)
    dur_secs = sum(r["total_time_s"] for r in ok)

    ttfts_ms = [r["ttft_s"] * 1000 for r in ok]
    # mean_tpot_ms: table column is PER-TOKEN. Our SSE block interval is the
    # server step time; for MTP each block carries ~avg_len tokens.
    tpot_step_ms = statistics.mean([r["avg_tpot_s"] * 1000 for r in ok]) if ok else 0.0
    tpot_pt_ms = tpot_step_ms / avg_len if avg_len else tpot_step_ms

    def _k(n):
        return f"{n//1024}K" if n >= 1024 else str(n)

    row = {
        "case": f"{_k(input_len)}/{_k(output_len)}",
        "input_len": input_len,
        "output_len": output_len,
        "max_concurrency": conc,
        "num_prompts": len(batch_results),
        "successful_requests": n_ok,
        "benchmark_duration_s": round(wall, 3),
        "request_throughput_req_s": round(n_ok / wall, 4) if wall else 0,
        "input_token_throughput_tok_s": round(total_in / wall, 2) if wall else 0,
        "output_token_throughput_tok_s": round(total_out / wall, 2) if wall else 0,
        "total_token_throughput_tok_s": round((total_in + total_out) / wall, 2) if wall else 0,
        "mean_ttft_ms": round(statistics.mean(ttfts_ms), 2) if ok else None,
        "mean_tpot_step_ms": round(tpot_step_ms, 3),
        "mean_tpot_ms": round(tpot_pt_ms, 3),          # per-token (table column)
        "mean_e2e_latency_ms": round(dur_secs / n_ok * 1000, 2) if n_ok else None,
        "concurrency": round(dur_secs / wall, 3) if wall else None,  # Little's law
        "peak_concurrent_requests": conc,
        "drafts": round(md["drafts"]),
        "accepted": round(md["accepted"]),
        "acc_rate": round(acc_rate, 4) if acc_rate is not None else None,
        "avg_len": round(avg_len, 4) if avg_len is not None else None,
        "total_generated_tokens": total_out,
    }
    return row, batch_results


def parse_cases(cases_str: str):
    out = []
    for tok in cases_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        il, ol = tok.split("/")
        out.append((int(il), int(ol)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--cases",
        default="1024/1024,2048/1024,2048/2048,4096/1024,7168/2048,16384/1024",
        help="comma-separated input_len/output_len pairs, e.g. 1024/1024,16384/1024",
    )
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases = parse_cases(args.cases)
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]

    model = args.model
    try:
        resp = requests.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()["data"]
            if data:
                model = data[0]["id"]
    except Exception:
        pass

    print(f"endpoint={args.endpoint} model={model} cases={cases} "
          f"concurrencies={concurrencies} runs={args.runs} "
          f"ignore_eos={args.ignore_eos} seed={args.seed}")

    all_rows = []
    all_requests = []
    for (ilen, olen) in cases:
        for conc in concurrencies:
            def _k(n):
                return f"{n//1024}K" if n >= 1024 else str(n)
            print(f"\n=== case {_k(ilen)}/{_k(olen)}  conc={conc} ===", flush=True)
            row, reqs = run_case(
                args.endpoint, model, ilen, olen, conc,
                args.runs, args.ignore_eos, args.temperature, args.timeout, args.seed,
            )
            all_rows.append(row)
            all_requests.extend(reqs)
            print(f"  ok={row['successful_requests']} wall={row['benchmark_duration_s']:.1f}s "
                  f"out_tput={row['output_token_throughput_tok_s']} tok/s "
                  f"ttft={row['mean_ttft_ms']}ms tpot={row['mean_tpot_ms']}ms "
                  f"conc_avg={row['concurrency']} "
                  f"acc_rate={row['acc_rate']} avg_len={row['avg_len']}")

            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            snapshot = {
                "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "cases": cases,
                "concurrencies": concurrencies,
                "runs": args.runs,
                "temperature": args.temperature,
                "ignore_eos": args.ignore_eos,
                "seed": args.seed,
                "rows": all_rows,
                "requests": all_requests,
            }
            with open(args.output, "w") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
