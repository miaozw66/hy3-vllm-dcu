#!/usr/bin/env python3
"""Tail-question long-context benchmark for HY3 MTP vs baseline.

Reads a prompt-list JSON produced by gen_tail_prompts.py (each item has a
long-context prompt with a GSM8K question embedded at the tail, plus the
reference answer). For each concurrency level it:
  * snapshots /metrics spec_decode counters BEFORE and AFTER the batch,
    and reports acceptance rate / average acceptance length (MTP only),
  * runs concurrent streaming completions, rotating prompts so request i of
    the batch uses prompt i (i < concurrency),
  * records each request's prompt id + output text so the tail GSM8K answer
    can be scored offline.

Usage:
  python3 bench_tail_prompts.py \
      --prompt-list benchmark/tail_prompts/gsm8k_tail_8k.json \
      --max-tokens 2048 --concurrencies 1,2,4,8,16 --runs 1 \
      --output results_tail_mtp_8k.json
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cudagraph_bench import call_vllm_streaming  # noqa: E402


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


def delta_metrics(a: dict, b: dict) -> dict:
    if a is None or b is None:
        return {"drafts": 0.0, "accepted": 0.0}
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in METRICS_PATTERNS}
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="defaults to server's model")
    parser.add_argument("--prompt-list", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.prompt_list) as f:
        prompts = json.load(f)
    n_prompts = len(prompts)
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]

    # Detect model
    model = args.model
    try:
        resp = requests.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()["data"]
            if data:
                model = data[0]["id"]
    except Exception:
        pass
    print(f"endpoint={args.endpoint} model={model} prompts={n_prompts} "
          f"max_tokens={args.max_tokens} concurrencies={concurrencies} runs={args.runs}")

    all_results = []
    spec_by_conc = {}
    for concurrency in concurrencies:
        print(f"\n=== concurrency={concurrency} ===", flush=True)
        m_before = fetch_metrics(args.endpoint)
        batch_results = []
        for run_idx in range(args.runs):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:

                def _call(i):
                    r = call_vllm_streaming(
                        args.endpoint, model, prompts[i]["prompt"],
                        args.max_tokens, args.temperature, args.timeout,
                        args.ignore_eos,
                    )
                    return i, r

                futures = [executor.submit(_call, i) for i in range(concurrency)]
                run_results = []
                for future in as_completed(futures):
                    i, r = future.result()
                    r["run_index"] = run_idx
                    r["concurrency"] = concurrency
                    r["prompt_id"] = prompts[i]["gsm8k_index"]
                    r["answer_ref"] = prompts[i]["answer"]
                    r["prompt_tokens"] = prompts[i]["token_count"]
                    run_results.append(r)
            wall = time.perf_counter() - t0
            for r in run_results:
                r["batch_wall_time_s"] = round(wall, 3)
            batch_results.extend(run_results)
        m_after = fetch_metrics(args.endpoint)
        md = delta_metrics(m_before, m_after)

        ok_results = [r for r in batch_results if r["status"] == "ok"]
        total_out = sum(r["output_tokens"] for r in ok_results)
        total_in = sum(r["input_tokens"] for r in ok_results)
        wall = max((r["batch_wall_time_s"] for r in batch_results), default=1)
        batch_tput = total_out / wall if wall > 0 else 0
        ttfts = [r["ttft_s"] for r in ok_results]
        tpots = [r["avg_tpot_s"] for r in ok_results]

        # acceptance stats (MTP only)
        acc_rate = None
        avg_len = None
        if md["drafts"] > 0:
            acc_rate = md["accepted"] / md["drafts"]
            avg_len = 1 + acc_rate
        spec_by_conc[concurrency] = {
            "drafts": round(md["drafts"]),
            "accepted": round(md["accepted"]),
            "acc_rate": round(acc_rate, 4) if acc_rate is not None else None,
            "avg_len": round(avg_len, 4) if avg_len is not None else None,
        }

        print(f"  ok={len(ok_results)}/{len(batch_results)} "
              f"wall={wall:.1f}s batch_tput={batch_tput:.1f} tok/s "
              f"avg_ttft={statistics.mean(ttfts):.2f}s "
              f"avg_tpot={statistics.mean(tpots)*1000:.0f}ms")
        if md["drafts"] > 0:
            print(f"  spec: drafts={md['drafts']:.0f} accepted={md['accepted']:.0f} "
                  f"acc_rate={acc_rate:.3f} avg_len={avg_len:.3f}")

        all_results.extend(batch_results)

        # persist incrementally so a crash doesn't lose earlier concurrency levels
        snapshot = {
            "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "prompt_list": args.prompt_list,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "runs": args.runs,
            "ignore_eos": args.ignore_eos,
            "spec_decode": spec_by_conc,
            "requests": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # ---- acceptance-rate summary table ----
    print("\n=== ACCEPTANCE SUMMARY ===")
    print(f"{'conc':>5} {'drafts':>8} {'accepted':>8} {'acc_rate':>9} {'avg_len':>8}")
    # recompute per-concurrency from the final metrics deltas is impossible after
    # the fact; here we print what we logged above. For a per-level table we'd
    # need per-level deltas stored; see results dict in file.
    print("(per-concurrency deltas were printed inline above)")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
