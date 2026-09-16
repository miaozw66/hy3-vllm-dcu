#!/usr/bin/env python3
"""Summarize RCCL A/B results from /tmp/rccl_ab/*.json snapshots.

Usage:
  python3 benchmark/rccl_ab_summary.py
Prints a comparison table of conc1 tpot (primary: decode latency) and conc16
throughput per config, plus the fold-change vs base.
"""
import glob
import json
import os


def load(path):
    d = json.load(open(path))
    rows = {r["max_concurrency"]: r for r in d["rows"]}
    return rows


def main():
    base = load("/tmp/rccl_ab/base.json")
    base_c1 = base[1]["mean_tpot_step_ms"]
    base_c16 = base[16]["output_token_throughput_tok_s"]

    print(f"{'config':<10} {'c1 tpot(ms)':>12} {'vs base':>8} "
          f"{'c16 tok/s':>10} {'vs base':>8}  {'c16 c1_ratio':>12}")
    print("-" * 70)
    print(f"{'base':<10} {base_c1:>12.1f} {'1.00x':>8} "
          f"{base_c16:>10.1f} {'1.00x':>8}  "
          f"{base_c16 / (1000 / base_c1):>10.2f}  (c16 tput / c1 single-stream)")
    for path in sorted(glob.glob("/tmp/rccl_ab/*.json")):
        name = os.path.basename(path)[:-5]
        if name == "base":
            continue
        rows = load(path)
        if 1 not in rows or 16 not in rows:
            print(f"{name:<10} MISSING rows {sorted(rows)}")
            continue
        c1 = rows[1]["mean_tpot_step_ms"]
        c16 = rows[16]["output_token_throughput_tok_s"]
        print(f"{name:<10} {c1:>12.1f} {c1 / base_c1:>7.2f}x "
              f"{c16:>10.1f} {c16 / base_c16:>7.2f}x")


if __name__ == "__main__":
    main()
