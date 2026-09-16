#!/usr/bin/env python3
"""Generate a random-ids benchmark report aligned with the sglang reference table.

Reads results_online_rids/{mtp,base}_cases.json (produced by bench_random_ids.py)
and writes a markdown report. Column set mirrors the Hy3 sheets in
K100AI 需求收集 & 基准性能 副本.xlsx: per (case, max_concurrency) row we show
  - TTFT (ms)                     = mean_ttft_ms
  - TPOT per-token (ms)           = mean_tpot_ms (table column; MTP already
                                    divided by avg acceptance length)
  - output_token_throughput (tok/s)
  - concurrency (avg, Little's law)
  - wall (s)
  - acceptance rate + avg_len (MTP only)
  - self-consistency check ratio  = output_tput / (concurrency * 1000/TPOT)
    (the same check applied to the sglang table: should be ~1 for short inputs,
     <1 when prefill dominates / actual concurrency < max)
"""
import json
import os
import statistics
from datetime import datetime

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_online_rids")


def load(name):
    with open(os.path.join(OUTDIR, name)) as f:
        return json.load(f)


def check_ratio(row):
    """Throughput vs concurrency*1000/TPOT (per-token) consistency check."""
    conc = row.get("concurrency")
    tpot = row.get("mean_tpot_ms")
    tput = row.get("output_token_throughput_tok_s")
    if not conc or not tpot or tpot <= 0:
        return None
    return tput / (conc * 1000.0 / tpot)


def main():
    out = []
    out.append("# HY3 random-ids 基准评测（对齐 sglang 参考表口径）")
    out.append("")
    out.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append("")
    out.append("**评测配置**：TP=8、max-model-len 32768、gpu-memory-utilization 0.90、"
               "--no-enable-prefix-caching、-O1 --no-async-scheduling、temperature=0、--ignore-eos。")
    out.append("")
    out.append("**输入**：random-ids —— 每条请求的 prompt 为随机 token-id 数组（非语义文本），"
               "精确等于 input_len；num_prompts = max_concurrency。case 为 1K~16K 六档，"
               "每档并发 1/2/4/8/16。")
    out.append("")
    out.append("**口径**：output_token_throughput = sum(usage output_tokens)/batch_wall；"
               "TPOT/词 = SSE 块间隔折算单 token（MTP 每块含 ~avg_len 个 token，除以 avg_len；"
               "baseline 每块 1 token）；并发 = Little's law（sum 请求时长/wall）；"
               "MTP 接受率 = /metrics `spec_decode_num_accepted_tokens_total` / `num_drafts_total` 快照差。")
    out.append("")
    out.append("**一致性校验**：`吞吐 / (并发 × 1000/TPOT)` —— 短输入应接近 1；"
               "长输入/高并发下 prefill 主导、实际并发不足时 <1（与 sglang 参考表同一现象）。")
    out.append("")

    data = {}
    for tag, label in [("mtp", "MTP"), ("base", "Baseline")]:
        p = os.path.join(OUTDIR, f"{tag}_cases.json")
        if not os.path.exists(p):
            out.append(f"## {label}\n\n(数据缺失: {p})\n")
            continue
        data[tag] = load(f"{tag}_cases.json")

        out.append(f"## {label}\n")
        out.append("| case | max_conc | TTFT(ms) | TPOT/词(ms) | 吞吐(tok/s) | "
                   "并发(avg) | wall(s) | 接受率 | avg_len | 校验比 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in data[tag]["rows"]:
            r = check_ratio(row)
            out.append(
                f"| {row['case']} | {row['max_concurrency']} | "
                f"{row['mean_ttft_ms']:.1f} | {row['mean_tpot_ms']:.1f} | "
                f"{row['output_token_throughput_tok_s']:.1f} | "
                f"{row['concurrency']:.2f} | {row['benchmark_duration_s']:.0f} | "
                f"{row['acc_rate'] if row['acc_rate'] is not None else '-'} | "
                f"{row['avg_len'] if row['avg_len'] is not None else '-'} | "
                f"{r:.3f}" if r is not None else "  -")
        out.append("")

    # speedup table (MTP vs baseline, same case/conc)
    if "mtp" in data and "base" in data:
        out.append("## MTP vs Baseline（MTP/Baseline，>1 表示 MTP 更快）\n")
        out.append("| case | max_conc | TTFT 加速 | TPOT/词 加速 | 吞吐加速 |")
        out.append("|---|---|---|---|---|")
        mtp_rows = {(r["case"], r["max_concurrency"]): r for r in data["mtp"]["rows"]}
        base_rows = {(r["case"], r["max_concurrency"]): r for r in data["base"]["rows"]}
        for key in sorted(set(mtp_rows) & set(base_rows)):
            m, b = mtp_rows[key], base_rows[key]
            ttft_s = b["mean_ttft_ms"] / m["mean_ttft_ms"] if m["mean_ttft_ms"] else 0
            tpot_s = b["mean_tpot_ms"] / m["mean_tpot_ms"] if m["mean_tpot_ms"] else 0
            tput_s = m["output_token_throughput_tok_s"] / b["output_token_throughput_tok_s"] if b["output_token_throughput_tok_s"] else 0
            out.append(f"| {key[0]} | {key[1]} | {ttft_s:.2f}x | {tpot_s:.2f}x | {tput_s:.2f}x |")
        out.append("")

    report_path = os.path.join(OUTDIR, "random_ids_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
