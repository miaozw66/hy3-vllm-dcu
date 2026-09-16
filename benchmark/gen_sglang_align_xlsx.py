#!/usr/bin/env python3
"""把 bench_hy3.py 的 JSON 结果生成为与 SGLang 表同格式的 xlsx。

对齐 sheet 'Hy3-Channel-INT8-w8a8+mtp213 L20 ' 的 37 列格式：
  R1  = 启动命令（对齐 SGLang serve 那行）
  R2  = 运行参数摘要
  R3  = 37 列表头
  R4+ = 每行一个 (input_len, output_len, concurrency) 组合

用法：
  python3 benchmark/gen_sglang_align_xlsx.py \
      --json benchmark/results_service_measurements/sglang_align_d3_use_*.json \
      --out /path/out.xlsx \
      --server-cmd "python3 -m vllm.entrypoints.openai.api_server ..." \
      --log-file /home/hy3-vllm-dcu/logs/vllm_tp8_sglang_align_d3_use_0828_1554.log
"""
import argparse
import json
import statistics

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

HEADERS = [
    "input_len", "output_len", "max_concurrency", "num_prompts",
    "traffic_request_rate", "successful_requests", "benchmark_duration_s",
    "request_throughput_req_s", "input_token_throughput_tok_s",
    "output_token_throughput_tok_s", "peak_output_token_throughput_tok_s",
    "total_token_throughput_tok_s", "mean_ttft_ms", "mean_tpot_ms",
    "total_input_tokens", "total_input_text_tokens", "total_generated_tokens",
    "total_generated_tokens_retokenized", "peak_concurrent_requests",
    "concurrency", "mean_e2e_latency_ms", "median_e2e_latency_ms",
    "p90_e2e_latency_ms", "p99_e2e_latency_ms", "mean_ttft_ms",
    "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
    "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p95_itl_ms",
    "p99_itl_ms", "max_itl_ms", "log_file", "command",
]


def pct(data, p):
    if not data:
        return None
    return float(np.percentile(data, p))


def r3(x):
    return None if x is None else round(x, 3)


def build_rows(doc, log_file, server_cmd, bench_cmd):
    rows = []
    for cell in doc["cells"]:
        if "error" in cell:
            rows.append({"input_len": cell.get("input_len"),
                         "output_len": cell.get("output_len"),
                         "max_concurrency": cell.get("concurrency"),
                         "successful_requests": 0,
                         "log_file": f"ERROR: {cell['error']}"})
            continue
        pr = [r for r in cell.get("per_request", []) if r.get("status") == "ok"]
        dur = cell.get("benchmark_duration_s") or 0.0
        succ = cell.get("successful_requests", 0)
        in_tok = sum(r.get("input_tokens", 0) for r in pr)
        out_tok = sum(r.get("output_tokens", 0) for r in pr)
        e2e = [r.get("e2e_ms", 0) for r in pr]
        ttft = [r.get("ttft_ms", 0) for r in pr]
        tpot = [r.get("tpot_sglang_ms", 0) for r in pr]
        itl = [r.get("itl_ms", 0) for r in pr]
        mean_e2e = statistics.mean(e2e) if e2e else None
        row = {
            "input_len": cell.get("input_len"),
            "output_len": cell.get("output_len"),
            "max_concurrency": cell.get("concurrency"),
            "num_prompts": cell.get("num_prompts"),
            "traffic_request_rate": "inf",
            "successful_requests": succ,
            "benchmark_duration_s": r3(dur),
            "request_throughput_req_s": r3(succ / dur) if dur else None,
            "input_token_throughput_tok_s": r3(in_tok / dur) if dur else None,
            "output_token_throughput_tok_s": r3(out_tok / dur) if dur else None,
            "peak_output_token_throughput_tok_s": "N/A",
            "total_token_throughput_tok_s": r3((in_tok + out_tok) / dur) if dur else None,
            "mean_ttft_ms": r3(statistics.mean(ttft)) if ttft else None,
            "mean_tpot_ms": r3(statistics.mean(tpot)) if tpot else None,
            "total_input_tokens": in_tok,
            "total_input_text_tokens": in_tok,
            "total_generated_tokens": out_tok,
            "total_generated_tokens_retokenized": out_tok,
            "peak_concurrent_requests": cell.get("concurrency"),
            "concurrency": cell.get("concurrency"),
            "mean_e2e_latency_ms": r3(mean_e2e),
            "median_e2e_latency_ms": r3(statistics.median(e2e)) if e2e else None,
            "p90_e2e_latency_ms": r3(pct(e2e, 90)),
            "p99_e2e_latency_ms": r3(pct(e2e, 99)),
            "mean_ttft_ms_": r3(statistics.mean(ttft)) if ttft else None,
            "median_ttft_ms": r3(statistics.median(ttft)) if ttft else None,
            "p99_ttft_ms": r3(pct(ttft, 99)),
            "mean_tpot_ms_": r3(statistics.mean(tpot)) if tpot else None,
            "median_tpot_ms": r3(statistics.median(tpot)) if tpot else None,
            "p99_tpot_ms": r3(pct(tpot, 99)),
            "mean_itl_ms": r3(statistics.mean(itl)) if itl else None,
            "median_itl_ms": r3(statistics.median(itl)) if itl else None,
            "p95_itl_ms": r3(pct(itl, 95)),
            "p99_itl_ms": r3(pct(itl, 99)),
            "max_itl_ms": r3(max(itl)) if itl else None,
            "log_file": log_file,
            "command": bench_cmd,
        }
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-cmd", default="")
    ap.add_argument("--log-file", default="")
    ap.add_argument("--bench-cmd", default="")
    ap.add_argument("--spec-note", default="MTP num_speculative_tokens=3")
    args = ap.parse_args()

    with open(args.json) as f:
        doc = json.load(f)

    wb = Workbook()
    ws = wb.active
    ws.title = "vLLM Hy3-Channel-INT8-w8a8+mtp213"
    ws.append([args.server_cmd])
    ws.append([("vLLM v0.18.1+das.dtk2604.hy3, TP=8, -O1 CUDA Graph, "
                f"{args.spec_note}, zth W8A8 use, "
                f"tag={doc.get('tag')}, timestamp={doc.get('timestamp')}")])
    ws.append(HEADERS)

    rows = build_rows(doc, args.log_file, args.server_cmd, args.bench_cmd)
    for row in rows:
        ws.append([row.get(h) for h in HEADERS])

    # 样式：表头加粗
    bold = Font(bold=True)
    for c in range(1, len(HEADERS) + 1):
        ws.cell(3, c).font = bold
    ws.freeze_panes = "A4"
    wb.save(args.out)
    print(f"wrote {args.out}  ({len(rows)} data rows, {len(HEADERS)} cols)")


if __name__ == "__main__":
    main()
