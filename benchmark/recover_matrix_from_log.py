#!/usr/bin/env python3
"""从 bench_hy3.py 的运行日志恢复已完成 cell，生成完整矩阵 xlsx。

bench 中途被 kill 时 JSON 不会写出（脚本最后统一 dump），但每个完成的 cell
会打印一行聚合数据。本脚本解析日志恢复这些 cell，并生成与
gen_sglang_align_xlsx_full.py 相同的 37 列 xlsx：已跑填数据，未跑矩阵留空。

用法：
  python3 benchmark/recover_matrix_from_log.py \
      --log logs/mtp_matrix_16k.log \
      --matrix "1024:1024,...,131072:1024" \
      --concurrency 1,2,4,8,16 \
      --out /path/out.xlsx
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_sglang_align_xlsx import HEADERS  # noqa: E402

RE_CELL = re.compile(r"^\[[^\]]+\] cell in=(\d+) out=(\d+) conc=(\d+)")
RE_RES = re.compile(
    r"ok=(\d+) dur=([\d.]+)s out_tput=([\d.]+) tok/s ttft=([\d.]+)ms "
    r"tpot\(sglang\)=([\d.]+)ms e2e=([\d.]+)ms")


def parse_log(path):
    cells = []
    cur = None
    for line in open(path):
        m = RE_CELL.match(line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m2 = RE_RES.search(line)
        if m2 and cur:
            cells.append({
                "input_len": cur[0], "output_len": cur[1], "concurrency": cur[2],
                "ok": int(m2.group(1)), "dur_s": float(m2.group(2)),
                "out_tput": float(m2.group(3)), "ttft_ms": float(m2.group(4)),
                "tpot_ms": float(m2.group(5)), "e2e_ms": float(m2.group(6)),
            })
            cur = None
    return cells


def r3(x):
    return None if x is None else round(x, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-cmd", default="")
    ap.add_argument("--log-file", default="")
    ap.add_argument("--bench-cmd", default="")
    ap.add_argument("--spec-note", default="MTP num_speculative_tokens=3")
    args = ap.parse_args()

    concs = [int(x) for x in args.concurrency.split(",")]
    matrix = []
    for part in args.matrix.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        matrix.append((int(a), int(b)))

    cells = parse_log(args.log)
    ran = {(c["input_len"], c["output_len"], c["concurrency"]) for c in cells}

    # 已跑行（日志恢复，itl 列无 per-request 数据留空）
    real_rows = []
    for c in cells:
        inl, outl, conc = c["input_len"], c["output_len"], c["concurrency"]
        ok, dur, tput, ttft, tpot, e2e = (c["ok"], c["dur_s"], c["out_tput"],
                                          c["ttft_ms"], c["tpot_ms"],
                                          c["e2e_ms"])
        in_tok = conc * inl
        out_tok = conc * outl
        row = {h: None for h in HEADERS}
        row.update({
            "input_len": inl, "output_len": outl, "max_concurrency": conc,
            "num_prompts": conc, "traffic_request_rate": "inf",
            "successful_requests": ok, "benchmark_duration_s": r3(dur),
            "request_throughput_req_s": r3(ok / dur) if dur else None,
            "input_token_throughput_tok_s": r3(in_tok / dur) if dur else None,
            "output_token_throughput_tok_s": r3(tput),
            "peak_output_token_throughput_tok_s": "N/A",
            "total_token_throughput_tok_s": r3((in_tok + out_tok) / dur) if dur else None,
            "mean_ttft_ms": r3(ttft), "mean_tpot_ms": r3(tpot),
            "total_input_tokens": in_tok, "total_input_text_tokens": in_tok,
            "total_generated_tokens": out_tok,
            "total_generated_tokens_retokenized": out_tok,
            "peak_concurrent_requests": conc, "concurrency": conc,
            "mean_e2e_latency_ms": r3(e2e),
            "mean_ttft_ms_": r3(ttft), "mean_tpot_ms_": r3(tpot),
            "log_file": args.log_file, "command": args.bench_cmd,
        })
        real_rows.append(row)

    # 未跑行：只填组合标识
    blank_rows = []
    for inl, outl in matrix:
        for conc in concs:
            if (inl, outl, conc) in ran:
                continue
            row = {h: None for h in HEADERS}
            row.update({"input_len": inl, "output_len": outl,
                        "max_concurrency": conc, "num_prompts": conc,
                        "traffic_request_rate": "inf", "concurrency": conc})
            blank_rows.append(row)

    rows = sorted(real_rows + blank_rows,
                  key=lambda r: (r["input_len"], r["output_len"],
                                 r["max_concurrency"]))

    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "vLLM Hy3-Channel-INT8-w8a8+mtp213"
    ws.append([args.server_cmd])
    ws.append([("vLLM v0.18.1+das.dtk2604.hy3, TP=8, CUDA Graph, "
                f"{args.spec_note}, zth W8A8 use, "
                f"(recovered from interrupted run log {args.log}), "
                "blank rows = matrix not run")])
    ws.append(HEADERS)
    for row in rows:
        ws.append([row.get(h) for h in HEADERS])
    bold = Font(bold=True)
    for c in range(1, len(HEADERS) + 1):
        ws.cell(3, c).font = bold
    ws.freeze_panes = "A4"
    wb.save(args.out)
    print(f"wrote {args.out}  ({len(rows)} rows: "
          f"{len(real_rows)} ran + {len(blank_rows)} blank)")

    # 同时落一份 JSON（聚合），方便后续
    jout = args.out.replace(".xlsx", ".json")
    with open(jout, "w") as f:
        json.dump({
            "tag": "mtp_zth_full_16k_recovered",
            "source_log": args.log,
            "matrix": matrix, "concurrency": concs,
            "cells": [{k: v for k, v in c.items()} for c in cells],
        }, f, indent=2, ensure_ascii=False)
    print(f"  also wrote {jout} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
