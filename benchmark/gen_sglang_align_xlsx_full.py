#!/usr/bin/env python3
"""把 bench_hy3.py 的 JSON 结果生成为 xlsx，保留完整矩阵（含未跑组合）。

与 gen_sglang_align_xlsx.py 同格式（37 列），但 --matrix 给出完整 9 组合时，
未跑的 (input_len, output_len, concurrency) 也插入一行，只填组合标识
（input/output/concurrency/num_prompts），其余数据列留空，避免误读为 0 成绩。

用法：
  python3 benchmark/gen_sglang_align_xlsx_full.py \
      --json benchmark/results_service_measurements/sglang_align_mtp_zth_full_16k_*.json \
      --matrix "1024:1024,2048:1024,2048:2048,4096:1024,7168:2048,16384:1024,32768:1024,65536:1024,131072:1024" \
      --concurrency 1,2,4,8,16 \
      --out /path/out.xlsx \
      --server-cmd "python3 -m vllm.entrypoints.openai.api_server ..."
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_sglang_align_xlsx import HEADERS, build_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrix", required=True,
                    help="完整矩阵，未跑组合留空行，如 1024:1024,...,131072:1024")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--server-cmd", default="")
    ap.add_argument("--log-file", default="")
    ap.add_argument("--bench-cmd", default="")
    ap.add_argument("--spec-note", default="MTP num_speculative_tokens=3")
    args = ap.parse_args()

    with open(args.json) as f:
        doc = json.load(f)

    concs = [int(x.strip()) for x in args.concurrency.split(",")]
    matrix = []
    for part in args.matrix.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        matrix.append((int(a), int(b)))

    ran = {(c["input_len"], c["output_len"], c["concurrency"]): c
           for c in doc["cells"] if c.get("successful_requests", 0) > 0}

    # 真实行（跑了）
    real_rows = build_rows(doc, args.log_file, args.server_cmd, args.bench_cmd)
    real_rows = [r for r in real_rows
                 if (r.get("input_len"), r.get("output_len"),
                     r.get("max_concurrency")) in ran]

    # 占位行（没跑）：只填组合标识，数据列留空
    placeholder_rows = []
    for inl, outl in matrix:
        for conc in concs:
            if (inl, outl, conc) in ran:
                continue
            row = {h: None for h in HEADERS}
            row["input_len"] = inl
            row["output_len"] = outl
            row["max_concurrency"] = conc
            row["num_prompts"] = conc
            row["traffic_request_rate"] = "inf"
            row["concurrency"] = conc
            placeholder_rows.append(row)

    rows = sorted(real_rows + placeholder_rows,
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
                f"tag={doc.get('tag')}, timestamp={doc.get('timestamp')}, "
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
          f"{len(real_rows)} ran + {len(placeholder_rows)} blank)")


if __name__ == "__main__":
    main()
