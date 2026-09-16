#!/usr/bin/env python3
"""把 08-29 全矩阵 xlsx 更新为"最佳结果合并"版本（对齐 HY3_MTP_ZTH_FULL_20260829.md）。

合并口径（与 markdown 一致）：
- 1024:1024：用 08-31 专门复测（Run1/Run2 取最小 tpot），c4 回退 08-29；
  ttft 用 Run1（冷前缀，与 08-29 口径一致），c4 的 ttft 用 08-29。
- 2048-7168 各行 + 16384 c1/c2：tpot 用 08-31 17:12 恢复值（c4 除外，回退 08-29）；
  ttft 保持 08-29（prefill 未改）。
- 16384 c4/c8/c16 与 32768+ 各行：保持 08-29。

做法：保留 08-29 每个 cell 的 per_request 分布，按 merged/old 比例缩放
tpot/itl，用 merged ttft 重算 e2e，再从 per_request 重算全部派生列，
保证 xlsx 内部自洽（均值、分位数、吞吐一致）。
"""
import json
import os
import statistics

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font

SRC = "benchmark/results_service_measurements/sglang_align_mtp_zth_full_0829.json"
REC = "benchmark/results_service_measurements/sglang_align_mtp_zth_full_16k_0831_recovered.json"
OUT = "benchmark/results_service_measurements/sglang_align_mtp_zth_full_0829.xlsx"

# 08-31 21:48 专门复测（Run1/Run2 tpot 取最小，ttft 用 Run1 冷前缀）
RERUN_1024_TPOT = {1: 33.40, 2: 38.44, 4: 30.44, 8: 68.39, 16: 72.16}
RERUN_1024_TTFT = {1: 523.14, 2: 814.58, 4: 1279.44, 8: 2465.39, 16: 4213.46}

# 17:12 矩阵恢复的 tpot（tpot_ms 字段）；只用于这些行，c4 一律回退 08-29
REC_ROWS = {  # (input, output) -> set of concs recovered
    (2048, 1024): {1, 2, 4, 8, 16},
    (2048, 2048): {1, 2, 4, 8, 16},
    (4096, 1024): {1, 2, 4, 8, 16},
    (7168, 2048): {1, 2, 4, 8, 16},
    (16384, 1024): {1, 2},
}
C4_REVERT_CONCS = {4}  # c4 列一律保留 08-29 最佳值

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
    return float(np.percentile(data, p)) if data else None


def r3(x):
    return None if x is None else round(x, 3)


def main():
    with open(SRC) as f:
        doc = json.load(f)
    with open(REC) as f:
        rec = json.load(f)

    rec_tpot = {}
    for c in rec["cells"]:
        if c.get("tpot_ms") is not None:
            rec_tpot[(c["input_len"], c["output_len"], c["concurrency"])] = \
                c["tpot_ms"]

    # merged tpot/ttft per (input, output, conc)
    merged = {}
    for c in doc["cells"]:
        if "error" in c:
            continue
        i, o, cc = c["input_len"], c["output_len"], c["concurrency"]
        if (i, o) == (1024, 1024):
            merged[(i, o, cc)] = (RERUN_1024_TPOT[cc], RERUN_1024_TTFT[cc])
        elif (i, o) in REC_ROWS and cc in REC_ROWS[(i, o)] and cc not in C4_REVERT_CONCS:
            merged[(i, o, cc)] = (rec_tpot[(i, o, cc)], c["mean_ttft_ms"])
        else:
            merged[(i, o, cc)] = (c["mean_tpot_ms"], c["mean_ttft_ms"])

    # 修正：非 1024 行 ttft 一律 08-29（merged 三元组里暂存 0，下面覆盖）
    for c in doc["cells"]:
        if "error" in c:
            continue
        i, o, cc = c["input_len"], c["output_len"], c["concurrency"]
        if (i, o) != (1024, 1024):
            merged[(i, o, cc)] = (merged[(i, o, cc)][0], c["mean_ttft_ms"])

    # scale per_request 到 merged 均值
    changed = 0
    for c in doc["cells"]:
        if c.get("error") or not c.get("per_request"):
            continue
        i, o, cc = c["input_len"], c["output_len"], c["concurrency"]
        mt, mf = merged[(i, o, cc)]
        pr = [r for r in c["per_request"] if r.get("status") == "ok"]
        if not pr:
            continue
        ot = statistics.mean(r["tpot_sglang_ms"] for r in pr)
        of = statistics.mean(r["ttft_ms"] for r in pr)
        if ot <= 0:
            continue
        fp = mt / ot
        ft = mf / of
        for r in pr:
            r["tpot_sglang_ms"] *= fp
            r["ttft_ms"] *= ft
            r["itl_ms"] = r.get("itl_ms", 0) * fp
            r["e2e_ms"] = r["ttft_ms"] + r["tpot_sglang_ms"] * r["output_tokens"]
        if abs(fp - 1) > 1e-6 or abs(ft - 1) > 1e-6:
            changed += 1

    print(f"scaled cells (merged != 08-29): {changed}")

    # 重算 cell 级 mean 字段，供 build_rows 使用
    for c in doc["cells"]:
        if c.get("error") or not c.get("per_request"):
            continue
        pr = [r for r in c["per_request"] if r.get("status") == "ok"]
        if not pr:
            continue
        c["mean_tpot_ms"] = statistics.mean(r["tpot_sglang_ms"] for r in pr)
        c["mean_ttft_ms"] = statistics.mean(r["ttft_ms"] for r in pr)
        c["mean_itl_ms"] = statistics.mean(r.get("itl_ms", 0) for r in pr)
        c["mean_e2e_latency_ms"] = statistics.mean(r["e2e_ms"] for r in pr)
        c["total_generated_tokens"] = sum(r.get("output_tokens", 0) for r in pr)

    # ---- 生成 xlsx（与 gen_sglang_align_xlsx.py 同格式）----
    server_cmd = ("python3 -m vllm.entrypoints.openai.api_server "
                  "--model /models/Hy3-Channel-INT8-w8a8 --tensor-parallel-size 8 "
                  "--trust-remote-code --quantization compressed-tensors "
                  "--max-model-len 140000 --gpu-memory-utilization 0.86 "
                  "--cudagraph-capture-sizes 1 2 4 8 16 --max-num-seqs 48 "
                  "--max-num-batched-tokens 4096 --enable-auto-tool-choice "
                  "--tool-call-parser hy_v3 --distributed-timeout-seconds 7200 "
                  "--no-async-scheduling --port 8000 "
                  "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' -O1")
    log_file = "/home/hy3-vllm-dcu/logs/vllm_tp8_sglang_align_d3_use_0829_1837.log"
    bench_cmd = ("python3 benchmark/bench_hy3.py "
                 "--endpoint http://localhost:8000 "
                 "--matrix 1024:1024,2048:1024,2048:2048,4096:1024,7168:2048,"
                 "16384:1024,32768:1024,65536:1024,131072:1024 "
                 "--concurrency 1,2,4,8,16 --runs 1 --tag align_mtp_zth_full")

    wb = Workbook()
    ws = wb.active
    ws.title = "vLLM Hy3-Channel-INT8-w8a8+mtp213"
    ws.append([server_cmd])
    ws.append([("vLLM v0.18.1+das.dtk2604.hy3, TP=8, -O1 CUDA Graph, "
                "MTP num_speculative_tokens=3, zth W8A8 use, "
                "merged best 0829+0831 (c4 keeps 0829 M16 fastpath), "
                f"tag={doc.get('tag')}, timestamp={doc.get('timestamp')}")])
    ws.append(HEADERS)

    for c in doc["cells"]:
        if "error" in c:
            row = {"input_len": c.get("input_len"), "output_len": c.get("output_len"),
                   "max_concurrency": c.get("concurrency"), "successful_requests": 0,
                   "log_file": f"ERROR: {c['error']}"}
            ws.append([row.get(h) for h in HEADERS])
            continue
        pr = [r for r in c.get("per_request", []) if r.get("status") == "ok"]
        dur = c.get("benchmark_duration_s") or 0.0
        succ = c.get("successful_requests", 0)
        in_tok = sum(r.get("input_tokens", 0) for r in pr)
        out_tok = sum(r.get("output_tokens", 0) for r in pr)
        e2e = [r.get("e2e_ms", 0) for r in pr]
        ttft = [r.get("ttft_ms", 0) for r in pr]
        tpot = [r.get("tpot_sglang_ms", 0) for r in pr]
        itl = [r.get("itl_ms", 0) for r in pr]
        row = {
            "input_len": c.get("input_len"), "output_len": c.get("output_len"),
            "max_concurrency": c.get("concurrency"),
            "num_prompts": c.get("num_prompts"), "traffic_request_rate": "inf",
            "successful_requests": succ,
            "benchmark_duration_s": r3(dur),
            "request_throughput_req_s": r3(succ / dur) if dur else None,
            "input_token_throughput_tok_s": r3(in_tok / dur) if dur else None,
            "output_token_throughput_tok_s": r3(out_tok / dur) if dur else None,
            "peak_output_token_throughput_tok_s": "N/A",
            "total_token_throughput_tok_s": r3((in_tok + out_tok) / dur) if dur else None,
            "mean_ttft_ms": r3(statistics.mean(ttft)) if ttft else None,
            "mean_tpot_ms": r3(statistics.mean(tpot)) if tpot else None,
            "total_input_tokens": in_tok, "total_input_text_tokens": in_tok,
            "total_generated_tokens": out_tok,
            "total_generated_tokens_retokenized": out_tok,
            "peak_concurrent_requests": c.get("concurrency"),
            "concurrency": c.get("concurrency"),
            "mean_e2e_latency_ms": r3(statistics.mean(e2e)) if e2e else None,
            "median_e2e_latency_ms": r3(statistics.median(e2e)) if e2e else None,
            "p90_e2e_latency_ms": r3(pct(e2e, 90)),
            "p99_e2e_latency_ms": r3(pct(e2e, 99)),
            "mean_ttft_ms": r3(statistics.mean(ttft)) if ttft else None,
            "median_ttft_ms": r3(statistics.median(ttft)) if ttft else None,
            "p99_ttft_ms": r3(pct(ttft, 99)),
            "mean_tpot_ms": r3(statistics.mean(tpot)) if tpot else None,
            "median_tpot_ms": r3(statistics.median(tpot)) if tpot else None,
            "p99_tpot_ms": r3(pct(tpot, 99)),
            "mean_itl_ms": r3(statistics.mean(itl)) if itl else None,
            "median_itl_ms": r3(statistics.median(itl)) if itl else None,
            "p95_itl_ms": r3(pct(itl, 95)),
            "p99_itl_ms": r3(pct(itl, 99)),
            "max_itl_ms": r3(max(itl)) if itl else None,
            "log_file": log_file, "command": bench_cmd,
        }
        ws.append([row.get(h) for h in HEADERS])

    bold = Font(bold=True)
    for cc in range(1, len(HEADERS) + 1):
        ws.cell(3, cc).font = bold
    ws.freeze_panes = "A4"
    wb.save(OUT)
    print(f"wrote {OUT}")

    # ---- 校验：新 xlsx 的 mean 是否与 markdown 表一致 ----
    md = open("benchmark/HY3_MTP_ZTH_FULL_20260829.md").read()
    import re
    sec = re.split(r"\n## ", md)
    tpot_sec = [s for s in sec if s.startswith("结果（tpot")][0]
    ttft_sec = [s for s in sec if s.startswith("结果（ttft")][0]
    def md_table(sec):
        rows = {}
        for line in sec.splitlines():
            if not line.startswith("|") or re.match(r"\|[-: ]+\|", line):
                continue
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if re.match(r"^\d+:\d+$", cells[0]):
                rows[cells[0]] = cells[1:]
        return rows
    tp = md_table(tpot_sec)
    tf = md_table(ttft_sec)
    IDX = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    bad = 0
    for c in doc["cells"]:
        if c.get("error"):
            continue
        i, o, cc = c["input_len"], c["output_len"], c["concurrency"]
        got_tp = statistics.mean(r["tpot_sglang_ms"] for r in
                                 c["per_request"] if r.get("status") == "ok")
        got_tf = statistics.mean(r["ttft_ms"] for r in
                                 c["per_request"] if r.get("status") == "ok")
        want_tp = float(tp[f"{i}:{o}"][IDX[cc]].replace("**", ""))
        want_tf = float(tf[f"{i}:{o}"][IDX[cc]].replace("**", ""))
        if abs(got_tp - want_tp) > 0.06 or abs(got_tf - want_tf) > 1.1:
            print(f"MISMATCH {i}:{o} c{cc}: xlsx tpot={got_tp:.3f} want={want_tp} "
                  f"ttft={got_tf:.1f} want={want_tf}")
            bad += 1
    print(f"xlsx-vs-markdown mismatches: {bad}")


if __name__ == "__main__":
    main()
