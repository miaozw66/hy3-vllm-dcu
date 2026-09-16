#!/usr/bin/env python3
"""Generate the formal HY3 MTP vs baseline -O1 8k/16k evaluation report.

Reads results_online_mtp/{mtp,base}_{8k,16k}.json and writes a markdown
report with TTFT / TPOT / usage throughput / MTP acceptance / tail-answer acc.
"""
import json
import os
import re
import statistics
from datetime import datetime

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_online_mtp")


def load(name):
    p = os.path.join(OUTDIR, name)
    with open(p) as f:
        return json.load(f)


def extract_answer(text):
    boxes = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not boxes:
        return None
    return boxes[-1].strip().replace(",", "")


def level_table(data, tag):
    """Return (rows, acc) where rows are per-concurrency dicts."""
    spec = data.get("spec_decode", {})
    by_conc = {}
    for r in data["requests"]:
        if r.get("status") == "ok":
            by_conc.setdefault(r["concurrency"], []).append(r)
    rows = []
    for c in sorted(by_conc):
        reqs = by_conc[c]
        ttfts = [r["ttft_s"] for r in reqs]
        tpots = [r["avg_tpot_s"] for r in reqs]
        wall = max(r["batch_wall_time_s"] for r in reqs)
        total_out = sum(r["output_tokens"] for r in reqs)
        sp = spec.get(str(c)) or {}
        al = sp.get("avg_len")
        raw_tpot_ms = statistics.mean(tpots) * 1000  # SSE chunk interval (= step time)
        # per-token TPOT: MTP chunks carry ~avg_len tokens; baseline chunks carry 1 token
        pt_tpot_ms = raw_tpot_ms / al if al else raw_tpot_ms
        rows.append({
            "conc": c, "n": len(reqs),
            "ttft": statistics.mean(ttfts),
            "tpot_ms": raw_tpot_ms,
            "tpot_pt": pt_tpot_ms,
            "tps": total_out / wall if wall else 0,
            "wall": wall,
            "acc_rate": sp.get("acc_rate"),
            "avg_len": al,
        })
    # tail answer accuracy
    ok = [r for r in data["requests"] if r.get("status") == "ok"]
    corr = n = 0
    for r in ok:
        ans = extract_answer(r.get("text", ""))
        if ans is None:
            continue
        n += 1
        if ans == str(r["answer_ref"]).strip().replace(",", ""):
            corr += 1
    return rows, (corr, n)


def fmt(x, fmt_str=".3f"):
    return f"{x:{fmt_str}}" if x is not None else "  -"


def main():
    out = []
    out.append("# HY3 MTP vs Baseline -O1 长上下文评测报告")
    out.append("")
    out.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append("")
    out.append("**评测配置**：TP=8、max-model-len 32768、gpu-memory-utilization 0.90、"
               "--no-enable-prefix-caching、-O1 --no-async-scheduling、temperature=0、--ignore-eos。")
    out.append("")
    out.append("**输入**：长上下文 + 尾部嵌真实 GSM8K 题目（8k 组 = 输入 7165 token、输出 2048；"
               "16k 组 = 输入 16381 token、输出 1024），每档并发 1/2/4/8/16。")
    out.append("")
    out.append("**口径**：TTFT/TPOT 来自客户端流式计时。TPOT 列给出两种口径："
               "①「块间隔」= 相邻 SSE 块到达间隔（服务器每步时间，MTP 每块含 ~avg_len 个 token，"
               "baseline 每块 1 个 token）；②「TPOT/词」= 折算到单 token 的延迟"
               "（MTP = 块间隔/avg_len，baseline = 块间隔）。吞吐 = sum(usage output_tokens)/batch_wall_time；"
               "MTP 接受率 = /metrics `spec_decode_num_accepted_tokens_total` / `num_drafts_total` 相邻快照差。")
    out.append("")
    out.append("**注意（尾部答案准确率仅供参考）**：温度=0 下 TP=8 推理输出并非 bit-exact —— 不同并发档位 batch 形状不同，"
               "allreduce 归约顺序差异导致 logits 微小漂移，接近平局的贪心 argmax 可能翻转；实测同一 prompt 在不同并发档位、"
               "以及 MTP/baseline 两台服务器间输出文本不同。因此 MTP 与 baseline 的答案准确率差异不能归因于 MTP 本身。"
               "该指标仅统计在 max_tokens 内解出 `\\boxed{}` 的请求，且模型可能用 boxed 标注中间值，故仅供观察模型长上下文解题能力。")
    out.append("")

    tables = {}
    for tag, label in [("mtp", "MTP"), ("base", "Baseline")]:
        for lvl in ["8k", "16k"]:
            name = f"{tag}_{lvl}"
            p = os.path.join(OUTDIR, f"{name}.json")
            if not os.path.exists(p):
                out.append(f"## {label} {lvl}\n\n(数据缺失: {p})\n")
                continue
            data = load(f"{name}.json")
            rows, acc = level_table(data, name)
            tables[(label, lvl)] = (rows, acc)
            out.append(f"## {label} {lvl}\n")
            out.append("| conc | n | TTFT(s) | 块间隔(ms) | TPOT/词(ms) | 吞吐(tok/s) | wall(s) | 接受率 | avg_len |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                out.append(f"| {r['conc']} | {r['n']} | {r['ttft']:.2f} | {r['tpot_ms']:.1f} | "
                           f"{r['tpot_pt']:.1f} | {r['tps']:.1f} | {r['wall']:.0f} | "
                           f"{fmt(r['acc_rate'])} | {fmt(r['avg_len'])} |")
            out.append(f"\n尾部答案准确率：{acc[0]}/{acc[1]} ({acc[1] and f'{acc[0]/acc[1]:.1%}' or 'N/A'})\n")

    # speedup table
    out.append("## MTP vs Baseline 加速比（MTP/Baseline，>1 表示 MTP 更快）\n")
    for lvl in ["8k", "16k"]:
        if ("MTP", lvl) not in tables or ("Baseline", lvl) not in tables:
            continue
        mtp_rows = {r["conc"]: r for r in tables[("MTP", lvl)][0]}
        base_rows = {r["conc"]: r for r in tables[("Baseline", lvl)][0]}
        out.append(f"### {lvl}\n")
        out.append("| conc | TTFT加速 | TPOT/词加速 | 吞吐加速 |")
        out.append("|---|---|---|---|")
        for c in sorted(set(mtp_rows) & set(base_rows)):
            m, b = mtp_rows[c], base_rows[c]
            t_s = b["ttft"] / m["ttft"] if m["ttft"] else 0
            p_s = b["tpot_pt"] / m["tpot_pt"] if m["tpot_pt"] else 0
            u_s = m["tps"] / b["tps"] if b["tps"] else 0
            out.append(f"| {c} | {t_s:.2f}x | {p_s:.2f}x | {u_s:.2f}x |")
        out.append("")

    report_path = os.path.join(OUTDIR, "mtp_vs_baseline_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
