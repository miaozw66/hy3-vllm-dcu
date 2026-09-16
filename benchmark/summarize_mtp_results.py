#!/usr/bin/env python3
"""Aggregate MTP vs baseline -O1 8k/16k benchmark results into a comparison report.

Reads the four result JSONs produced by bench_tail_prompts.py:
  results_online_mtp/{mtp,base}_{8k,16k}.json

Report per concurrency level (1,2,4,8,16):
  * TTFT (mean over ok requests)
  * TPOT (mean over ok requests)
  * usage throughput = sum(output_tokens) / batch_wall_time_s
  * acceptance rate + avg acceptance length (MTP only, from /metrics deltas)
  * tail GSM8K answer accuracy (exact match of boxed answer)
"""
import argparse
import json
import os
import re
import statistics

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_online_mtp")


def load(path):
    with open(path) as f:
        return json.load(f)


def extract_answer(text: str) -> str | None:
    # take the last \boxed{...}
    m = [x for x in re.findall(r"\\boxed\{([^}]+)\}", text)]
    if not m:
        return None
    return m[-1].strip().replace(",", "")


def score_answers(data):
    """Exact-match accuracy of tail GSM8K answers."""
    ok = [r for r in data["requests"] if r.get("status") == "ok"]
    correct = 0
    n = 0
    for r in ok:
        ans = extract_answer(r.get("text", ""))
        if ans is None:
            continue
        n += 1
        if ans == str(r["answer_ref"]).strip().replace(",", ""):
            correct += 1
    return correct, n


def level_stats(data):
    by_conc = {}
    for r in data["requests"]:
        if r.get("status") != "ok":
            continue
        c = r["concurrency"]
        by_conc.setdefault(c, []).append(r)
    out = {}
    for c, reqs in sorted(by_conc.items()):
        ttfts = [r["ttft_s"] for r in reqs]
        tpots = [r["avg_tpot_s"] for r in reqs]
        wall = max(r["batch_wall_time_s"] for r in reqs)
        total_out = sum(r["output_tokens"] for r in reqs)
        out[c] = {
            "n": len(reqs),
            "ttft_s": statistics.mean(ttfts),
            "tpot_ms": statistics.mean(tpots) * 1000,
            "usage_tps": total_out / wall if wall > 0 else 0,
            "wall_s": wall,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--save", default=None, help="save report markdown path")
    args = ap.parse_args()

    print("=" * 78)
    print("HY3 MTP vs baseline -O1  |  8k/16k tail-GSM8K benchmark")
    print("=" * 78)

    rows = []  # (name, level, stats dict)
    for tag, name in [("mtp", "MTP"), ("base", "Baseline")]:
        for lvl, out_tok in [("8k", 2048), ("16k", 1024)]:
            p = os.path.join(args.outdir, f"{tag}_{lvl}.json")
            if not os.path.exists(p):
                print(f"[missing] {p}")
                continue
            data = load(p)
            stats = level_stats(data)
            spec = data.get("spec_decode", {})
            correct, n = score_answers(data)
            print(f"\n--- {name} {lvl} (max_tokens={out_tok}, in≈{lvl[0]}k) ---")
            print(f"    tail-answer acc: {correct}/{n} = {correct/n:.2%}")
            print(f"    {'conc':>4} {'n':>3} {'TTFT(s)':>8} {'TPOT(ms)':>8} "
                  f"{'usage_tps':>9} {'wall(s)':>7} {'acc_rate':>9} {'avg_len':>7}")
            for c, s in stats.items():
                sp = spec.get(str(c)) or spec.get(c) or {}
                ar = sp.get("acc_rate")
                al = sp.get("avg_len")
                print(f"    {c:>4} {s['n']:>3} {s['ttft_s']:>8.2f} {s['tpot_ms']:>8.1f} "
                      f"{s['usage_tps']:>9.1f} {s['wall_s']:>7.1f} "
                      f"{ar if ar is None else f'{ar:.3f}':>9} {al if al is None else f'{al:.3f}':>7}")
            rows.append((name, lvl, stats))

    # ---- speedup / comparison table ----
    print("\n" + "=" * 78)
    print("MTP vs baseline speedup (MTP/base)")
    print("=" * 78)
    mtp = {lvl: stats for name, lvl, stats in rows if name == "MTP"}
    base = {lvl: stats for name, lvl, stats in rows if name == "Baseline"}
    for lvl in ["8k", "16k"]:
        if lvl not in mtp or lvl not in base:
            continue
        print(f"\n{lvl}:")
        print(f"    {'conc':>4} {'TTFT spd':>9} {'TPOT spd':>9} {'usage_tps spd':>13}")
        for c in sorted(set(mtp[lvl]) & set(base[lvl])):
            m, b = mtp[lvl][c], base[lvl][c]
            t_s = b["ttft_s"] / m["ttft_s"] if m["ttft_s"] else 0
            p_s = b["tpot_ms"] / m["tpot_ms"] if m["tpot_ms"] else 0
            u_s = m["usage_tps"] / b["usage_tps"] if b["usage_tps"] else 0
            print(f"    {c:>4} {t_s:>9.2f} {p_s:>9.2f} {u_s:>13.2f}")

    if args.save:
        with open(args.save, "w") as f:
            f.write("HY3 MTP vs baseline -O1 8k/16k benchmark report\n")
            f.write(f"generated: results in {args.outdir}\n")
            for name, lvl, stats in rows:
                f.write(f"\n{name} {lvl}\n")
                for c, s in stats.items():
                    f.write(f"  conc={c} n={s['n']} ttft={s['ttft_s']:.2f}s "
                            f"tpot={s['tpot_ms']:.1f}ms usage_tps={s['usage_tps']:.1f}\n")
        print(f"\nReport saved to {args.save}")


if __name__ == "__main__":
    main()
