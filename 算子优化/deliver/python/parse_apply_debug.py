#!/usr/bin/env python3
"""Summarize /tmp/zth_apply_debug.txt produced by ZTH_W8A8_DEBUG=1.

Per (pid, kn) shows the process() pack result and every APPLY branch decision
(m value -> branch). Usage: python3 parse_apply_debug.py [path]
"""
import sys
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zth_apply_debug.txt"
    proc = {}
    apply_rows = defaultdict(list)
    order = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            pid = parts[0]
            kind = parts[1]
            if kind == "PROCESS" and len(parts) >= 9:
                proc[pid] = line
                order.append(("PROCESS", pid, line))
            elif kind == "APPLY":
                # kn=..., m=..., ->BRANCH ...
                apply_rows[pid].append(line)
    print("== PROCESS (per worker pid) ==")
    seen = set()
    for _kind, pid, line in sorted(order):
        if pid in seen:
            continue
        seen.add(pid)
        print(f"  pid {pid}:")
        print(f"    {line.split('PROCESS', 1)[1][1:]}")
    print("\n== APPLY branch summary (per pid, per kn) ==")
    for pid, rows in apply_rows.items():
        # group by kn
        by_kn = defaultdict(lambda: defaultdict(int))
        for r in rows:
            kn = ""
            m = ""
            br = ""
            for t in r.split("\t")[2:]:
                if t.startswith("kn="):
                    kn = t
                elif t.startswith("m="):
                    m = t
                elif t.startswith("->"):
                    br = t
            by_kn[kn][f"{m} {br}"] += 1
        for kn, mbs in sorted(by_kn.items()):
            print(f"  pid {pid} {kn}:")
            for mb, c in sorted(mbs.items()):
                print(f"      {c:4d}  {mb}")
    print("\n== raw tail (last 40 lines) ==")
    with open(path) as f:
        lines = f.readlines()
    for l in lines[-40:]:
        print("  " + l.rstrip("\n")[:180])


if __name__ == "__main__":
    main()
