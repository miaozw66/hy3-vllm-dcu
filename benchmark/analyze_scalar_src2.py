#!/usr/bin/env python3
"""Map w8a8 scalar kernels to their launching CPU op via correlation id.
Usage: python3 analyze_scalar_src2.py <trace.json.gz> [kernel_substr]
"""
import gzip
import json
import sys
from collections import Counter

path = sys.argv[1]
substr = sys.argv[2] if len(sys.argv) > 2 else "w8a8_scalar_gemm_kernel"

tr = json.load(gzip.open(path, "rt"))
ev = tr["traceEvents"]

# cpu_op events: map correlation -> (name, stack)
op_by_corr = {}
for e in ev:
    if e.get("cat") == "cpu_op":
        a = e.get("args", {})
        c = a.get("correlation")
        if c is not None:
            st = a.get("stack") or []
            # shorten stack to module-level frames
            op_by_corr.setdefault(c, (e["name"], st))

# kernel events: collect correlation for matching kernels
kc = Counter()
kstack = Counter()
unknown = 0
for e in ev:
    if e.get("cat") == "kernel" and substr in e.get("name", ""):
        c = e.get("args", {}).get("correlation")
        if c is None:
            unknown += 1
            continue
        name, st = op_by_corr.get(c, ("<no-cpu-op>", []))
        kc[name] += 1
        # stack tail: which frames appear
        for fr in st:
            pass
        # record full stack once
        kstack[(name, tuple(st[-6:]))] += 1

print(f"kernels matched: {sum(kc.values())}  (no-corr: {unknown})")
print("\nLaunching CPU op distribution:")
for n, cnt in kc.most_common(20):
    print(f"  {cnt:6d}  {n}")

print("\nSample stacks for top launchers:")
seen = set()
for (name, sttail), cnt in kstack.most_common(8):
    if name in seen:
        continue
    seen.add(name)
    print(f"\n-- {name} (x{cnt}) stack tail:")
    for fr in sttail:
        print(f"     {fr}")

# also count all kernel names to see if scalar kernels are isolated to a phase
if unknown:
    print(f"\nNOTE: {unknown} matching kernels had no correlation (graph replay?)")
