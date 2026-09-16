#!/usr/bin/env python3
"""Extract call context of w8a8_scalar_gemm_kernel from a torch profiler trace.
Usage: python3 analyze_scalar_src.py <trace.json.gz> [kernel_name_substr]
"""
import gzip
import json
import sys

path = sys.argv[1]
substr = sys.argv[2] if len(sys.argv) > 2 else "w8a8_scalar_gemm_kernel"

with gzip.open(path, "rt") as f:
    tr = json.load(f)

events = tr["traceEvents"]
devs = tr.get("deviceProperties", [])

# collect kernels matching name
kernels = [e for e in events if e.get("cat") == "kernel" and substr in e.get("name", "")]
print(f"matching kernel events: {len(kernels)}")
if not kernels:
    print("(no match; try substring like 'scalar' or 'gemm')")
    # list a few kernel names for orientation
    names = {}
    for e in events:
        if e.get("cat") == "kernel":
            n = e["name"]
            names[n] = names.get(n, 0) + 1
    for n, c in sorted(names.items(), key=lambda x: -x[1])[:25]:
        print(f"  {c:6d}  {n}")
    sys.exit(0)

# each kernel event has tid + ts; find enclosing CPU/GPU ops. torch profiler
# kernel events carry "args.stack" sometimes. Also the parent is the nearest
# preceding duration event on the same stream that contains this ts.
# Simpler robust proxy: nearest preceding CPU-side op (cat==op) that encloses,
# via event name grouping on the same tid before the kernel.
def enclosing(cat, e):
    best = None
    for o in events:
        if o.get("cat") != cat:
            continue
        try:
            ts, dur = o["ts"], o.get("dur", 0)
        except KeyError:
            continue
        if o["tid"] == e["tid"] and ts <= e["ts"] < ts + dur:
            if best is None or ts > best[1]:
                best = (o["name"], ts)
    return best[0] if best else None

from collections import Counter
cops = Counter()
cstack = Counter()
for k in kernels:
    p = enclosing("op", k)
    cops[p] += 1
    # name of enclosing device-side function (args.function?) if present
print("\nTop enclosing CPU op names (cat=op) for the kernel:")
for n, c in cops.most_common(20):
    print(f"  {c:6d}  {n}")

# check stack field on first few
for k in kernels[:3]:
    st = k.get("args", {}).get("stack", [])
    if st:
        print("\nsample stack:", json.dumps(st[:15], ensure_ascii=False)[:800])

# print one kernel's raw event for structure
print("\nsample raw kernel event keys:", sorted(kernels[0].keys()))
