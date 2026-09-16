#!/usr/bin/env python3
"""Rebuild kernel timeline inside the first execute_context(4) annotation.
Usage: python3 rebuild_step.py <trace.json.gz> [max_kernels]
"""
import gzip
import json
import sys

tr = json.load(gzip.open(sys.argv[1], "rt"))
ev = tr["traceEvents"]
MAXK = int(sys.argv[2]) if len(sys.argv) > 2 else 60

# find first execute_context(4) gpu annotation interval
iv = None
for e in ev:
    if e.get("cat") == "gpu_user_annotation" and "generation_1(4)" in e["name"]:
        iv = e
        break
if iv is None:
    print("no execute_context(4) gpu annotation")
    sys.exit(0)
t0, t1, tid = iv["ts"], iv["ts"] + iv.get("dur", 0), iv["tid"]
print(f"interval ts={t0:.0f}..{t1:.0f} dur={t1-t0:.0f}us tid={tid}  {iv['name']}")

# kernels inside interval on same stream
ks = [e for e in ev if e.get("cat") == "kernel" and e.get("tid") == tid
      and t0 <= e["ts"] < t1]
ks.sort(key=lambda e: e["ts"])
print(f"kernels inside: {len(ks)}; showing first {min(MAXK, len(ks))} + shape rollup")
for k in ks[:MAXK]:
    name = k["name"]
    # shorten template noise
    for s in ("(signed char const*, signed char const*, float const*, float const*, hip_bfloat16*, int, int, int)",
              "(signed char const*, signed char const*, float const*, float const*, float*, int, int, int)"):
        name = name.replace(s, "()")
    print(f"  +{k['ts']-t0:8.1f}us  dur={k.get('dur',0):9.1f}us  {name[:110]}")

# rollup of kernel names inside
from collections import Counter
c = Counter()
for k in ks:
    n = k["name"]
    if "scalar" in n:
        n = "SCALAR:" + n.split("(")[0].split("::")[-1]
    else:
        n = n.split("(")[0]
        for tok in ("::", " "):
            if tok in n:
                n = n.split(tok)[-1]
    c[n] += 1
print("\nrollup of kernel names inside interval (top 25):")
for n, cnt in c.most_common(25):
    print(f"  {cnt:6d}  {n[:80]}")
