#!/usr/bin/env python3
"""Map scalar kernels into gpu_user_annotation intervals in a torch trace.
Usage: python3 analyze_annots.py <trace.json.gz>
"""
import gzip
import json
import sys
from collections import Counter

tr = json.load(gzip.open(sys.argv[1], "rt"))
ev = tr["traceEvents"]

annots = [e for e in ev if e.get("cat") in ("gpu_user_annotation", "user_annotation")]
print(f"annotations: {len(annots)}")
# show distinct annotation names with counts
names = Counter(a["name"] for a in annots)
for n, c in names.most_common(30):
    print(f"  {c:5d}  {n}")

# gpu_user_annotation intervals: tid is the gpu stream
gpu_int = [a for a in annots if a.get("cat") == "gpu_user_annotation"]
print(f"\ngpu intervals: {len(gpu_int)}")
# sample them
for a in gpu_int[:20]:
    print(f"  ts={a['ts']:<14} dur={a.get('dur',0):<10} tid={a['tid']}  {a['name'][:80]}")

# kernels of interest
knames = ["w8a8_scalar_gemm_kernel", "w8a8_gemm_scalar_fallback_kernel"]
ks = [e for e in ev if e.get("cat") == "kernel" and any(s in e["name"] for s in knames)]
print(f"\nscalar kernels: {len(ks)}")

# Map each kernel ts to the gpu_user_annotation (same stream) that encloses it
def enclosing_gpu(k):
    best = None
    for a in gpu_int:
        try:
            ts, dur = a["ts"], a.get("dur", 0)
        except KeyError:
            continue
        if a["tid"] == k["tid"] and ts <= k["ts"] < ts + dur:
            if best is None or ts > best[1]:
                best = (a["name"], ts)
    return best[0] if best else None

cnt = Counter()
for k in ks:
    cnt[enclosing_gpu(k)] += 1
print("\nscalar kernels enclosed by gpu_user_annotation:")
for n, c in cnt.most_common(20):
    print(f"  {c:6d}  {n}")

# Also: check execute_context names in all gpu annotations
print("\nexecute_context-like annotations:")
for n in names:
    if "execute" in n.lower() or "generation" in n.lower() or "graph" in n.lower():
        print(f"  {n[:120]}")
