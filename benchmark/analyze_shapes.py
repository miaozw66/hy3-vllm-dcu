#!/usr/bin/env python3
"""Extract CPU-op input shapes for gemm/quant ops from a record_shapes trace.
Usage: python3 analyze_shapes.py <trace.json.gz>
"""
import gzip
import json
import sys
from collections import Counter

tr = json.load(gzip.open(sys.argv[1], "rt"))
ev = tr["traceEvents"]

# cpu_op events are ph=='X' with cat 'cpu_op' (or op). args.input_shapes present when record_shapes=True.
hits = Counter()       # op name -> Counter of first-dim shapes
examples = []
for e in ev:
    if e.get("ph") != "X":
        continue
    name = e.get("name", "")
    a = e.get("args", {})
    shapes = a.get("input_shapes") or a.get("Input Dims")
    if not shapes:
        continue
    lname = name.lower()
    if any(k in lname for k in ("gemm", "zth", "quant", "scaled_mm", "linear", "mm", "matmul", "moe")):
        mvals = Counter()
        for s in shapes:
            if isinstance(s, (list, tuple)) and len(s) >= 1:
                mvals[s[0]] += 1
        hits[(name, tuple(sorted(mvals.items())))] += 1
        if len(examples) < 5:
            examples.append((name, shapes[:4]))

print("CPU ops with shapes (name -> M-dim distribution -> count):")
for (name, md), cnt in hits.most_common(40):
    print(f"  {cnt:6d}  {name[:70]}  M={md}")

print("\nsample events:")
for name, shapes in examples:
    print(f"  {name[:60]} shapes={shapes}")
