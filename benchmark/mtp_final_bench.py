#!/usr/bin/env python3
"""Final MTP benchmark: non-streaming, measures ttft and steady-state tpot
from a single completion. Usage: python3 mtp_final_bench.py [output] [prompt_tok]
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
NUM_OUT = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PROMPT_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
repeat = max(1, PROMPT_TOK // 18)
prompt = sentence * repeat


def post(prompt, mt, stream=False):
    p = {"model": "/models/Hy3-Channel-INT8-w8a8", "prompt": prompt,
         "max_tokens": mt, "temperature": 0.0, "stream": stream}
    r = urllib.request.Request(BASE + "/v1/completions",
                               data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.loads(resp.read().decode())
    return time.perf_counter() - t0, d.get("usage", {})


# ttft probe (1 output token ≈ prefill time)
dt1, _ = post(prompt, 1)
# full run
dt, usage = post(prompt, NUM_OUT)
out = usage.get("completion_tokens", NUM_OUT)
decode = (dt - dt1)  # rough decode portion
tpot = decode / out * 1000 if out else float("nan")
print(f"ttft(1tok)={dt1*1000:.0f}ms  full={dt:.2f}s out={out} "
      f"tpot≈{tpot:.1f}ms/tok  e2e_tok_rate={out/dt:.1f}tok/s", flush=True)
