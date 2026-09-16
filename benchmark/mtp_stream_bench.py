#!/usr/bin/env python3
"""Streaming benchmark for the MTP server: measures ttft and per-token decode
time (tpot) precisely using the streaming API.

Usage: python3 mtp_stream_bench.py <num_requests> [output_tokens] [prompt_tokens]
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
NUM_REQS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NUM_OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 96
PROMPT_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
tokens_per_sentence = 18
repeat = max(1, PROMPT_TOK // tokens_per_sentence)
prompt = sentence * repeat


def stream_completion(prompt, max_tokens, model="/models/Hy3-Channel-INT8-w8a8"):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    token_times = []  # (ts, dt_since_prev)
    t0 = time.perf_counter()
    first_ts = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        for chunk in resp:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                now = time.perf_counter()
                if first_ts is None:
                    first_ts = now
                    token_times.append((now, 0.0))
                else:
                    token_times.append((now, now - token_times[-1][0]))
        e2e = time.perf_counter() - t0
    if first_ts is None:
        return None, None, None
    ttft = first_ts - t0
    n = len(token_times)
    if n <= 1:
        return ttft, e2e, None
    dt = [token_times[i][1] for i in range(1, n)]
    return ttft, e2e, dt


all_ttft = []
all_tpot = []
all_e2e = []
for i in range(NUM_REQS):
    ttft, e2e, dt = stream_completion(prompt, NUM_OUT)
    if ttft is None:
        print(f"req {i}: FAILED (no tokens)", flush=True)
        continue
    # drop first inter-token (includes post-ttft scheduling) for steady state
    steady = dt[2:] if dt and len(dt) > 3 else (dt or [])
    tpot = sum(steady) / len(steady) * 1000 if steady else float("nan")
    all_ttft.append(ttft * 1000)
    all_tpot.append(tpot)
    all_e2e.append(e2e)
    print(f"req {i}: ttft={ttft*1000:.0f}ms e2e={e2e:.2f}s out={len(dt)+1} "
          f"steady_tpot={tpot:.1f}ms/tok", flush=True)

if all_ttft:
    print(f"\nSUMMARY n={len(all_ttft)}:")
    print(f"  ttft: avg={sum(all_ttft)/len(all_ttft):.0f}ms")
    print(f"  steady_tpot: avg={sum(all_tpot)/len(all_tpot):.1f}ms/tok")
    print(f"  e2e: avg={sum(all_e2e)/len(all_e2e):.2f}s")
