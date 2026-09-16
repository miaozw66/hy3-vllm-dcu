#!/usr/bin/env python3
"""Multi-concurrency MTP benchmark (non-streaming). Measures ttft & steady tpot
across concurrency levels. Usage: python3 mtp_conc_bench.py [prompt_tok] [out_tok]
"""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://localhost:8000"
PROMPT_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
NUM_OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 128

sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
repeat = max(1, PROMPT_TOK // 18)
prompt = sentence * repeat


def post_once(conc, idx):
    """Single completion at given concurrency. Returns (ttft_ms, tpot_ms, e2e, out)."""
    p = {"model": "/models/Hy3-Channel-INT8-w8a8", "prompt": prompt,
         "max_tokens": NUM_OUT, "temperature": 0.0, "stream": False}
    r = urllib.request.Request(BASE + "/v1/completions",
                               data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.loads(resp.read().decode())
    dt = time.perf_counter() - t0
    out = d.get("usage", {}).get("completion_tokens", NUM_OUT)
    # ttft probe separately below
    return dt, out


def bench(conc):
    # warm ttft probe first (1 output token)
    p = {"model": "/models/Hy3-Channel-INT8-w8a8", "prompt": prompt,
         "max_tokens": 1, "temperature": 0.0, "stream": False}
    r = urllib.request.Request(BASE + "/v1/completions",
                               data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=600) as resp:
        json.loads(resp.read().decode())
    ttft = (time.perf_counter() - t0) * 1000

    results = [None] * conc
    def worker(i):
        results[i] = post_once(conc, i)
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0

    dts = [r[0] for r in results if r]
    outs = [r[1] for r in results if r]
    n = len(dts)
    if n == 0:
        return None
    total_out = sum(outs)
    # decode portion per request
    decode_avg = (sum(dts) / n - ttft / 1000)
    tpot = decode_avg / max(1, sum(outs) / n) * 1000
    rate = total_out / wall
    return {"conc": conc, "n": n, "ttft_ms": ttft,
            "tpot_ms": tpot, "e2e_avg_s": sum(dts) / n,
            "wall_s": wall, "tok_s": rate}


if __name__ == "__main__":
    print(f"prompt_tok={PROMPT_TOK} out_tok={NUM_OUT}", flush=True)
    for conc in [1, 2, 4, 8]:
        try:
            r = bench(conc)
            if r is None:
                print(f"conc={conc}: FAILED", flush=True)
            else:
                print(f"conc={conc}: ttft={r['ttft_ms']:.0f}ms "
                      f"tpot={r['tpot_ms']:.1f}ms/tok "
                      f"tok_s={r['tok_s']:.1f} wall={r['wall_s']:.1f}s "
                      f"n={r['n']}", flush=True)
        except Exception as e:
            print(f"conc={conc}: EXC {e!r}", flush=True)
    print("DONE", flush=True)
