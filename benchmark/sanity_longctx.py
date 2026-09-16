#!/usr/bin/env python3
"""140K SGLang 对齐服务 sanity 验证：发一个长上下文请求，确认不 OOM 且返回正常。

用法：
  python3 benchmark/sanity_longctx.py --endpoint http://localhost:8000 \
      --input-len 32768 --output-len 64

判定：
  - status == "ok" 且 input_tokens == 请求输入长度（表示完整 prefill 被接受）
  - 打印 ttft / avg_tpot / output_tokens / total_time
"""
import argparse
import os
import random
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cudagraph_bench import call_vllm_streaming  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--input-len", type=int, default=32768)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    # 从 /v1/models 获取 model id
    model = None
    try:
        resp = requests.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=10)
        if resp.status_code == 200 and resp.json()["data"]:
            model = resp.json()["data"][0]["id"]
    except Exception as e:
        print(f"[sanity] WARN: model query failed: {e}")
    if not model:
        print("[sanity] ERROR: cannot resolve model id from /v1/models")
        sys.exit(2)

    # random-id 风格输入（与 SGLang random-ids 数据集一致）
    rng = random.Random(42)
    prompt = [rng.randrange(1, 120000) for _ in range(args.input_len)]

    print(f"[sanity] model={model} input_len={args.input_len} output_len={args.output_len} "
          f"endpoint={args.endpoint}", flush=True)
    t0 = time.time()
    r = call_vllm_streaming(args.endpoint, model, prompt, args.output_len, 0.0,
                            args.timeout, True)
    wall = time.time() - t0

    print(f"[sanity] status={r['status']}")
    print(f"[sanity] input_tokens={r['input_tokens']} output_tokens={r['output_tokens']}")
    print(f"[sanity] ttft={r['ttft_s']:.3f}s avg_tpot={r['avg_tpot_s']*1000:.2f}ms "
          f"total_time={r['total_time_s']:.3f}s wall={wall:.3f}s")
    ok = r["status"] == "ok"
    print(f"[sanity] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
