#!/usr/bin/env python3
"""MTP vs baseline divergence probe: same 10 prompts, greedy, logprobs=N.

Saves per-step top-N logprob distributions so we can compare head logits
at divergence positions between MTP and baseline runs.

Usage:
    python3 benchmark/mtp_correctness_logprobs.py --output /tmp/mtp_lp.json
"""
import argparse
import json
import time
import urllib.request

PROMPTS = [
    "中国的首都是",
    "1+1等于几？请简短回答。",
    "Write a Python function to reverse a string:",
    "床前明月光，下一句是",
    "Explain what a tensor is in one paragraph:",
    "从1数到20：",
    "The capital of France is",
    "用一句话解释光合作用：",
    "def fibonacci(n):",
    "地球绕太阳一圈需要多长时间？",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--logprobs", type=int, default=10)
    args = parser.parse_args()

    results = []
    for i, prompt in enumerate(PROMPTS):
        payload = json.dumps(
            {
                "model": "/models/Hy3-Channel-INT8-w8a8",
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "return_token_ids": True,
                "logprobs": args.logprobs,
            }
        ).encode()
        req = urllib.request.Request(
            f"{args.endpoint}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        dt = time.perf_counter() - t0
        choice = data["choices"][0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError("Server did not return token_ids.")
        # vLLM logprobs: choice["logprobs"] = {tokens, token_logprobs, top_logprobs}
        lp = choice.get("logprobs") or {}
        top = lp.get("top_logprobs") or []
        results.append(
            {
                "prompt": prompt,
                "token_ids": token_ids,
                "top_logprobs": top,  # list of {token_str: logprob} per step
                "latency_s": round(dt, 2),
            }
        )
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:20]!r} -> "
              f"{token_ids} ({dt:.1f}s)")

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
