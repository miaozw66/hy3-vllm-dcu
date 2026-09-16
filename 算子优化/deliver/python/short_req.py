#!/usr/bin/env python3
"""Short request to trigger CUDA-graph capture (sizes 1..16) + a few decode
steps, so the torch-profiler trace (record_shapes=true) contains the zth_* op
shapes. Usage: python3 short_req.py [--input-tokens N] [--max-tokens N]
"""
import argparse
import json
import sys
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", default="/models/Hy3-Channel-INT8-w8a8")
    p.add_argument("--input-tokens", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/models/Hy3-Channel-INT8-w8a8", trust_remote_code=True)
    unit = ("HY3 inference on the DCU node exercises the quantized INT8 "
            "scaled matrix multiplication path. ")
    n = 1
    while len(tok.encode(unit * n)) < args.input_tokens:
        n += 1
    prompt = unit * n
    print(f"prompt tokens={len(tok.encode(prompt))} chars={len(prompt)}",
          flush=True)

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
        method="POST",
    )
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "error" in ev:
                print("SERVER ERROR:", json.dumps(ev["error"],
                                                  ensure_ascii=False),
                      file=sys.stderr)
                return 1
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("text"):
                    chunks += 1
    print(f"chunks={chunks} usage={usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
