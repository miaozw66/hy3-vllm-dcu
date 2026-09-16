#!/usr/bin/env python3
"""E2E test: 7k-token input -> 2048-token completion against the zth-HIP vLLM server.

Builds a prompt that tokenizes to exactly ~7000 tokens (via the model tokenizer),
POSTs /v1/completions (stream, max_tokens=2048, temperature=0), prints usage and
wall-clock, and saves the generated text.
Usage: python3 e2e_7k_2048.py [--url http://127.0.0.1:8000/v1/completions]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", default="/models/Hy3-Channel-INT8-w8a8")
    p.add_argument("--model-path", default="/models/Hy3-Channel-INT8-w8a8")
    p.add_argument("--input-tokens", type=int, default=7000)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--ignore-eos", action="store_true", default=True,
                   help="generate exactly max_tokens (force long decode)")
    p.add_argument("--out", default="/tmp/e2e_7k_2048_gen.txt")
    return p.parse_args()


def build_prompt(path: str, target: int) -> tuple[str, int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # A sentence that tokenizes predictably; repeat until we hit target tokens.
    unit = ("The HY3 processor on this GPU node performs matrix multiplication "
            "for language model inference with INT8 quantization and pipeline "
            "parallelism across eight accelerators. ")
    n_unit = 1
    while True:
        text = unit * n_unit
        n = len(tok.encode(text))
        if n >= target:
            # trim unit-by-unit to get as close as possible
            while n_unit > 1 and len(tok.encode(unit * (n_unit - 1))) >= target:
                n_unit -= 1
                text = unit * n_unit
                n = len(tok.encode(text))
            return text, n
        n_unit += 1


def main() -> int:
    args = parse_args()
    prompt, n_tok = build_prompt(args.model_path, args.input_tokens)
    print(f"prompt chars={len(prompt)} approx_tokens={n_tok} target={args.input_tokens}",
          flush=True)

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_tok_t = None
    n_chunks = 0
    last_finish = None
    usage = None
    pieces = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if "error" in ev:
                    print("SERVER ERROR:", json.dumps(ev["error"], ensure_ascii=False),
                          file=sys.stderr)
                    return 1
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    if ch.get("finish_reason"):
                        last_finish = ch["finish_reason"]
                    txt = ch.get("text", "")
                    if txt:
                        if first_tok_t is None:
                            first_tok_t = time.perf_counter()
                        pieces.append(txt)
                        n_chunks += 1
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1

    t1 = time.perf_counter()
    gen = "".join(pieces)
    with open(args.out, "w") as f:
        f.write(gen)
    dt = t1 - t0
    ttf = first_tok_t - t0 if first_tok_t is not None else float("nan")
    print(f"\n=== RESULTS ===")
    print(f"total wall: {dt:.2f}s   first-token ttf: {ttf:.2f}s")
    print(f"stream chunks(with text): {n_chunks}   finish_reason: {last_finish}")
    if usage:
        print(f"usage: prompt_tokens={usage.get('prompt_tokens')} "
              f"completion_tokens={usage.get('completion_tokens')} "
              f"total={usage.get('total_tokens')}")
    gen_tok = (usage or {}).get("completion_tokens", n_chunks)
    if gen_tok and dt - ttf > 0:
        print(f"decode speed: {gen_tok / (dt - ttf):.1f} tok/s "
              f"(({gen_tok} tokens) / ({dt - ttf:.2f}s decode))")
    print(f"generated chars: {len(gen)}   saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
