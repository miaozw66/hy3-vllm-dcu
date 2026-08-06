#!/usr/bin/env python3
"""
HY3 Long-Context Needle-in-a-Haystack (NIAH) Correctness & Performance Test.

Tests the vLLM-deployed HY3 model at multiple context lengths:
  4K, 8K, 16K, 32K, 64K, 128K, 256K

For each length, inserts a "needle" (passkey) at 5 positions (0%, 25%, 50%, 75%, 100%)
within a "haystack" of filler text, then verifies the model correctly retrieves it.

Metrics collected:
  - Correctness (pass/fail — did the model retrieve the passkey?)
  - TTFT (Time To First Token, seconds)
  - TPOT (Time Per Output Token, ms/token)
  - Total tokens generated
  - Prefill tokens processed

Usage:
  # Quick test (4K, 8K only):
  python3 niah_test.py --endpoint http://localhost:8000 --lengths 4096,8192

  # Full test:
  python3 niah_test.py --endpoint http://localhost:8000 --lengths 4096,8192,16384,32768,65536,131072,262144

  # Custom positions:
  python3 niah_test.py --endpoint http://localhost:8000 --positions 0,50,100

NOTE for long contexts (128K+):
  - Set --gpu-memory-utilization 0.85 or higher in the vLLM launch command
  - Set --max-model-len to at least the max test length + 256 (decode margin)
  - KV cache per token: ~4KB per token per layer (GQA 8 KV heads × 128 dim × 2 bytes × 2 K/V)
    At 256K tokens × 80 layers = ~84 GB total KV cache (distributed across 8 GPUs)
"""

import argparse
import json
import random
import re
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
PASSKEY_PATTERN = "ALPHA-BRAVO-{code:04d}"  # {code} will be replaced with random 4-digit
QUERY_TEMPLATE = "What is the secret passkey? Answer with only the passkey, nothing else."

# Filler text: publicly available domain-neutral text.
# We use a paragraph from "The Elements of Style" — neutral, short, repeatable.
FILLER_PARAGRAPH = (
    "Vigorous writing is concise. A sentence should contain no unnecessary words, "
    "a paragraph no unnecessary sentences, for the same reason that a drawing should "
    "have no unnecessary lines and a machine no unnecessary parts. This requires not "
    "that the writer make all his sentences short, or that he avoid all detail and "
    "treat his subjects only in outline, but that every word tell. The secret to "
    "effective communication lies not in complexity but in clarity and precision. "
    "Many writers struggle with verbosity, filling pages with redundant expressions "
    "that dilute the core message. The art of revision is largely the art of deletion. "
    "Cutting words that add nothing to meaning or rhythm strengthens every sentence. "
    "This principle applies equally to technical documentation, academic writing, "
    "and creative prose. The reader's attention is a finite resource that must be "
    "respected. Each unnecessary phrase consumes cognitive bandwidth that could be "
    "devoted to understanding the author's actual argument rather than parsing "
    "decorative language. Good writing is transparent writing, where the medium "
    "becomes invisible and only the message remains visible to the reader. "
)


def estimate_tokens(text: str) -> int:
    """Rough token count estimation: ~4 chars/token for Chinese, ~4 chars/token for English."""
    # Conservative: assume 3.5 chars per token (typical for English+Chinese mixed)
    return int(len(text) / 3.5)


def generate_haystack(target_tokens: int, needle: str, position_pct: float) -> tuple[str, int, int]:
    """Generate a haystack of approximately target_tokens with the needle inserted.

    Args:
        target_tokens: desired total token count
        needle: the needle text to insert
        position_pct: where to insert the needle (0.0 = start, 1.0 = end)

    Returns:
        (full_text, needle_char_offset, actual_estimated_tokens)
    """
    needle_placeholder = "{{NEEDLE}}"
    needle_tokens = estimate_tokens(needle)
    filler_needed = target_tokens - needle_tokens - estimate_tokens(QUERY_TEMPLATE) - 10  # margin

    # Generate filler text by repeating the paragraph
    para_tokens = estimate_tokens(FILLER_PARAGRAPH)
    repeats = max(1, filler_needed // para_tokens + 2)
    filler = (FILLER_PARAGRAPH + "\n\n") * repeats

    # Calculate insertion point
    # We reserve space for the query at the end, so position is within the filler+needle portion
    body = filler + "\n\n" + needle_placeholder + "\n\n"
    body_tokens = estimate_tokens(body)

    # Build the full context by placing needle at specified position within the body
    half_filler_len = len(filler) // 2
    if position_pct <= 0:
        context = needle + "\n\n" + filler
        needle_pos = 0
    elif position_pct >= 1.0:
        context = filler + "\n\n" + needle
        needle_pos = len(filler) + 2
    else:
        split = int(len(filler) * position_pct)
        context = filler[:split] + "\n\n" + needle + "\n\n" + filler[split:]
        needle_pos = split + 2

    # Trim to approximate target length
    max_chars = target_tokens * 4  # rough upper bound
    if len(context) > max_chars:
        # Trim from the middle of filler sections
        context = context[:max_chars]

    return context, needle_pos


def make_prompt(context: str, question: str, cache_breaker: str = "") -> str:
    """Build the full prompt for HY3 (chat or completion format).

    Args:
        context: the haystack context with embedded needle
        question: the retrieval question
        cache_breaker: unique prefix to disable vLLM automatic prefix caching.
                       When set, this string is prepended to the prompt, ensuring
                       each request has a distinct prefix so KV cache from prior
                       requests is never reused.
    """
    if cache_breaker:
        return f"{cache_breaker}\n{context}\n\n{question}"
    return f"{context}\n\n{question}"


def call_vllm(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    timeout: int = 600,
) -> dict:
    """Call vLLM OpenAI-compatible API, measure TTFT and TPOT.

    Returns:
        dict with keys: text, ttft_s, tpot_ms, total_time_s, num_tokens,
                        prompt_tokens, completion_tokens, status
    """
    url = f"{endpoint.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    ttft = None
    token_times = []
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return {"status": "error", "http_status": resp.status_code, "text": resp.text[:200]}

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Capture TTFT on first token
                if ttft is None:
                    choices = data.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        ttft = time.perf_counter() - t_start

                # Collect token text and timing
                now = time.perf_counter()
                choices = data.get("choices", [])
                if choices:
                    text_delta = choices[0].get("text", "")
                    if text_delta:
                        token_times.append((now, text_delta))
                        full_text += text_delta

                # Usage info (usually in last chunk with stream_options.include_usage)
                usage = data.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)

    except requests.Timeout:
        return {"status": "timeout", "text": full_text}
    except requests.ConnectionError as e:
        return {"status": "connection_error", "text": str(e)}
    except Exception as e:
        return {"status": "error", "text": str(e)}

    total_time = time.perf_counter() - t_start

    # Compute TPOT (inter-token latency)
    tpot_ms = 0.0
    num_tokens = len(token_times)
    if num_tokens >= 2:
        intervals = []
        for i in range(1, num_tokens):
            intervals.append(token_times[i][0] - token_times[i - 1][0])
        tpot_ms = sum(intervals) / len(intervals) * 1000
    elif num_tokens == 1:
        tpot_ms = 0.0  # single token, no inter-token gap

    return {
        "status": "ok",
        "text": full_text.strip(),
        "ttft_s": ttft if ttft else total_time,
        "tpot_ms": tpot_ms,
        "total_time_s": total_time,
        "num_tokens": num_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens if completion_tokens else num_tokens,
    }


def check_passkey(response_text: str, expected_passkey: str) -> bool:
    """Check if the expected passkey appears in the response."""
    # Normalize: remove whitespace, lowercase
    resp_clean = response_text.strip().replace(" ", "").upper()
    expected_clean = expected_passkey.strip().replace(" ", "").upper()
    return expected_clean in resp_clean


def run_test(
    endpoint: str,
    model: str,
    length: int,
    positions: list[float],
    needle_code: int,
    max_new_tokens: int = 30,
    disable_prefix_cache: bool = False,
) -> list[dict]:
    """Run NIAH tests for one context length across all positions.

    Returns list of result dicts, one per position.
    """
    import uuid

    passkey = PASSKEY_PATTERN.format(code=needle_code)
    needle_text = f"The secret passkey is: {passkey}. Remember this passkey."
    results = []

    for pos in positions:
        pos_label = f"{int(pos * 100)}%"
        context, needle_offset = generate_haystack(length, needle_text, pos)
        cache_breaker = f"doc{uuid.uuid4().hex[:8]}\n" if disable_prefix_cache else ""
        prompt = make_prompt(context, QUERY_TEMPLATE, cache_breaker=cache_breaker)
        est_tokens = estimate_tokens(prompt)

        print(f"  [{pos_label}] context_len={length}, est_tokens={est_tokens}, "
              f"needle_offset={needle_offset} ... ", end="", flush=True)

        result = call_vllm(endpoint, model, prompt, max_tokens=max_new_tokens)

        if result["status"] == "ok":
            correct = check_passkey(result["text"], passkey)
            result["correct"] = correct
            result["position"] = pos_label
            result["context_length"] = length
            result["expected_passkey"] = passkey
            result["needle_offset"] = needle_offset
            status = "PASS" if correct else "FAIL"
            print(f"{status} | ttft={result['ttft_s']:.2f}s, "
                  f"tpot={result['tpot_ms']:.1f}ms, "
                  f"tokens={result['num_tokens']}, "
                  f"response={result['text'][:60]!r}")
        else:
            result["correct"] = False
            result["position"] = pos_label
            result["context_length"] = length
            result["expected_passkey"] = passkey
            print(f"ERROR: {result['status']} — {result.get('text', '')[:120]}")

        results.append(result)
        sys.stdout.flush()

    return results


def print_summary(all_results: list[dict]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("  NIAH TEST SUMMARY")
    print("=" * 100)

    # Group by context length
    by_length = {}
    for r in all_results:
        length = r["context_length"]
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(r)

    lengths = sorted(by_length.keys())

    # Header
    print(f"\n{'Length':>8s} | {'Position':>8s} | {'Correct':>7s} | "
          f"{'TTFT(s)':>8s} | {'TPOT(ms)':>8s} | {'Tokens':>6s} | {'Total(s)':>8s} | Response")
    print("-" * 100)

    all_pass = True
    for length in lengths:
        for r in by_length[length]:
            if r.get("status") != "ok":
                print(f"{length:>8,} | {r['position']:>8s} | {'ERROR':>7s} | "
                      f"{'N/A':>8s} | {'N/A':>8s} | {'N/A':>6s} | {'N/A':>8s} | {r.get('text','')[:40]}")
                all_pass = False
            else:
                mark = "PASS" if r["correct"] else "FAIL"
                print(f"{length:>8,} | {r['position']:>8s} | {mark:>7s} | "
                      f"{r['ttft_s']:>8.2f} | {r['tpot_ms']:>8.1f} | "
                      f"{r['num_tokens']:>6d} | {r['total_time_s']:>8.2f} | {r['text'][:40]!r}")
                if not r["correct"]:
                    all_pass = False

    # Per-length averages
    print(f"\n{'─' * 100}")
    print(f"  PER-LENGTH AVERAGES")
    print(f"{'─' * 100}")
    print(f"{'Length':>8s} | {'Pass Rate':>9s} | {'Avg TTFT':>8s} | {'Avg TPOT':>9s} | {'Avg Tokens':>10s}")
    print("-" * 65)

    for length in lengths:
        ok_results = [r for r in by_length[length] if r.get("status") == "ok"]
        if ok_results:
            n = len(ok_results)
            passed = sum(1 for r in ok_results if r["correct"])
            avg_ttft = sum(r["ttft_s"] for r in ok_results) / n
            avg_tpot = sum(r["tpot_ms"] for r in ok_results) / n
            avg_tokens = sum(r["num_tokens"] for r in ok_results) / n
            print(f"{length:>8,} | {passed}/{n} ({passed/n*100:.0f}%) | "
                  f"{avg_ttft:>8.2f}s | {avg_tpot:>8.1f}ms | {avg_tokens:>10.1f}")
        else:
            print(f"{length:>8,} | 0/{len(by_length[length])} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s}")

    print()
    if all_pass:
        print("  RESULT: ALL TESTS PASSED")
    else:
        failed_count = sum(1 for r in all_results if r.get("correct") is False or r.get("status") != "ok")
        print(f"  RESULT: {failed_count}/{len(all_results)} TESTS FAILED")


def main():
    parser = argparse.ArgumentParser(
        description="HY3 Long-Context Needle-in-a-Haystack Test"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:8000",
        help="vLLM OpenAI-compatible API endpoint"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (auto-detected from /v1/models if not specified)"
    )
    parser.add_argument(
        "--lengths", default="4096,8192,16384,32768,65536,131072,262144",
        help="Comma-separated context lengths to test"
    )
    parser.add_argument(
        "--positions", default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated needle positions (0.0 = start, 1.0 = end)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=30,
        help="Max tokens to generate per request"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for passkey generation"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate token counts and memory without making API calls"
    )
    parser.add_argument(
        "--disable-prefix-cache", action="store_true",
        help="Prepend a unique prefix to each prompt to prevent vLLM prefix caching. "
             "Use this to get fair, uncached TTFT measurements across different "
             "context lengths."
    )
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    positions = [float(x.strip()) for x in args.positions.split(",")]

    # Auto-detect model name if not specified
    model = args.model
    if model is None:
        try:
            resp = requests.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json()
                model = models["data"][0]["id"]
                max_len = models["data"][0].get("max_model_len", "?")
                print(f"Auto-detected model: {model} (max_model_len={max_len})")
        except Exception:
            model = "hy3"  # fallback

    print("=" * 80)
    print("  HY3 Long-Context Needle-in-a-Haystack Test")
    print("=" * 80)
    print(f"  Endpoint:     {args.endpoint}")
    print(f"  Model:        {model}")
    print(f"  Lengths:      {lengths}")
    print(f"  Positions:    {[f'{int(p*100)}%' for p in positions]}")
    print(f"  Max tokens:   {args.max_new_tokens}")
    print(f"  Seed:         {args.seed}")
    print()

    # Memory estimation
    print("─" * 80)
    print("  KV CACHE MEMORY ESTIMATES (GQA: 8 KV heads, 128 dim, BF16)")
    print("─" * 80)
    kv_per_token_per_layer = 2 * 8 * 128 * 2  # K + V, 8 heads, 128 dim, 2 bytes BF16
    kv_per_token_80l = kv_per_token_per_layer * 80
    print(f"  KV cache per token:         {kv_per_token_per_layer:,} B/layer, {kv_per_token_80l:,} B/80L")

    for length in lengths:
        kv_total = length * kv_per_token_80l
        kv_total_gb = kv_total / (1024 ** 3)
        # With PP=2, each stage has 40 layers. With TP=4, 2 KV heads per TP rank.
        kv_per_gpu_pp2_tp4 = length * 40 * (2 * 2 * 128 * 2)  # 40 layers, 2 heads/rank
        kv_per_gpu_gb = kv_per_gpu_pp2_tp4 / (1024 ** 3)
        print(f"  {length:>8,} tokens: ~{kv_total_gb:.1f} GB total, "
              f"~{kv_per_gpu_gb:.2f} GB per GPU (PP=2, TP=4)")

    print()
    print("  NOTE: If per-GPU KV cache exceeds available memory, reduce gpu-memory-utilization")
    print("        or increase it in the vLLM launch command (e.g., --gpu-memory-utilization 0.85).")
    print()

    if args.dry_run:
        for length in lengths:
            needle_code = random.randint(0, 9999)
            passkey = PASSKEY_PATTERN.format(code=needle_code)
            needle_text = f"The secret passkey is: {passkey}. Remember this passkey."
            context, offset = generate_haystack(length, needle_text, 0.5)
            prompt = make_prompt(context, QUERY_TEMPLATE)
            est = estimate_tokens(prompt)
            print(f"  Length {length:,}: est ~{est:,} tokens, needle at offset {offset}")
        return

    # ── Run tests ──────────────────────────────────────────────────────────
    random.seed(args.seed)
    all_results = []

    for i, length in enumerate(lengths):
        needle_code = random.randint(1000, 9999)
        passkey = PASSKEY_PATTERN.format(code=needle_code)
        print(f"\n{'─' * 80}")
        print(f"  [{i+1}/{len(lengths)}] Context Length: {length:,} tokens  |  "
              f"Passkey: {passkey}")
        print(f"{'─' * 80}")

        results = run_test(
            endpoint=args.endpoint,
            model=model,
            length=length,
            positions=positions,
            needle_code=needle_code,
            max_new_tokens=args.max_new_tokens,
            disable_prefix_cache=args.disable_prefix_cache,
        )
        all_results.extend(results)

    # ── Summary ────────────────────────────────────────────────────────────
    print_summary(all_results)

    # Save results
    output_file = f"niah_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
