#!/usr/bin/env python3
"""Print OpenAI completion SSE deltas as they arrive."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream each generated completion token immediately."
    )
    parser.add_argument("prompt", help="Prompt sent to the completion endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="hy3")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--hide-token-ids",
        action="store_true",
        help="Do not print streamed token IDs to stderr.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                event = json.loads(data)
                if "error" in event:
                    print(json.dumps(event["error"], ensure_ascii=False), file=sys.stderr)
                    return 1
                for choice in event.get("choices", []):
                    text = choice.get("text", "")
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    token_ids = choice.get("token_ids")
                    if token_ids and not args.hide_token_ids:
                        timestamp = time.strftime("%H:%M:%S")
                        print(
                            f"[{timestamp}] token_ids={token_ids}",
                            file=sys.stderr,
                            flush=True,
                        )
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"request failed: {error.reason}", file=sys.stderr)
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
