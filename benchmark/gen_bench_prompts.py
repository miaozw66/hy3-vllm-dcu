#!/usr/bin/env python3
"""Generate fixed-length Chinese prompts for online 8k/16k benchmarking.

Targets match the SGLang screenshot parameters (OCR 2026-08-20):
  8k  group: input 7168 tokens
  16k group: input 16384 tokens

Uses the model's own tokenizer (AutoTokenizer) to measure token counts and
iteratively trims the text until it hits the target within a small tolerance.
Fixed seed -> deterministic text across MTP/baseline runs.
"""

import argparse
import random

from transformers import AutoTokenizer


CHARS = (
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成"
    "会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着"
    "等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把"
    "性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新"
    "线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公"
    "无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活"
    "设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运"
    "农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规"
    "热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美"
    "再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完"
    "类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉"
    "林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委"
    "素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号"
    "率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂"
    "京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
)


def build_text(target_tokens: int, rng: random.Random) -> str:
    """Build random Chinese text approximating target_tokens tokens."""
    # ~1.15 chars/token for common Chinese chars; generate 20% headroom
    n_chars = int(target_tokens * 1.15 * 1.2)
    pieces = []
    for _ in range(n_chars // 64 + 1):
        pieces.append("".join(rng.choice(CHARS) for _ in range(64)))
    return "\n".join(pieces)


def trim_to_tokens(tokenizer, text: str, target: int) -> str:
    """Iteratively trim text so tokenizer.encode() lands near target."""
    cur = text
    for _ in range(8):
        n = len(tokenizer.encode(cur))
        if n <= target:
            return cur
        # overshoot: cut proportionally, re-encode and repeat
        keep_ratio = target / n
        cut = int(len(cur) * keep_ratio)
        cur = cur[:cut]
        if n - target < target * 0.02:
            break
    # final binary search on token count
    lo, hi = 0, len(cur)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(cur[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    return cur[:lo]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Hy3-Channel-INT8-w8a8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="/home/hy3-vllm-dcu/benchmark/prompts")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rng = random.Random(args.seed)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    for target, tag in [(7168, "8k"), (16384, "16k")]:
        text = build_text(target, rng)
        text = trim_to_tokens(tokenizer, text, target)
        n = len(tokenizer.encode(text))
        out = os.path.join(args.outdir, f"prompt_{tag}.txt")
        with open(out, "w") as f:
            f.write(text)
        print(f"{tag}: target={target} actual={n} chars={len(text)} -> {out}")


if __name__ == "__main__":
    main()
