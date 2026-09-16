#!/usr/bin/env python3
"""Generate 8k/16k long-context prompts with a real GSM8K question embedded
at the TAIL (needle-in-haystack style, but the needle is a full few-shot math
question). Front padding is deterministic random Chinese text filled to an exact
token count (7168 / 16384), so the run matches the MTP_RESULTS online benchmark
caliber while letting us also score the tail question's answer.

Usage:
  python3 benchmark/gen_tail_prompts.py \
      --gsm8k /tmp/eval_mtp_gsm8k/predictions/Hy3-Channel-INT8-w8a8/gsm8k_main.jsonl \
      --num-questions 16 --outdir benchmark/tail_prompts --seed 42
"""
import argparse
import json
import os
import random
import re

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


def build_padding(target_tokens: int, rng: random.Random) -> str:
    """Build random Chinese text approximating target_tokens tokens."""
    n_chars = int(target_tokens * 1.15 * 1.2)
    pieces = []
    for _ in range(n_chars // 64 + 1):
        pieces.append("".join(rng.choice(CHARS) for _ in range(64)))
    return "\n".join(pieces)


def trim_to_tokens(tokenizer, text: str, target: int) -> str:
    """Trim text so tokenizer.encode() lands near target (binary search from head)."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(text[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def extract_answer(assistant_content: str) -> str | None:
    m = re.search(r"boxed\{([^}]+)\}", assistant_content)
    if m:
        return m.group(1).strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", required=True, help="Path to gsm8k_main.jsonl")
    parser.add_argument("--model", default="/models/Hy3-Channel-INT8-w8a8")
    parser.add_argument("--num-questions", type=int, default=16)
    parser.add_argument("--outdir", default="/home/hy3-vllm-dcu/benchmark/tail_prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load GSM8K rows (user content = few-shot + question, answer from assistant).
    rows = []
    with open(args.gsm8k) as f:
        for line in f:
            d = json.loads(line)
            user_content = next(m["content"] for m in d["messages"] if m["role"] == "user")
            asst_content = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
            ans = extract_answer(asst_content)
            rows.append({"index": d["index"], "prompt": user_content, "answer": ans})
    rows = rows[: args.num_questions]
    assert all(r["answer"] for r in rows), "Some rows missing boxed answer"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    os.makedirs(args.outdir, exist_ok=True)

    for target, tag in [(7168, "8k"), (16384, "16k")]:
        out_items = []
        for r in rows:
            rng = random.Random(args.seed + r["index"])
            tail_tokens = len(tokenizer.encode(r["prompt"]))
            # separator "\n\n题目：\n\n" is a handful of tokens; target padding below it.
            pad_target = target - tail_tokens - 6
            padding = build_padding(pad_target, rng)
            padding = trim_to_tokens(tokenizer, padding, pad_target)
            full = padding + "\n\n题目：\n\n" + r["prompt"]
            n = len(tokenizer.encode(full))
            out_items.append({
                "id": r["index"],
                "gsm8k_index": r["index"],
                "answer": r["answer"],
                "prompt": full,
                "token_count": n,
            })
            print(f"  {tag} q{r['index']}: tokens={n} (target={target}) tail_tokens={tail_tokens}")

        out = os.path.join(args.outdir, f"gsm8k_tail_{tag}.json")
        with open(out, "w") as f:
            json.dump(out_items, f, indent=2, ensure_ascii=False)
        counts = [it["token_count"] for it in out_items]
        print(f"{tag}: wrote {len(out_items)} prompts to {out} "
              f"(token range {min(counts)}-{max(counts)}, mean {sum(counts)//len(counts)})")


if __name__ == "__main__":
    main()
