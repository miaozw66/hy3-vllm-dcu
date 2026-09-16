#!/bin/bash
# Run the 8k and 16k tail-prompt benchmarks sequentially against a ready server.
# Usage:
#   bash benchmark/run_bench_levels.sh mtp
#   bash benchmark/run_bench_levels.sh base
set -e
cd "$(dirname "$0")/.."

TAG="$1"
[ -z "$TAG" ] && { echo "usage: $0 <mtp|base>"; exit 1; }
END=${ENDPOINT:-http://localhost:8000}
OUTDIR="benchmark/results_online_mtp"
mkdir -p "$OUTDIR"

# 8k: 16 prompts x 2048 output tokens
python3 benchmark/bench_tail_prompts.py \
    --endpoint "$END" \
    --prompt-list benchmark/tail_prompts/gsm8k_tail_8k.json \
    --max-tokens 2048 \
    --concurrencies 1,2,4,8,16 \
    --runs 1 --ignore-eos \
    --output "$OUTDIR/${TAG}_8k.json"

# 16k: 16 prompts x 1024 output tokens (MTP_RESULTS caliber)
python3 benchmark/bench_tail_prompts.py \
    --endpoint "$END" \
    --prompt-list benchmark/tail_prompts/gsm8k_tail_16k.json \
    --max-tokens 1024 \
    --concurrencies 1,2,4,8,16 \
    --runs 1 --ignore-eos \
    --output "$OUTDIR/${TAG}_16k.json"

echo "=== DONE ${TAG} ==="
ls -la "$OUTDIR/${TAG}"_*.json
