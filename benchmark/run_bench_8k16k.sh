#!/bin/bash
# 8k/16k benchmark per SGLang screenshot parameters (MTP vs baseline).
#
# SGLang ref (带MTP的SGLang hy3 nvidia-l20.jpg, OCR 2026-08-20):
#   8k  group: input 7168 + output 2048, concurrency 1/2/4/8/16
#   16k group: input 16384 + output 1024, concurrency 1/2/4/8/16
#
# vllm bench throughput is OFFLINE engine mode (no openai backend) - the API
# server must be stopped before running. Outputs JSON per (mode,len,conc).
# NOTE: for --dataset-name random, use --random-input-len/--random-output-len
# (the plain --input-len/--output-len are IGNORED in random mode, defaults 1024/128).
#
# Usage:
#   MODE=base bash benchmark/run_bench_8k16k.sh   # baseline (no MTP)
#   MODE=mtp  bash benchmark/run_bench_8k16k.sh   # MTP speculative
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/deploy/env.sh"

MODE=${MODE:-base}
OUTDIR=${OUTDIR:-$PROJECT_ROOT/benchmark/results_8k16k_$(date +%m%d_%H%M)}
mkdir -p "$OUTDIR"

# ── Environment: MUST stay identical across MTP/baseline runs ──
export NCCL_DEBUG=WARN
export VLLM_ROCM_USE_AITER=0
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_DISABLE_LOG_LOGO=1

SEED=42
CONCS="1 2 4 8 16"

# (tag, input_len, output_len)
LEN_GROUPS=("8k 7168 2048" "16k 16384 1024")

if [ "$MODE" = "mtp" ]; then
    SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
else
    SPEC_ARGS=""
fi

echo "=== 8k/16k bench: MODE=$MODE seed=$SEED ==="
echo "Output dir: $OUTDIR"

for group in "${LEN_GROUPS[@]}"; do
    read -r TAG ILEN OLEN <<< "$group"
    for CONC in $CONCS; do
        OUT_JSON="$OUTDIR/${MODE}_${TAG}_i${ILEN}_o${OLEN}_c${CONC}.json"
        echo ""
        echo "--- $TAG: input=$ILEN output=$OLEN conc=$CONC ---"
        CMD="vllm bench throughput \
          --model $MODEL_PATH \
          --tensor-parallel-size 8 \
          --trust-remote-code \
          --max-model-len 32768 \
          --gpu-memory-utilization 0.90 \
          --no-enable-prefix-caching \
          --dataset-name random \
          --random-input-len $ILEN \
          --random-output-len $OLEN \
          --num-prompts $CONC \
          --seed $SEED \
          --output-json $OUT_JSON \
          -O1 --no-async-scheduling \
          $SPEC_ARGS"
        echo "  \$ $CMD"
        eval "$CMD"
        echo "  saved: $OUT_JSON"
    done
done

echo ""
echo "=== All done. Results in $OUTDIR ==="
