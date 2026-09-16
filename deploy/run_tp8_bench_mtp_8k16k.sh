#!/bin/bash
# TP=8 -O1 单机 8k/16k 评测服务器启动脚本（MTP vs baseline 同口径对比用）。
# 口径与 MTP_RESULTS 在线性能一致：max-model-len 32768、显存 0.90、
# 关 prefix caching、-O1 + --no-async-scheduling、TP=8。
#
#   bash deploy/run_tp8_bench_mtp_8k16k.sh            # MTP（默认，spec mtp 1 draft）
#   SPEC=none bash deploy/run_tp8_bench_mtp_8k16k.sh  # baseline（无 speculative）
#
# 评测输入用 gen_tail_prompts.py 生成的长上下文+尾部GSM8K题目 prompt。
set -e
source "$(dirname "$0")/env.sh"

SPEC=${SPEC:-mtp}
NCCL_DEBUG=WARN
export NCCL_DEBUG=WARN
export VLLM_ROCM_USE_AITER=0
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
# cand_03（HY3_MTP_TRACE_VERIFICATION_PLAN_20260906）：MTP 复测 LL/Tree。
# env.sh 已为 LL+Tree；去掉此处 Simple/Ring 覆盖并无 profiler TPOT A/B 复测。
#export NCCL_PROTO=Simple
#export NCCL_ALGO=Ring

# ── GPU memory preflight ────────────────────────────────────
python3 "$PROJECT_ROOT/deploy/check_gpu_memory.py" \
    --expected-gpus "$GPU_COUNT" --min-free-mib 60000

# ── Clean up residual processes ─────────────────────────────
pkill -9 -f vllm.entrypoints 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f Worker_TP 2>/dev/null || true
sleep 2

LOG_TAG="mtp"
if [ "$SPEC" = "none" ]; then
    LOG_TAG="base"
    SPEC_ARGS=""
else
    SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
fi

NODE0_LOG="$LOG_DIR/vllm_tp8_bench_${LOG_TAG}_o1_$(date +%m%d_%H%M).log"
echo "=== TP=8 -O1 bench server (SPEC=$SPEC) ==="
echo "  Log: $NODE0_LOG"

ARGS="--model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --distributed-timeout-seconds $DISTRIBUTED_TIMEOUT_SECONDS \
  --port 8000 \
  $SPEC_ARGS \
  -O1 --no-async-scheduling"

# Use script -f for unbuffered output
exec script -f -c "python3 -u -m vllm.entrypoints.openai.api_server $ARGS" "$NODE0_LOG"
