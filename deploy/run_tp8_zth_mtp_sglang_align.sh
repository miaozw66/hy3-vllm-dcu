#!/bin/bash
# TP=8 -O1 MTP 服务器启动脚本，对齐 SGLang MTP 表（'Hy3-Channel-INT8-w8a8+mtp213 L20' sheet）参数。
#
# SGLang 启动命令（xlsx A1 单元格）：
#   sglang serve --model-path /home/models/Hy3-Channel-INT8-w8a8/ \
#     --reasoning-parser auto --tool-call-parser auto \
#     --tp 8 --host 0.0.0.0 --port 30000 \
#     --quantization w8a8_int8 \
#     --mem-fraction-static 0.86 \
#     --cuda-graph-max-bs 16 \
#     --context-length 140000 \
#     --speculative-algorithm EAGLE --speculative-num-steps 2 \
#     --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
#
# SGLang 运行时（A2 单元格）：
#   max_total_num_tokens=140023, chunked_prefill_size=4096,
#   max_prefill_tokens=16384, max_running_requests=48,
#   context_len=140000, available_gpu_mem=1.39 GB
#
# 参数映射（公共参数对齐；SGLang 独有忽略）：
#   --tp 8                     -> --tensor-parallel-size 8
#   --quantization w8a8_int8   -> --quantization compressed-tensors (W8A8)
#   --mem-fraction-static 0.86 -> --gpu-memory-utilization 0.86
#   --cuda-graph-max-bs 16     -> --cudagraph-capture-sizes "1 2 4 8 16"
#   --context-length 140000    -> --max-model-len 140000
#   EAGLE draft 3              -> MTP num_speculative_tokens=3
#   --tool-call-parser auto    -> --tool-call-parser hy_v3
#   --reasoning-parser auto    -> (SGLang 独有，忽略)
#   --speculative-num-steps 2  -> (SGLang EAGLE 独有，忽略)
#   --speculative-eagle-topk 1 -> (SGLang EAGLE 独有，忽略)
#   chunked_prefill_size=4096  -> --max-num-batched-tokens 4096
#   max_running_requests=48    -> --max-num-seqs 48
#   max_prefill_tokens=16384   -> (SGLang 调度独有，无 vLLM 直接对应，忽略)
#
# 用法：
#   ZTH_W8A8_MODE=use  bash deploy/run_tp8_zth_mtp_sglang_align.sh   # zth 内核（默认）
#   ZTH_W8A8_MODE=off bash deploy/run_tp8_zth_mtp_sglang_align.sh   # 纯 vLLM Triton
#   DRAFT=1            bash deploy/run_tp8_zth_mtp_sglang_align.sh   # 覆盖 draft token 数
#   MTP=off            bash deploy/run_tp8_zth_mtp_sglang_align.sh   # 关闭 MTP（baseline）
set -e

source "$(dirname "$0")/env.sh"

# ── 可选 zth W8A8 内核（交付方 import-hook，仅当 deliver 存在时生效） ──
ZTH_W8A8_MODE="${ZTH_W8A8_MODE:-use}"
DELIVER="${ZTH_W8A8_DELIVER:-$PROJECT_ROOT/算子优化/deliver}"
case "$ZTH_W8A8_MODE" in
    off|validate|use) ;;
    *) echo "ZTH_W8A8_MODE must be off, validate, or use" >&2; exit 2 ;;
esac

# ── 覆盖项 ────────────────────────────────────────────────────
DRAFT="${DRAFT:-3}"     # SGLang EAGLE draft-tokens=3
MTP="${MTP:-on}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-140000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1 2 4 8 16}"   # SGLang cuda-graph-max-bs=16
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"             # SGLang max_running_requests=48
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"  # SGLang chunked_prefill_size=4096
EAGER="${EAGER:-off}"                           # on = --enforce-eager（规避 -O1 编译约束 bug）

# ── 环境（必须与既有 MTP/baseline 评测一致，避免 W8A8 量化边界数值漂移） ──
export NCCL_DEBUG=WARN
export VLLM_ROCM_USE_AITER=0
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
# cand_03（HY3_MTP_TRACE_VERIFICATION_PLAN_20260906）：MTP 复测 LL/Tree。
# env.sh(42 行 source) 已是 NCCL_PROTO=LL + NCCL_ALGO=Tree；此处 Simple/Ring 覆盖
# 令 0903 trace 及 0829/0831 测量落在 Simple/Ring 延迟地板。去掉覆盖以让 LL/Tree 生效，
# 用无 profiler TPOT A/B 复测（no-MTP 单流已验 -18%，MTP 复测为 TODO）。
# Allow controlled protocol A/B without changing the production default in env.sh.
if [ -n "${NCCL_PROTO_OVERRIDE:-}" ]; then
    export NCCL_PROTO="$NCCL_PROTO_OVERRIDE"
fi
if [ -n "${NCCL_ALGO_OVERRIDE:-}" ]; then
    export NCCL_ALGO="$NCCL_ALGO_OVERRIDE"
fi

# zth hook 注入
if [ "$ZTH_W8A8_MODE" = "off" ]; then
    IFS=':' read -r -a pythonpath_entries <<< "${PYTHONPATH:-}"
    filtered_pythonpath=()
    for entry in "${pythonpath_entries[@]}"; do
        [ "$entry" = "$DELIVER/python" ] || filtered_pythonpath+=("$entry")
    done
    PYTHONPATH="$(IFS=:; printf '%s' "${filtered_pythonpath[*]}")"
    export PYTHONPATH
    unset ZTH_W8A8_CACHE
    export ZTH_W8A8_MODE="off"
else
    export PYTHONPATH="$DELIVER/python${PYTHONPATH:+:$PYTHONPATH}"
    export ZTH_W8A8_CACHE="$DELIVER/so_cache"
    export ZTH_W8A8_MODE
fi

# ── GPU memory preflight ────────────────────────────────────
python3 "$PROJECT_ROOT/deploy/check_gpu_memory.py" \
    --expected-gpus "$GPU_COUNT" --min-free-mib 60000

# ── Clean up residual processes ─────────────────────────────
pkill -9 -f vllm.entrypoints 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f Worker_TP 2>/dev/null || true
sleep 2

LOG_TAG="sglang_align_d${DRAFT}_${ZTH_W8A8_MODE}"
if [ "$MTP" = "off" ]; then
    LOG_TAG="sglang_align_base_${ZTH_W8A8_MODE}"
    SPEC_ARGS=""
else
    SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT}'"
fi
if [ "$EAGER" = "on" ]; then
    EAGER_ARG="--enforce-eager"
    LOG_TAG="${LOG_TAG}_eager"
else
    EAGER_ARG=""
fi

NODE0_LOG="$LOG_DIR/vllm_tp8_${LOG_TAG}_$(date +%m%d_%H%M).log"
echo "=== TP=8 SGLang-aligned server (MTP=$MTP draft=$DRAFT zth=$ZTH_W8A8_MODE eager=$EAGER) ==="
echo "  max-model-len=$MAX_MODEL_LEN  gpu-mem=$GPU_MEM_UTIL  capture=[$CAPTURE_SIZES]"
echo "  max-num-seqs=$MAX_NUM_SEQS  max-num-batched-tokens=$MAX_BATCHED_TOKENS"
echo "  Log: $NODE0_LOG"

ARGS="--model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --cudagraph-capture-sizes $CAPTURE_SIZES \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens $MAX_BATCHED_TOKENS \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --distributed-timeout-seconds $DISTRIBUTED_TIMEOUT_SECONDS \
  --no-async-scheduling \
  --port 8000 \
  $SPEC_ARGS \
  $EAGER_ARG \
  -O1"

exec script -f -c "python3 -u -m vllm.entrypoints.openai.api_server $ARGS" "$NODE0_LOG"
