#!/bin/bash
# PP=2 带 dump 的 vLLM 启动脚本 (单机 Ray)
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_pp_dump.sh
set -e

source "$(dirname "$0")/env.sh"

export NCCL_SOCKET_IFNAME=$NIC
export NCCL_DEBUG=WARN
export VLLM_HY3_DUMP_DIR=/tmp/pp_dump

rm -rf /tmp/pp_dump

echo "=== Starting vLLM PP=2 with dump ==="
echo "Dump dir: $VLLM_HY3_DUMP_DIR"
echo "Log file: /tmp/vllm_pp_dump.log"
echo "Started at: $(date)"

exec python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --max-model-len 8192 \
  --enforce-eager \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85
