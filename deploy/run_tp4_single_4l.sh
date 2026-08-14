#!/bin/bash
# Single-node TP=4 4-layer submodel launch (for quick verification)
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_tp4_single_4l.sh
set -e

source "$(dirname "$0")/env.sh"

# 单机 TP=4 走 PCIe，无需网卡配置
export NCCL_DEBUG=WARN

LOG=/tmp/vllm_single_tp4_4l_v2.log
rm -f "$LOG"
echo "Starting single TP=4 4-layer server at $(date)"
echo "Log: $LOG"

exec script -f -c "python3 -u -m vllm.entrypoints.openai.api_server \
  --model $SUBMODEL_PATH \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4 \
  --port 8001" "$LOG"
