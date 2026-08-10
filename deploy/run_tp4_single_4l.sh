#!/bin/bash
# Single-node TP=4 4-layer submodel launch (for quick verification)
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_tp4_single_4l.sh
set -e

source "$(dirname "$0")/env.sh"

export NCCL_SOCKET_IFNAME=$NIC
export NCCL_DEBUG=WARN

LOG=/tmp/vllm_single_tp4_4l_v2.log
rm -f "$LOG"
rm -rf "$DUMP_DIR/single_tp4_4l" 2>/dev/null || true
mkdir -p "$DUMP_DIR/single_tp4_4l"

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
