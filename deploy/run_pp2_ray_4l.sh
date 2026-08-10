#!/bin/bash
# Single-node PP=2 4-layer debug launch (Ray backend)
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_pp2_ray_4l.sh
set -e

source "$(dirname "$0")/env.sh"

export NCCL_SOCKET_IFNAME=$NIC
export NCCL_DEBUG=WARN
export VLLM_HY3_DUMP_DIR=$PROJECT_ROOT/dumps/pp2_4l
export VLLM_HY3_DUMP_SKIP=1

LOG=/tmp/vllm_pp2_4l_ray6.log
rm -f "$LOG"

echo "Starting PP=2 Ray 4-layer server at $(date)"
echo "Log: $LOG"
echo "PID: $$"
echo "VLLM_HY3_DUMP_DIR=$VLLM_HY3_DUMP_DIR"
echo "VLLM_HY3_DUMP_SKIP=$VLLM_HY3_DUMP_SKIP"
env | grep VLLM >> "$LOG"

exec script -f -c "python3 -u -m vllm.entrypoints.openai.api_server \
  --model $SUBMODEL_PATH \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --distributed-executor-backend ray \
  --port 8000" "$LOG"
