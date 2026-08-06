#!/bin/bash
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_HY3_DUMP_DIR=/data/mzw/vllm-hy3/dumps/single_tp4_4l
export VLLM_HY3_DUMP_SKIP=2

LOG=/tmp/vllm_single_tp4_4l_v2.log
rm -f "$LOG"
rm -rf "$VLLM_HY3_DUMP_DIR"
mkdir -p "$VLLM_HY3_DUMP_DIR"

echo "Starting single TP=4 4-layer server at $(date)"
echo "Dump dir: $VLLM_HY3_DUMP_DIR"
echo "Log: $LOG"

exec script -f -c "python3 -u -m vllm.entrypoints.openai.api_server \
  --model /data/mzw/vllm-hy3/submodel_debug/test4 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4 \
  --port 8001" "$LOG"
