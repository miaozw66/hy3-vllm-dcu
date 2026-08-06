#!/bin/bash
# PP=2 带 dump 的 vLLM 启动脚本
# 用法: bash run_pp_dump.sh

set -e
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONUNBUFFERED=1
export VLLM_HY3_DUMP_DIR=/tmp/pp_dump

rm -rf /tmp/pp_dump

echo "=== Starting vLLM PP=2 with dump ==="
echo "Dump dir: $VLLM_HY3_DUMP_DIR"
echo "Log file: /tmp/vllm_pp_dump.log"
echo "Started at: $(date)"

exec python -m vllm.entrypoints.openai.api_server \
  --model /data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --max-model-len 8192 \
  --enforce-eager \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85
