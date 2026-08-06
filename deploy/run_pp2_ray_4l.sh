#!/bin/bash
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_HY3_DUMP_DIR=/data/mzw/vllm-hy3/dumps/pp2_4l
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
  --model /data/mzw/vllm-hy3/submodel_debug/test4 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --distributed-executor-backend ray \
  --port 8000" "$LOG"
