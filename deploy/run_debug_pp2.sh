#!/bin/bash
# Debug PP=2 launch — test progressively larger max-model-len
# Usage: bash run_debug_pp2.sh <MAX_MODEL_LEN>
set -e

MAX_MODEL_LEN=${1:-8192}

export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# AITER: Enable Composable Kernel acceleration for Hygon DCU
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MHA=1

LOG_DIR=/data/mzw/vllm-hy3/logs
MASTER_ADDR=10.18.17.71
MASTER_PORT=29561
NODE1_IP=10.18.17.74
DOCKER_NAME=mmh_qwen_opt
MODEL_PATH=/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master

mkdir -p "$LOG_DIR"

echo "=== Debug PP=2 Launch: max-model-len=${MAX_MODEL_LEN} ==="
echo "Started at: $(date)"
echo ""

# ── Launch Node 1 first ──────────────────────────────────────
echo "[$(date)] Launching Node 1 (PP rank 1)..."
# Clean up any residual processes on node 1 (must match api_server, VLLM::EngineCore, and VLLM::Worker_*)
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$NODE1_IP" \
  "docker exec $DOCKER_NAME bash -c 'pkill -9 -f vllm.entrypoints 2>/dev/null || true; pkill -9 -f VLLM:: 2>/dev/null || true; sleep 2; echo cleaned'" 2>/dev/null || true
sleep 1

ssh -o StrictHostKeyChecking=no "$NODE1_IP" \
  "docker exec -e NCCL_SOCKET_IFNAME=eno1 \
      -e NCCL_DEBUG=WARN \
      -e NCCL_IB_DISABLE=1 \
      -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
      -e PYTHONUNBUFFERED=1 \
      -e VLLM_ROCM_USE_AITER=1 \
      -e VLLM_ROCM_USE_AITER_LINEAR=1 \
      -e VLLM_ROCM_USE_AITER_MOE=1 \
      -e VLLM_ROCM_USE_AITER_RMSNORM=1 \
      -e VLLM_ROCM_USE_AITER_MHA=1 \
      $DOCKER_NAME bash -c \"
        rm -f /tmp/node1_debug.log
        python3 -u /data/mzw/vllm-hy3/run_api_server.py \
          --model $MODEL_PATH \
          --pipeline-parallel-size 2 \
          --tensor-parallel-size 4 \
          --nnodes 2 \
          --node-rank 1 \
          --master-addr $MASTER_ADDR \
          --master-port $MASTER_PORT \
          --trust-remote-code \
          --enforce-eager \
          --max-model-len $MAX_MODEL_LEN \
          --gpu-memory-utilization 0.85 \
          --port 8000 \
          > /tmp/node1_debug.log 2>&1 &
        echo \\\$!
      \"" &
NODE1_SSH_PID=$!
echo "[$(date)] Node 1 SSH PID: $NODE1_SSH_PID"

# Give node 1 more time to start (matching original working script)
sleep 5

# ── Launch Node 0 ────────────────────────────────────────────
echo "[$(date)] Launching Node 0 (PP rank 0)..."
NODE0_LOG="$LOG_DIR/vllm_node0_debug_${MAX_MODEL_LEN}_$(date +%m%d_%H%M).log"
echo "  Node 0 log: $NODE0_LOG"

script -f -c "python3 -u /data/mzw/vllm-hy3/run_api_server.py \
  --model $MODEL_PATH \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization 0.85 \
  --port 8000" "$NODE0_LOG" 2>&1 &
NODE0_PID=$!

echo "[$(date)] Node 0 PID: $NODE0_PID"
echo "[$(date)] Waiting for server to start..."
echo ""

for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] ✓ Server is ready! (after ${i}0 seconds)"
        echo "Node 0 log: $NODE0_LOG"
        echo "Node 1 log: /tmp/node1_debug.log (Docker on node 1)"
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "[$(date)] Server did not start within 20 minutes."
echo "Node 0 log: $NODE0_LOG"
echo "Check Node 1: ssh 10.18.17.74 'docker exec mmh_qwen_opt tail -100 /tmp/node1_debug.log'"
exit 1
