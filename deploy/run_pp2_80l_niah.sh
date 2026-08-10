#!/bin/bash
# Full 80-layer HY3 vLLM PP=2 launch — NIAH test configuration
# max-model-len=262144, gpu-memory-utilization=0.90
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_pp2_80l_niah.sh
#
# Prerequisites:
#   - Node 0 (MASTER_ADDR) and Node 1 (NODE1_IP) accessible via SSH
#   - vLLM 0.18.1 installed on both nodes
#   - Model path accessible on both nodes (NFS or Docker volume)
set -e

source "$(dirname "$0")/env.sh"

# ── Environment ────────────────────────────────────────────
export NCCL_SOCKET_IFNAME=$NIC
export NCCL_DEBUG=WARN
# AITER: Disabled — CK kernels not compiled for gfx928
export VLLM_ROCM_USE_AITER=0
# MoE tuning config
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR

MASTER_PORT=29551

mkdir -p "$LOG_DIR"

echo "=== Full 80-Layer HY3 PP=2 NIAH Launch ==="
echo "Started at: $(date)"
echo "Model: $MODEL_PATH"
echo "max-model-len: 262144 | gpu-memory-utilization: 0.90"
echo "Log dir: $LOG_DIR"
echo ""

# ── Helper: remote execution (Docker or bare-metal) ────────
_remote_exec() {
    local node_ip=$1; shift
    local env_vars="$1"; shift
    local cmd="$1"
    if [ -n "$DOCKER_NAME" ]; then
        ssh -o StrictHostKeyChecking=no "$node_ip" \
            "docker exec $env_vars $DOCKER_NAME bash -c \"$cmd\""
    else
        ssh -o StrictHostKeyChecking=no "$node_ip" \
            "bash -c \"$env_vars $cmd\""
    fi
}

# ── Launch Node 1 first (PP follower) ──────────────────────
echo "[$(date)] Launching Node 1 (PP rank 1, $NODE1_IP)..."
# Clean up any residual processes on node 1
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$NODE1_IP" \
    "if [ -n \"$DOCKER_NAME\" ]; then docker exec $DOCKER_NAME bash -c 'pkill -9 -f vllm.entrypoints 2>/dev/null || true; pkill -9 -f VLLM:: 2>/dev/null || true'; else pkill -9 -f vllm.entrypoints 2>/dev/null || true; pkill -9 -f VLLM:: 2>/dev/null || true; fi; sleep 2; echo cleaned" 2>/dev/null || true

_remote_exec "$NODE1_IP" \
    "-e NCCL_SOCKET_IFNAME=$NIC -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e RCCL_BUFFSIZE=$RCCL_BUFFSIZE -e NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS -e NCCL_PROTO=$NCCL_PROTO -e NCCL_ALGO=$NCCL_ALGO -e PYTHONUNBUFFERED=1 -e VLLM_ROCM_USE_AITER=0" \
    "rm -f /tmp/node1_80l_niah.log
     python3 -u -m vllm.entrypoints.openai.api_server \
       --model $MODEL_PATH \
       --pipeline-parallel-size 2 \
       --tensor-parallel-size 4 \
       --nnodes 2 \
       --node-rank 1 \
       --master-addr $MASTER_ADDR \
       --master-port $MASTER_PORT \
       --trust-remote-code \
       --enforce-eager \
       --max-model-len 262144 \
       --gpu-memory-utilization 0.90 \
       --model-loader-extra-config '{\"enable_multithread_load\": true}' \
       --port 8000 \
       > /tmp/node1_80l_niah.log 2>&1 &" &
NODE1_SSH_PID=$!
echo "[$(date)] Node 1 SSH PID: $NODE1_SSH_PID"

# Give node 1 time to start (short sleep to avoid rate limiting)
sleep 1.5

# ── Launch Node 0 (PP leader) ──────────────────────────────
echo "[$(date)] Launching Node 0 (PP rank 0, $MASTER_ADDR)..."
NODE0_LOG="$LOG_DIR/vllm_node0_pp2_80l_niah_$(date +%m%d_%H%M).log"
echo "  Node 0 log: $NODE0_LOG"

script -f -c "python3 -u -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --model-loader-extra-config '{\"enable_multithread_load\": true}' \
  --port 8000" "$NODE0_LOG" 2>&1 &
NODE0_PID=$!

echo "[$(date)] Node 0 PID: $NODE0_PID"
echo "[$(date)] Waiting for server to start (this may take 5-10 minutes for 256K pre-allocation)..."
echo ""
echo "To check status: curl -s http://localhost:8000/health"
echo "Node 0 log: $NODE0_LOG"
echo "Node 1 log: /tmp/node1_80l_niah.log (inside Docker on node 1)"

# Wait for server to be ready (up to 20 minutes for 256K)
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] Server is ready! (after ${i}0 seconds)"
        echo ""
        echo "=== To run NIAH tests ==="
        echo "python3 niah_test.py --endpoint http://localhost:8000 --lengths 4096,8192,..."
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "[$(date)] Server did not start within 20 minutes. Check logs."
exit 1
