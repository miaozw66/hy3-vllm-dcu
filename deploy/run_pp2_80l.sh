#!/bin/bash
# Full 80-layer HY3 vLLM PP=2 launch script with dump enabled
# Launches on 2 nodes (8 GPUs total: 4/node, TP=4, PP=2)
#
# Usage:
#   bash run_pp2_80l.sh
#
# Prerequisites:
#   - Node 0 (10.18.17.71) and Node 1 (10.18.17.74) accessible
#   - Node 1 has Docker container "mmh_qwen_opt" with same vLLM installation
#   - /data/mzw is NFS-mounted on both nodes
#   - vLLM 0.18.1 installed (with dump hooks already patched on both nodes)
set -e

# ── Environment ────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# RCCL communication tuning
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# AITER: Disabled — CK kernels not compiled for gfx928
export VLLM_ROCM_USE_AITER=0
# MoE tuning config
export VLLM_TUNED_CONFIG_FOLDER=/data/mzw/vllm-hy3/moe_configs

DUMP_DIR=/data/mzw/vllm-hy3/dumps/pp2_80l
LOG_DIR=/data/mzw/vllm-hy3/logs
MASTER_ADDR=10.18.17.71
MASTER_PORT=29511
NODE1_IP=10.18.17.74
DOCKER_NAME=mmh_qwen_opt
MODEL_PATH=/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master

# ── Cleanup previous dumps ─────────────────────────────────
rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"
mkdir -p "$LOG_DIR"

echo "=== Full 80-Layer HY3 PP=2 Launch ==="
echo "Started at: $(date)"
echo "Model: $MODEL_PATH"
echo "Dump dir: $DUMP_DIR"
echo "Log dir: $LOG_DIR"
echo ""

# ── Launch Node 1 first (PP follower) ──────────────────────
echo "[$(date)] Launching Node 1 (PP rank 1, 10.18.17.74)..."
# Clean up any residual processes on node 1 (must match api_server, VLLM::EngineCore, and VLLM::Worker_*)
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$NODE1_IP" \
  "docker exec $DOCKER_NAME bash -c 'pkill -9 -f vllm.entrypoints 2>/dev/null || true; pkill -9 -f VLLM:: 2>/dev/null || true; sleep 2; echo cleaned'" 2>/dev/null || true

ssh -o StrictHostKeyChecking=no "$NODE1_IP" \
  "docker exec -e NCCL_SOCKET_IFNAME=eno1 \
      -e NCCL_DEBUG=WARN \
      -e NCCL_IB_DISABLE=1 \
      -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
      -e RCCL_BUFFSIZE=8388608 \
      -e NCCL_MIN_NCHANNELS=4 \
      -e NCCL_PROTO=Simple \
      -e NCCL_ALGO=Ring \
      -e PYTHONUNBUFFERED=1 \
      -e VLLM_ROCM_USE_AITER=0 \
      -e VLLM_HY3_DUMP_DIR=$DUMP_DIR \
      -e VLLM_HY3_DUMP_SKIP=2 \
      $DOCKER_NAME bash -c \"
        rm -f /tmp/node1_80l.log
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
          --enable-auto-tool-choice \
          --tool-call-parser hy_v3 \
          --port 8000 \
          > /tmp/node1_80l.log 2>&1 &
        echo \\\$!
      \"" &
NODE1_SSH_PID=$!
echo "[$(date)] Node 1 SSH PID: $NODE1_SSH_PID"

# Give node 1 time to start
sleep 5

# ── Launch Node 0 (PP leader) ──────────────────────────────
echo "[$(date)] Launching Node 0 (PP rank 0, 10.18.17.71)..."
export VLLM_HY3_DUMP_DIR="$DUMP_DIR"
export VLLM_HY3_DUMP_SKIP=2

NODE0_LOG="$LOG_DIR/vllm_node0_pp2_80l_$(date +%m%d_%H%M).log"
echo "  Node 0 log: $NODE0_LOG"

# Use script -f for unbuffered output
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
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --port 8000" "$NODE0_LOG" 2>&1 &
NODE0_PID=$!

echo "[$(date)] Node 0 PID: $NODE0_PID"
echo "[$(date)] Waiting for server to start (this may take 2-5 minutes)..."
echo ""
echo "To check status: curl -s http://localhost:8000/health"
echo "To check dumps: ls $DUMP_DIR/"
echo "Node 0 log: $NODE0_LOG"
echo "Node 1 log: /tmp/node1_80l.log (inside Docker on node 1)"

# Wait for server to be ready
for i in $(seq 1 180); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] ✓ Server is ready! (after ${i}0 seconds)"
        echo ""
        echo "=== To send a test request ==="
        echo "curl -s http://localhost:8000/v1/completions \\"
        echo "  -H \"Content-Type: application/json\" \\"
        echo "  -d '{\"model\":\"hy3\",\"prompt\":\"中国的首都是\",\"max_tokens\":1}'"
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "[$(date)] Server did not start within 30 minutes. Check logs."
exit 1
