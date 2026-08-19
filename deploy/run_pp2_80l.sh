#!/bin/bash
# Full 80-layer HY3 vLLM PP=2 launch script with dump enabled
# Launches on 2 nodes (8 GPUs total: 4/node, TP=4, PP=2)
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_pp2_80l.sh
#
# Prerequisites:
#   - Node 0 (MASTER_ADDR) and Node 1 (NODE1_IP) accessible via SSH
#   - vLLM 0.18.1 installed on both nodes
#   - Model path accessible on both nodes (NFS or Docker volume)
#
# HY3/DCU PP fixes (2026-08-12, validated at ~12.5 tok/s with correct output):
#   1. PP group uses gloo backend (RCCL 2.22.3 P2P isend/irecv is corrupt).
#      Gloo's D2H copies do NOT wait on any CUDA stream event, so
#      vllm/compilation/cuda_graph.py records the last cudagraph replay
#      stream and gpu_worker.py synchronizes exactly that stream before
#      isend_tensor_dict (otherwise output is garbage).
#   2. --no-async-scheduling: with async scheduling, the engine's per-step
#      input token ids lag one step behind on non-last PP ranks; vLLM
#      compensates by broadcasting sampled token ids from the last rank
#      (gpu_model_runner._pp_receive_prev_sampled_token_ids_to_input_batch).
#      On gloo that broadcast costs ~1.45s PER STEP (CPU stack, ~0.65 tok/s).
#      Sync scheduling keeps input ids fresh without the broadcast and
#      restores ~12.5 tok/s. (VLLM_HY3_SKIP_PP_TOKID_BCAST=1 also guards the
#      broadcast code path if async scheduling is ever re-enabled.)
set -e

source "$(dirname "$0")/env.sh"

# ── Environment ────────────────────────────────────────────
export NCCL_SOCKET_IFNAME=$NIC
export NCCL_DEBUG=WARN
# AITER: Disabled — CK kernels not compiled for gfx928
export VLLM_ROCM_USE_AITER=0
# MoE tuning config
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR

MASTER_PORT=29511

# Torch profiler: must be configured at launch time for /start_profile to exist.
# Same absolute path on both nodes (node1 container /data is the shared NFS).
# NOTE: after stop_profile the engine main loop dies on DCU (torch profiler
# teardown issue). Enable only for trace capture, then disable for serving.
# 最新 trace（修复后配置）: torch1_cudagraph_20260813_2 (2026-08-13 采集)
# PROFILER_DIR=$PROJECT_ROOT/profiles/torch1_cudagraph_20260813_2
# PROFILER_ARGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$PROFILER_DIR"

# ── Speculative decoding (HY3 MTP) ─────────────────────────
# HY3 MTP draft: checkpoint 有 1 个 nextn 层（num_nextn_predict_layers=1），
# 投机深度 = 1。MTP drafter 使用 rank-local PP=1，并仅在目标模型最后一个
# PP rank（node1）构造。
# node1 走 base64 传递（三层 shell 转义，沿用 profiler 方案）。
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
SPEC_CONFIG_B64=$(echo -n "$SPEC_CONFIG" | base64 -w0)

DUMP_DIR=$PROJECT_ROOT/dumps/pp2_80l

mkdir -p "$DUMP_DIR"
mkdir -p "$LOG_DIR"
# mkdir -p "$PROFILER_DIR"

echo "=== Full 80-Layer HY3 PP=2 Launch ==="
echo "Started at: $(date)"
echo "Model: $MODEL_PATH"
echo "Dump dir: $DUMP_DIR"
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

# ── GPU memory preflight ─────────────────────────────────────
echo "[$(date)] Checking Node 0 GPU memory..."
python3 "$PROJECT_ROOT/deploy/check_gpu_memory.py" \
    --expected-gpus "$GPU_COUNT" --min-free-mib 60000
echo "[$(date)] Checking Node 1 GPU memory..."
_remote_exec "$NODE1_IP" "" \
    "python3 $PROJECT_ROOT/deploy/check_gpu_memory.py --expected-gpus $GPU_COUNT --min-free-mib 60000"

# ── Launch Node 1 first (PP follower) ──────────────────────
echo "[$(date)] Launching Node 1 (PP rank 1, $NODE1_IP)..."
_remote_exec "$NODE1_IP" \
    "-e NCCL_SOCKET_IFNAME=$NIC -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=1 -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e RCCL_BUFFSIZE=$RCCL_BUFFSIZE -e NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS -e NCCL_PROTO=$NCCL_PROTO -e NCCL_ALGO=$NCCL_ALGO -e PYTHONUNBUFFERED=1 -e VLLM_ROCM_USE_AITER=0 -e VLLM_HY3_SKIP_PP_TOKID_BCAST=1 -e VLLM_RPC_TIMEOUT=$VLLM_RPC_TIMEOUT -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
    "rm -f /tmp/node1_80l.log
     echo $SPEC_CONFIG_B64 | base64 -d > /tmp/spec_config.json
     python3 -u -m vllm.entrypoints.openai.api_server \
       --model $MODEL_PATH \
       --pipeline-parallel-size 2 \
       --tensor-parallel-size 4 \
       --nnodes 2 \
       --node-rank 1 \
       --master-addr $MASTER_ADDR \
       --master-port $MASTER_PORT \
       --trust-remote-code \
       --max-model-len 16384 \
       --gpu-memory-utilization 0.90 \
       --enable-auto-tool-choice \
       --tool-call-parser hy_v3 \
       --distributed-timeout-seconds $DISTRIBUTED_TIMEOUT_SECONDS \
       --no-async-scheduling \
       -O0 \
       --speculative-config \\\"\\\$(cat /tmp/spec_config.json)\\\" \
       --port 8000 \
       $PROFILER_ARGS \
       > /tmp/node1_80l.log 2>&1 &" &
NODE1_SSH_PID=$!
echo "[$(date)] Node 1 SSH PID: $NODE1_SSH_PID"

# Give node 1 time to start
sleep 5

# ── Launch Node 0 (PP leader) ──────────────────────────────
echo "[$(date)] Launching Node 0 (PP rank 0, $MASTER_ADDR)..."
# NOTE: HY3 dump (VLLM_HY3_DUMP_DIR/SKIP) disabled when using CUDA graph.
# sitecustomize.py's _patched_model_forward uses open() which Dynamo cannot trace.
# Use --enforce-eager if dump logs are needed.

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
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --distributed-timeout-seconds $DISTRIBUTED_TIMEOUT_SECONDS \
  --no-async-scheduling \
  -O0 \
  --speculative-config '$SPEC_CONFIG' \
  --port 8000 \
  $PROFILER_ARGS" "$NODE0_LOG" 2>&1 &
NODE0_PID=$!

echo "[$(date)] Node 0 PID: $NODE0_PID"
echo "[$(date)] Waiting for server to start (this may take 2-5 minutes)..."
echo ""
echo "To check status: curl -s http://localhost:8000/health"
echo "To check dumps: ls $DUMP_DIR/"
echo "Node 0 log: $NODE0_LOG"
echo "Node 1 log: /tmp/node1_80l.log (inside Docker on node 1)"

# Wait for server to be ready
for i in $(seq 1 1080); do
    if curl --noproxy '*' --fail --silent http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] ✓ Server is ready! (after ${i}0 seconds)"
        echo ""
        echo "=== To send a test request ==="
        echo "curl -s http://localhost:8000/v1/completions \\"
        echo "  -H \"Content-Type: application/json\" \\"
        echo "  -d '{\"model\":\"hy3\",\"prompt\":\"中国的首都是\",\"max_tokens\":1}'"
        echo ""
        echo "=== To stream each decoded token ==="
        echo "python3 $PROJECT_ROOT/deploy/stream_completion.py '中国的首都是' --model '$MODEL_PATH' --max-tokens 32"
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "[$(date)] Server did not start within 180 minutes. Check logs."
exit 1
