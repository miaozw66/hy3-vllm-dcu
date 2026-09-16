#!/bin/bash
# Single-node TP=8 full 80-layer HY3 launch script WITHOUT speculative decoding.
# Baseline for MTP correctness comparison (non-losslessness check).
#
# Usage:
#   bash deploy/run_tp8_single_80l.sh               # eager mode (-O0, default)
#   MODE=graph bash deploy/run_tp8_single_80l.sh    # CUDA graph mode (-O1)
#   DEBUG_MODE=1 bash deploy/run_tp8_single_80l.sh  # full INFO logs (debug)
set -e

source "$(dirname "$0")/env.sh"

# ── Environment ────────────────────────────────────────────
export NCCL_DEBUG=WARN
# AITER: Disabled - CK kernels not compiled for gfx928
export VLLM_ROCM_USE_AITER=0
# MoE tuning config
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR
# PP tokid broadcast guard (only relevant for PP>1; harmless here)
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1
# NCCL/RCCL tuning: MUST stay identical across MTP/baseline launches.
# Different reduction numerics cross W8A8 INT8 quantization boundaries and
# flip logits at near-tie positions, producing spurious divergence between
# runs (found 2026-08-20: the "req1" MTP divergence was purely this env
# artifact, not an MTP bug).
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

MODE=${MODE:-eager}
DEBUG=${DEBUG_MODE:-0}

# ── Logging control ────────────────────────────────────────
if [ "$DEBUG" = "1" ]; then
    LOG_ARGS=""
else
    export VLLM_LOGGING_LEVEL=WARNING
    export VLLM_DISABLE_LOG_LOGO=1
    LOG_ARGS="--disable-uvicorn-access-log --uvicorn-log-level warning"
fi

mkdir -p "$LOG_DIR"

echo "=== Full 80-Layer HY3 Single-Node TP=8 Launch (baseline, no MTP; mode=$MODE, debug=$DEBUG) ==="
echo "Started at: $(date)"
echo "Model: $MODEL_PATH"
echo "Log dir: $LOG_DIR"
echo ""

# ── GPU memory preflight ────────────────────────────────────
python3 "$PROJECT_ROOT/deploy/check_gpu_memory.py" \
    --expected-gpus "$GPU_COUNT" --min-free-mib 60000

# ── Clean up residual processes ────────────────────────────
pkill -9 -f vllm.entrypoints 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f Worker_TP 2>/dev/null || true
sleep 2

NODE0_LOG="$LOG_DIR/vllm_tp8_single_80l_${MODE}_$(date +%m%d_%H%M).log"
echo "  Log: $NODE0_LOG"

# ── Build launch args per mode ─────────────────────────────
COMMON_ARGS="--model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --distributed-timeout-seconds $DISTRIBUTED_TIMEOUT_SECONDS \
  --no-async-scheduling \
  --port 8000 \
  $LOG_ARGS"

if [ "$MODE" = "graph" ]; then
    MODE_ARGS="-O1"
else
    MODE_ARGS="-O0"
fi

# Use script -f for unbuffered output
script -f -c "python3 -u -m vllm.entrypoints.openai.api_server $COMMON_ARGS $MODE_ARGS" "$NODE0_LOG" 2>&1 &
NODE0_PID=$!

echo "[$(date)] Server PID: $NODE0_PID"
echo "[$(date)] Waiting for server to start..."
echo ""
echo "To check status: curl -s http://localhost:8000/health"
echo "Log: $NODE0_LOG"

# Wait for server to be ready
for i in $(seq 1 360); do
    if curl --noproxy '*' -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] ✓ Server is ready! (after $((i*10)) seconds)"
        echo ""
        echo "=== To send a test request ==="
        echo "curl -s http://localhost:8000/v1/completions \\"
        echo "  -H \"Content-Type: application/json\" \\"
        echo "  -d '{\"model\":\"$MODEL_PATH\",\"prompt\":\"中国的首都是\",\"max_tokens\":32}'"
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "[$(date)] Server did not start within 60 minutes. Check log: $NODE0_LOG"
exit 1
