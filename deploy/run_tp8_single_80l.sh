#!/bin/bash
# Single-node TP=8 full 80-layer HY3 launch script
# Launches on one machine with 8 GPUs (TP=8, no PP).
#
# Usage:
#   1. Edit deploy/env.sh with your machine configuration
#   2. bash deploy/run_tp8_single_80l.sh           # CUDA graph mode (default)
#      MODE=eager bash deploy/run_tp8_single_80l.sh  # enforce-eager mode
#      DEBUG_MODE=1 bash deploy/run_tp8_single_80l.sh  # full INFO logs (debug)
#
# Notes:
#   - PP=2-specific fixes (gloo PP group, tokid broadcast) are not used when
#     PP=1; the wheel patches are harmless in TP-only mode.
#   - CUDA graph (-O1) with TP=8 on a single node was not validated on the
#     old 2-node deployment. Use MODE=eager to establish a functional
#     baseline first.
set -e

source "$(dirname "$0")/env.sh"

# ── Environment ────────────────────────────────────────────
# Single-node TP=8 communicates over PCIe; no socket ifname needed.
export NCCL_DEBUG=WARN
# AITER: Disabled — CK kernels not compiled for gfx928
export VLLM_ROCM_USE_AITER=0
# MoE tuning config
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR
# PP tokid broadcast guard (only relevant for PP>1; harmless here)
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1

MODE=${MODE:-graph}
DEBUG=${DEBUG_MODE:-0}

# ── Logging control ────────────────────────────────────────
# Default: quiet mode (WARNING+ only, no uvicorn access log, no per-10s engine stats).
# DEBUG_MODE=1: full INFO logs for troubleshooting.
if [ "$DEBUG" = "1" ]; then
    LOG_ARGS=""
else
    export VLLM_LOGGING_LEVEL=WARNING
    export VLLM_DISABLE_LOG_LOGO=1
    LOG_ARGS="--disable-uvicorn-access-log --uvicorn-log-level warning"
fi

mkdir -p "$LOG_DIR"

echo "=== Full 80-Layer HY3 Single-Node TP=8 Launch (mode=$MODE, debug=$DEBUG) ==="
echo "Started at: $(date)"
echo "Model: $MODEL_PATH"
echo "Log dir: $LOG_DIR"
echo ""

# ── Clean up residual processes ────────────────────────────
pkill -9 -f vllm.entrypoints 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f Worker_TP 2>/dev/null || true
sleep 2

NODE0_LOG="$LOG_DIR/vllm_tp8_80l_${MODE}_$(date +%m%d_%H%M).log"
echo "  Log: $NODE0_LOG"

# ── Build launch args per mode ─────────────────────────────
COMMON_ARGS="--model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --distributed-timeout-seconds 1800 \
  --port 8000 \
  $LOG_ARGS"

if [ "$MODE" = "eager" ]; then
    MODE_ARGS="--enforce-eager"
else
    # CUDA graph mode: sync scheduling is required (see README problem 7)
    MODE_ARGS="-O1 --no-async-scheduling"
fi

# Use script -f for unbuffered output
script -f -c "python3 -u -m vllm.entrypoints.openai.api_server $COMMON_ARGS $MODE_ARGS" "$NODE0_LOG" 2>&1 &
NODE0_PID=$!

echo "[$(date)] Server PID: $NODE0_PID"
echo "[$(date)] Waiting for server to start (this may take 2-15 minutes; first request triggers torch.compile)..."
echo ""
echo "To check status: curl -s http://localhost:8000/health"
echo "Log: $NODE0_LOG"

# Wait for server to be ready
for i in $(seq 1 180); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] ✓ Server is ready! (after ${i}0 seconds)"
        echo ""
        echo "=== To send a test request ==="
        echo "curl -s http://localhost:8000/v1/completions \\"
        echo "  -H \"Content-Type: application/json\" \\"
        echo "  -d '{\"model\":\"$MODEL_PATH\",\"prompt\":\"中国的首都是\",\"max_tokens\":1}'"
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "[$(date)] Server did not start within 30 minutes. Check log: $NODE0_LOG"
exit 1
