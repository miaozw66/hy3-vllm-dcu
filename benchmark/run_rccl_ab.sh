#!/bin/bash
# RCCL parameter A/B harness for the allreduce-bound decode diagnosis.
#
# For one named configuration: start the baseline server (SPEC=none, TP=8, -O1)
# with the RCCL/NCCL env overrides for that config, wait for readiness, run the
# random-ids benchmark (conc1 = per-token decode latency, conc16 = throughput),
# then tear the server down.  No profiler, no clock changes.
#
# Usage:
#   bash benchmark/run_rccl_ab.sh base      # Ring/Simple/nchan4  (current)
#   bash benchmark/run_rccl_ab.sh tree      # Tree/Simple/nchan4
#   bash benchmark/run_rccl_ab.sh ll        # Ring/LL/nchan4
#   bash benchmark/run_rccl_ab.sh tree_ll   # Tree/LL/nchan4
#   bash benchmark/run_rccl_ab.sh nchan1    # Ring/Simple/nchan1
#   bash benchmark/run_rccl_ab.sh nchan8    # Ring/Simple/nchan8
#
# Results land in ${AB_DIR}/<NAME>.json (per-config bench snapshot).
set +e
source "$(dirname "$0")/../deploy/env.sh"

export VLLM_ROCM_USE_AITER=0
export VLLM_TUNED_CONFIG_FOLDER=$MOE_CONFIG_DIR
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1
export RCCL_BUFFSIZE=8388608

AB_DIR=${AB_DIR:-/tmp/rccl_ab}
mkdir -p "$AB_DIR"

NAME=${1:?usage: run_rccl_ab.sh <base|tree|ll|tree_ll|nchan1|nchan8>}
case "$NAME" in
  base)    export NCCL_ALGO=Ring   NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 ;;
  tree)    export NCCL_ALGO=Tree   NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 ;;
  ll)      export NCCL_ALGO=Ring   NCCL_PROTO=LL     NCCL_MIN_NCHANNELS=4 ;;
  tree_ll) export NCCL_ALGO=Tree   NCCL_PROTO=LL     NCCL_MIN_NCHANNELS=4 ;;
  nchan1)  export NCCL_ALGO=Ring   NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 ;;
  nchan8)  export NCCL_ALGO=Ring   NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=8 ;;
  *) echo "unknown config: $NAME"; exit 1 ;;
esac

LOG="$LOG_DIR/vllm_tp8_rccl_ab_${NAME}_$(date +%m%d_%H%M).log"
OUT="$AB_DIR/$NAME.json"
echo "=== RCCL A/B config: $NAME ==="
echo "  NCCL_ALGO=$NCCL_ALGO NCCL_PROTO=$NCCL_PROTO NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS"
echo "  log: $LOG"
echo "  out: $OUT"

ARGS="--model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --distributed-timeout-seconds $DISTRIBUTED_TIMEOUT_SECONDS \
  --port 8000 \
  -O1 --no-async-scheduling"

# ── start server in background ────────────────────────────────────────────────
echo "starting server ..."
setsid bash -c "exec python3 -u -m vllm.entrypoints.openai.api_server $ARGS" >"$LOG" 2>&1 &
SERVER_SID=$!
echo "  server setsid=$SERVER_SID log=$LOG"

# ── wait for readiness (graph capture of 80-layer MoE can take minutes) ───────
READY=0
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "server READY after ~$((i*5))s"; break; fi
  if ! kill -0 "$SERVER_SID" 2>/dev/null; then
    echo "SERVER DIED during startup; tail of log:"; tail -n 20 "$LOG"; break
  fi
  sleep 5
done

if [ "$READY" != "1" ]; then
  echo "server not ready in time; tearing down"; pkill -9 -f 'vllm.entrypoin[t]s'; sleep 3; exit 1
fi

# ── run benchmark: conc1 (decode latency) + conc16 (throughput) ───────────────
echo "running benchmark (2048/256 conc1 + conc16) ..."
python3 benchmark/bench_random_ids.py \
  --cases 2048/256 --concurrencies 1,16 --runs 1 --ignore-eos \
  --output "$OUT"
RC=$?
echo "benchmark rc=$RC"

# ── tear down server tree ─────────────────────────────────────────────────────
pkill -9 -f 'vllm.entrypoin[t]s'
sleep 2
pkill -9 -f 'EngineCo[r]e' 2>/dev/null
pkill -9 -f 'vllm_wrapper_hyv[3]' 2>/dev/null
sleep 2
echo "=== $NAME done (rc=$RC); result: $OUT ==="
exit $RC
