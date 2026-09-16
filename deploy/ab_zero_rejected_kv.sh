#!/usr/bin/env bash
# Service-level A/B for the HY3/DCU MTP per-step KV clearing.
#
# Two sequential runs of the SAME service config, toggling only
# VLLM_HY3_ZERO_REJECTED_KV (1 = clearing ON, 0 = official logic / no clearing):
#   TP=8, MTP depth=3, TRITON_ATTN (AITER off), -O1 PIECEWISE graph,
#   no prefix caching, no async scheduling, standard 1K benchmark cell.
# The torch profiler is deliberately NOT enabled: it inflates TPOT ~21%.
#
# Measures, per run: TPOT/ITL (bench_hy3.py), bitwise token IDs on the 10 fixed
# greedy prompts (mtp_correctness_test.py), and the server's reported draft
# acceptance rate.
#
# Lifecycle: owns and terminates only its own process group (Rule 12) and never
# pkills unrelated processes. Results land under profiles/, never /tmp.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/deploy/env.sh"

PORT="${PORT:-18001}"
DRAFT="${DRAFT:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-140000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1 2 4 8 16}"
ZTH_W8A8_MODE="${ZTH_W8A8_MODE:-use}"
DELIVER="${ZTH_W8A8_DELIVER:-$PROJECT_ROOT/算子优化/deliver}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"
AB_TAG="${AB_TAG:-zero_ab_$(date +%m%d_%H%M%S)}"
AB_DIR="$PROJECT_ROOT/profiles/$AB_TAG"

SERVER_PID=""
SERVER_PGID=""

cleanup() {
    [[ -n "$SERVER_PID" ]] || return 0
    local target="-${SERVER_PGID:-$SERVER_PID}" i
    if [[ -n "$SERVER_PGID" && "$SERVER_PGID" == "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]]; then
        echo "refusing to signal own process group $SERVER_PGID" >&2
        return 0
    fi
    kill -0 -- "$target" 2>/dev/null || return 0
    kill -TERM -- "$target" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for i in $(seq 1 120); do
        kill -0 -- "$target" 2>/dev/null || return 0
        sleep 1
    done
    echo "process group $target outlived SIGTERM (120s); sending SIGKILL" >&2
    kill -KILL -- "$target" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    for i in $(seq 1 30); do
        kill -0 -- "$target" 2>/dev/null || return 0
        sleep 1
    done
    echo "process group $target survived SIGKILL" >&2
    SERVER_PID=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -d "$DELIVER/python" || ! -d "$DELIVER/so_cache" ]]; then
    echo "Invalid ZTH deliver directory: $DELIVER" >&2
    exit 2
fi
mkdir -p "$AB_DIR" "$PROJECT_ROOT/logs"

export PYTHONPATH="$DELIVER/python${PYTHONPATH:+:$PYTHONPATH}"
export ZTH_W8A8_CACHE="$DELIVER/so_cache"
export ZTH_W8A8_MODE
export NCCL_DEBUG=WARN
export VLLM_ROCM_USE_AITER=0
export VLLM_TUNED_CONFIG_FOLDER="$MOE_CONFIG_DIR"
export VLLM_HY3_SKIP_PP_TOKID_BCAST=1
# NCCL_PROTO/ALGO stay at the env.sh LL/Tree default; both runs share them.

run_one() {
    local zero="$1"
    local tag="zero${zero}"
    local log="$PROJECT_ROOT/logs/ab_${AB_TAG}_${tag}.log"

    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port $PORT still in use before $tag; aborting." >&2
        return 1
    fi
    python3 "$PROJECT_ROOT/deploy/check_gpu_memory.py" \
        --expected-gpus "$GPU_COUNT" --min-free-mib 60000 || return 1

    echo "=== run $tag (VLLM_HY3_ZERO_REJECTED_KV=$zero) log=$log ==="
    VLLM_HY3_ZERO_REJECTED_KV="$zero" setsid python3 -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --tensor-parallel-size 8 \
        --trust-remote-code \
        --quantization compressed-tensors \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --cudagraph-capture-sizes $CAPTURE_SIZES \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --no-enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser hy_v3 \
        --distributed-timeout-seconds "$DISTRIBUTED_TIMEOUT_SECONDS" \
        --no-async-scheduling \
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT}" \
        --host 127.0.0.1 \
        --port "$PORT" \
        -O1 \
        >"$log" 2>&1 &
    SERVER_PID=$!
    SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
    echo "  pid=$SERVER_PID pgid=$SERVER_PGID"

    local ok=0
    for _ in $(seq 1 2400); do
        if curl --noproxy '*' --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
            ok=1; break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  server exited during startup; see $log" >&2
            return 1
        fi
        sleep 1
    done
    [[ "$ok" -eq 1 ]] || { echo "  server never became healthy" >&2; return 1; }
    echo "  server healthy"

    # Confirm the actual runtime config from the process command line.
    ps -o pid,pgid,cmd -p "$SERVER_PID" | tail -1
    grep -c "speculative" "$log" >/dev/null 2>&1 || true

    curl --noproxy '*' --silent --show-error --max-time 600 -o /dev/null \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_PATH\",\"prompt\":\"法国的首都是\",\"max_tokens\":4}" \
        "http://127.0.0.1:$PORT/v1/completions" || true

    echo "  bench 1024:1024 c1 ..."
    python3 "$PROJECT_ROOT/benchmark/bench_hy3.py" \
        --endpoint "http://127.0.0.1:$PORT" \
        --matrix "1024:1024" --concurrency "1" --runs 3 \
        --tag "${AB_TAG}_${tag}_c1" \
        --out "$AB_DIR/bench_c1_${tag}.json" \
        --timeout "$BENCH_TIMEOUT" || echo "  bench c1 rc=$?"

    echo "  bench 1024:1024 c4 ..."
    python3 "$PROJECT_ROOT/benchmark/bench_hy3.py" \
        --endpoint "http://127.0.0.1:$PORT" \
        --matrix "1024:1024" --concurrency "4" --runs 3 \
        --tag "${AB_TAG}_${tag}_c4" \
        --out "$AB_DIR/bench_c4_${tag}.json" \
        --timeout "$BENCH_TIMEOUT" || echo "  bench c4 rc=$?"

    echo "  correctness 10 greedy prompts x128 ..."
    python3 "$PROJECT_ROOT/benchmark/mtp_correctness_test.py" \
        --endpoint "http://127.0.0.1:$PORT" \
        --output "$AB_DIR/correctness_${tag}.json" \
        --max-tokens 128 || echo "  correctness rc=$?"

    grep -iE "acceptance rate|Avg Draft|spec_decode" "$log" | tail -20 \
        > "$AB_DIR/accept_${tag}.txt" || true
    cp "$log" "$AB_DIR/server_${tag}.log"

    cleanup
    for i in $(seq 1 60); do
        python3 "$PROJECT_ROOT/deploy/check_gpu_memory.py" \
            --expected-gpus "$GPU_COUNT" --min-free-mib 60000 >/dev/null 2>&1 && break
        sleep 2
    done
    echo "=== run $tag done ==="
}

run_one 1 || { echo "zero-ON run failed"; }
run_one 0 || { echo "zero-OFF run failed"; }

echo "AB_DIR=$AB_DIR"
ls -la "$AB_DIR"
