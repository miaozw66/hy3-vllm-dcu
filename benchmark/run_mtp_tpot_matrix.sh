#!/bin/bash
# ============================================================
# HY3 MTP TPOT 统一无 profiler 测量基座（A/B 驱动）
#
# 目的（P0-3，见 benchmark/HY3_MTP_TRACE_VERIFICATION_PLAN_20260906.md §7）：
#   对"固定 SGLang 口径"的服务端启动 + SGLang-aligned 客户端，按 (上下文长度 ×
#   concurrency) 矩阵重定标真实无 profiler step 墙，并支持 cand_03 的
#   LL/Tree vs Simple/Ring A/B 对照——同一 spec/zth/服务端参数，仅 NCCL 协议不同。
#
# 服务端口径固定为（run_tp8_zth_mtp_sglang_align.sh，已落地 cand_03）：
#   -O1、--no-async-scheduling、capture sizes 1 2 4 8 16、max-num-seqs 48、
#   max-num-batched-tokens 4096、--quantization compressed-tensors、
#   --enable-auto-tool-choice --tool-call-parser hy_v3
#
# 用法示例（本脚本所有参数都是 env 变量覆盖，不支持 --flag）：
#   # cand_03 验收：MTP-3，8K/c1 (7168:96) + 1K/c1 校准，LL/Tree vs Simple/Ring 双侧
#   TAG=mtp3_8k_ab SIDES=base_sr SPEC=on DRAFT=3 \
#     MATRIX="7168:96,1024:1024" CONC="1,4" RUNS=2 \
#     CONFIRM_STOP_EXISTING_VLLM=YES \
#     bash benchmark/run_mtp_tpot_matrix.sh
#
#   # 只跑基准矩阵单侧（LL/Tree，MTP-3，用于 P0-3 定标）
#   SIDES=base MATRIX="1024:1024,2048:1024,4096:1024,7168:96,16384:1024" \
#     CONC="1,4" CONFIRM_STOP_EXISTING_VLLM=YES \
#     bash benchmark/run_mtp_tpot_matrix.sh
#
#   # baseline（无 speculative）对照
#   TAG=base_8k_ab SPEC=off SIDES=base_sr MATRIX="7168:96" CONC="1,4" \
#     CONFIRM_STOP_EXISTING_VLLM=YES bash benchmark/run_mtp_tpot_matrix.sh
#
# A/B 一致性要求：任何跨 run 数值对比，两侧被对比的 env（NCCL_PROTO/ALGO、
# RCCL_BUFFSIZE、NCCL_MIN_NCHANNELS、zth、spec、capture sizes、max-model-len）
# 除被测维度外必须一致，否则 W8A8 INT8 归约顺序漂移翻转 logits（2026-08-20）。
# ============================================================
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/deploy/env.sh"

# ── 参数（均可 env 覆盖） ─────────────────────────────────────
TAG="${TAG:-mtp_tpot}"
SIDES="${SIDES:-base_sr}"          # base | sr | base_sr
SPEC="${SPEC:-on}"                 # on=MTP | off=baseline
DRAFT="${DRAFT:-3}"
ZTH="${ZTH:-use}"                  # use | off | validate（deliver zth hook）
MATRIX="${MATRIX:-7168:96,1024:1024}"
CONC="${CONC:-1,4}"
RUNS="${RUNS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/benchmark/results_service_measurements}"
SERVER_WAIT_SECS="${SERVER_WAIT_SECS:-2400}"
GRACE_SECS="${GRACE_SECS:-150}"   # 启动宽限：此期间不做进程死亡检测
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
SETTLE_SECS="${SETTLE_SECS:-15}"
PORT="${PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

case "$ZTH" in use|off|validate) ;; *) echo "ZTH must be use|off|validate" >&2; exit 2 ;; esac
case "$SPEC" in on|off) ;; *) echo "SPEC must be on|off" >&2; exit 2 ;; esac

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── side 定义（被测维度 = NCCL 协议） ─────────────────────────
# 返回 (proto algo label) 到全局变量
# 可选 override：设 NCCL_PROTO_OVERRIDE/NCCL_ALGO_OVERRIDE（+可选
# SIDE_LABEL_OVERRIDE）可从外层注入任意协议组合（如 LL128/Tree），
# 覆盖 base/sr 默认；label 缺省按 "proto_algo" 小写自动生成。
set_side() {  # $1 = base | sr
    if [ -n "${NCCL_PROTO_OVERRIDE:-}" ]; then
        NCCL_PROTO_V="$NCCL_PROTO_OVERRIDE"
        NCCL_ALGO_V="${NCCL_ALGO_OVERRIDE:-Tree}"
        SIDE_LABEL="${SIDE_LABEL_OVERRIDE:-${NCCL_PROTO_V,,}_${NCCL_ALGO_V,,}}"
        return
    fi
    case "$1" in
        base) NCCL_PROTO_V="LL";     NCCL_ALGO_V="Tree";    SIDE_LABEL="ll_tree" ;;
        sr)   NCCL_PROTO_V="Simple"; NCCL_ALGO_V="Ring";    SIDE_LABEL="simple_ring" ;;
        *) echo "unknown side $1" >&2; exit 2 ;;
    esac
}

# ── 服务端进程管理 ───────────────────────────────────────────
stop_server() {
    pkill -9 -f vllm.entrypoints 2>/dev/null || true
    pkill -9 -f EngineCore 2>/dev/null || true
    pkill -9 -f Worker_TP 2>/dev/null || true
    sleep "$SETTLE_SECS"
}

wait_health() {  # $1 = server log（诊断用）
    local log="$1" waited=0 dead_rounds=0
    echo "    waiting for /health (timeout ${SERVER_WAIT_SECS}s, grace ${GRACE_SECS}s) ..." | tee -a "$DRV_LOG"
    while [ "$waited" -lt "$SERVER_WAIT_SECS" ]; do
        if curl --noproxy '*' -s -o /dev/null "$HEALTH_URL" 2>/dev/null; then
            echo "    server ready after ~${waited}s" | tee -a "$DRV_LOG"
            # 再稳一段，吸收 server 就绪后的任何 lazy init
            sleep 10
            return 0
        fi
        # 死亡检测只在宽限期后生效；且需连续 2 轮无进程才判死，
        # 避免启动早期（GPU preflight / python 未 exec）误判。
        if [ "$waited" -ge "$GRACE_SECS" ]; then
            if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 \
               && ! pgrep -f "EngineCore" >/dev/null 2>&1; then
                dead_rounds=$((dead_rounds + 1))
                if [ "$dead_rounds" -ge 2 ]; then
                    echo "    ERROR: server process died during startup; log tail:" | tee -a "$DRV_LOG"
                    tail -n 40 "$log" 2>/dev/null | tee -a "$DRV_LOG"
                    return 1
                fi
            else
                dead_rounds=0
            fi
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "    ERROR: /health not ready in ${SERVER_WAIT_SECS}s; log tail:" | tee -a "$DRV_LOG"
    tail -n 60 "$log" 2>/dev/null | tee -a "$DRV_LOG"
    return 1
}

# 简短 warmup：触发任何 lazy compile 后再计测
warmup() {
    echo "    warmup request (max_tokens=4) ..." | tee -a "$DRV_LOG"
    local body="{\"model\":\"$MODEL_PATH\",\"prompt\":\"法国的首都是\",\"max_tokens\":4}"
    curl --noproxy '*' -s -o /dev/null -m 300 \
        -H "Content-Type: application/json" \
        -d "$body" "http://127.0.0.1:${PORT}/v1/completions" || true
}

launch_server() {  # 前台阻塞，调用方后台化
    # 逐 side 显式 export NCCL env（source env.sh 的 ${VAR:-default} 不覆盖）
    export NCCL_PROTO="$NCCL_PROTO_V"
    export NCCL_ALGO="$NCCL_ALGO_V"
    export MTP="$SPEC"                 # on/off -> run 脚本 MTP
    export DRAFT="$DRAFT"
    export ZTH_W8A8_MODE="$ZTH"
    export MAX_MODEL_LEN="$MAX_MODEL_LEN"
    export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}"
    export CAPTURE_SIZES="${CAPTURE_SIZES:-1 2 4 8 16}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
    export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
    export EAGER="${EAGER:-off}"
    bash "$PROJECT_ROOT/deploy/run_tp8_zth_mtp_sglang_align.sh"
}

# ── 主流程 ──────────────────────────────────────────────────
stamp="$(date +%m%d_%H%M)"
DRV_LOG="$LOG_DIR/mtp_matrix_${TAG}_${stamp}.log"
gitrev="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

SIDE_LIST=()
case "$SIDES" in
    base_sr) SIDE_LIST=(base sr) ;;
    base)    SIDE_LIST=(base) ;;
    sr)      SIDE_LIST=(sr) ;;
    *) echo "SIDES must be base_sr|base|sr" >&2; exit 2 ;;
esac

{
    echo "=== MTP TPOT matrix driver ==="
    echo "  tag=$TAG sides=$SIDES spec=$SPEC draft=$DRAFT zth=$ZTH"
    echo "  matrix=$MATRIX conc=$CONC runs=$RUNS max_model_len=$MAX_MODEL_LEN"
    echo "  git=$gitrev  port=$PORT  out_dir=$OUT_DIR"
    echo "  (source env.sh NCCL default: PROTO=$NCCL_PROTO ALGO=$NCCL_ALGO)"
} | tee "$DRV_LOG"

if [[ "${CONFIRM_STOP_EXISTING_VLLM:-}" != "YES" ]]; then
    echo "Refusing to run: this driver stops existing vLLM processes and takes all" >&2
    echo "${GPU_COUNT} GPUs (server 逐侧冷启动, A/B 每侧一次)." >&2
    echo "Set CONFIRM_STOP_EXISTING_VLLM=YES after confirming impact." >&2
    exit 2
fi

declare -a result_files=()
for side in "${SIDE_LIST[@]}"; do
    set_side "$side"
    side_log="$LOG_DIR/server_${TAG}_${SIDE_LABEL}_${stamp}.log"
    echo ""
    echo ">>> SIDE=${side} (NCCL_PROTO=$NCCL_PROTO_V NCCL_ALGO=$NCCL_ALGO_V)"
    echo "    server log: $side_log" | tee -a "$DRV_LOG"

    stop_server
    echo "    launching server (bash deploy/run_tp8_zth_mtp_sglang_align.sh) ..." | tee -a "$DRV_LOG"
    launch_server >"$side_log" 2>&1 &
    SRV_PID=$!

    if ! wait_health "$side_log"; then
        echo "    FAIL: side $side server startup failed; aborting remaining sides" | tee -a "$DRV_LOG"
        kill -9 "$SRV_PID" 2>/dev/null || true
        stop_server
        exit 1
    fi
    echo "    server PID group: $SRV_PID" | tee -a "$DRV_LOG"
    warmup

    # server log 定位（run 脚本内 exec script，driver 只能按目录取最新同 tag 文件）
    # bench 客户端
    out_json="$OUT_DIR/${TAG}_${SIDE_LABEL}_${stamp}.json"
    echo "    bench: matrix=$MATRIX conc=$CONC runs=$RUNS" | tee -a "$DRV_LOG"
    python3 "$PROJECT_ROOT/benchmark/bench_hy3.py" \
        --endpoint "http://127.0.0.1:${PORT}" \
        --matrix "$MATRIX" --concurrency "$CONC" --runs "$RUNS" \
        --tag "${TAG}_${SIDE_LABEL}" --out "$out_json" \
        --timeout "$BENCH_TIMEOUT"
    rc=$?
    echo "    bench rc=$rc -> $out_json" | tee -a "$DRV_LOG"
    result_files+=("$out_json")

    # 收集 spec-decode accept 统计（vLLM SpecDecodingLogging 周期写日志 + /metrics；
    # 离线无法核实确切格式，先 best-effort 存档供 GPU 阶段校准 accept length 口径）
    stats_txt="$OUT_DIR/${TAG}_${SIDE_LABEL}_${stamp}_specstats.txt"
    {
        echo "# spec_decode accept stats capture (best-effort)"
        echo "## prometheus /metrics spec_decode rows:"
        curl --noproxy '*' -s -m 30 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
            | grep -iE "spec_decode|num_accepted" || echo "(no /metrics rows; prometheus may be off)"
        echo "## server log spec lines (tail 4000):"
        tail -n 4000 "$side_log" 2>/dev/null \
            | grep -iE "spec.?decode|num_accepted|accept.*token|draft" | tail -n 40 \
            || echo "(no spec lines in server log tail)"
    } > "$stats_txt" 2>&1 || true
    echo "    spec stats -> $stats_txt" | tee -a "$DRV_LOG"

    echo "    stopping server ..." | tee -a "$DRV_LOG"
    kill -9 "$SRV_PID" 2>/dev/null || true
    stop_server
done

echo ""
echo "=== A/B 归档 ===" | tee -a "$DRV_LOG"
printf '  %s\n' "${result_files[@]}" | tee -a "$DRV_LOG"
echo "  驱动日志: $DRV_LOG"
echo "  对比提示: mean_tpot_ms 用两 json 里 cells[].mean_tpot_ms 同 (input,output,conc) 比较"
