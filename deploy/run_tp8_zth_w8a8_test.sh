#!/bin/bash
# Launch the existing TP=8 service with the delivered W8A8 kernel hook enabled.
# This delegates process cleanup and server startup to the selected existing script.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DELIVER="${ZTH_W8A8_DELIVER:-$PROJECT_ROOT/算子优化/deliver}"
ZTH_W8A8_MODE="${ZTH_W8A8_MODE:-validate}"
ZTH_TARGET="${ZTH_TARGET:-base}"

case "$ZTH_W8A8_MODE" in
    off|validate|use) ;;
    *)
        echo "ZTH_W8A8_MODE must be off, validate, or use" >&2
        exit 2
        ;;
esac

case "$ZTH_TARGET" in
    base) LAUNCH_SCRIPT="$PROJECT_ROOT/deploy/run_tp8_single_80l.sh" ;;
    mtp) LAUNCH_SCRIPT="$PROJECT_ROOT/deploy/run_tp8_mtp_80l.sh" ;;
    *)
        echo "ZTH_TARGET must be base or mtp" >&2
        exit 2
        ;;
esac

if [[ ! -d "$DELIVER/python" || ! -d "$DELIVER/so_cache" ]]; then
    echo "Invalid deliver directory: $DELIVER" >&2
    exit 2
fi

if [[ "${CONFIRM_STOP_EXISTING_VLLM:-}" != "YES" ]]; then
    echo "Refusing to run: delegated launcher stops existing vLLM processes." >&2
    echo "Set CONFIRM_STOP_EXISTING_VLLM=YES after confirming the impact." >&2
    exit 2
fi

if [[ "$ZTH_W8A8_MODE" == "off" ]]; then
    PYTHONPATH="${PYTHONPATH#"$DELIVER/python:"}"
    unset ZTH_W8A8_CACHE ZTH_W8A8_MODE
else
    export PYTHONPATH="$DELIVER/python${PYTHONPATH:+:$PYTHONPATH}"
    export ZTH_W8A8_CACHE="$DELIVER/so_cache"
    export ZTH_W8A8_MODE
fi

echo "ZTH W8A8 mode: ${ZTH_W8A8_MODE:-off}"
echo "ZTH W8A8 deliver: $DELIVER"
echo "Stopping existing TP8 service before delegated preflight."
pkill -9 -f vllm.entrypoints 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f Worker_TP 2>/dev/null || true
sleep 2
echo "Delegating to: $LAUNCH_SCRIPT"
exec bash "$LAUNCH_SCRIPT"
