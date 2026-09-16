#!/bin/bash
# Kill current MTP bench server and relaunch as baseline (SPEC=none) for -O1 8k/16k benchmark.
# Usage: bash deploy/switch_to_baseline.sh
set -e
source "$(dirname "$0")/env.sh"

echo "=== Stopping current server ==="
pkill -9 -f vllm.entrypoints 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f Worker_TP 2>/dev/null || true
sleep 3
echo "Remaining: $(pgrep -f 'vllm.entrypoints|EngineCore|Worker_TP' | wc -l)"

echo "=== Launching baseline server (SPEC=none) ==="
exec SPEC=none bash "$(dirname "$0")/run_tp8_bench_mtp_8k16k.sh"
