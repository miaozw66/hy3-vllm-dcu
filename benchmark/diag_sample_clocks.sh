#!/bin/bash
# Load-time DCU clock/use/power sampler for bottleneck diagnosis.
# Usage: bash benchmark/diag_sample_clocks.sh [seconds] [interval]
#   default: 180s, every 3s → 60 points
SECS=${1:-180}
INT=${2:-3}
OUT="${3:-benchmark/diag_clocks_$(date +%Y%m%d_%H%M%S).txt}"

echo "sampling ${SECS}s every ${INT}s → $OUT"
for i in $(seq 1 $((SECS / INT))); do
    ts=$(date +%H:%M:%S)
    busy=$(for c in 1 2 3 4 5 6 7 8; do printf "%s " "$(cat /sys/class/drm/card${c}/device/gpu_busy_percent 2>/dev/null)"; done)
    sclk=$(hy-smi --showdcuclocks 2>/dev/null | grep 'sclk clock level' | sed 's/.*level: \([0-9]\) (\([0-9]*\)Mhz).*/\1@\2/' | tr '\n' ' ')
    pw=$(hy-smi -P 2>/dev/null | grep 'Average Graphics Package Power' | sed 's/.*(W): \([0-9.]*\).*/\1/' | tr '\n' ' ')
    tmp=$(hy-smi -t 2>/dev/null | grep 'Sensor edge' | sed 's/.*(C): \([0-9.]*\).*/\1/' | tr '\n' ' ')
    echo "[$ts] busy=[$busy] sclk=[$sclk] W=[$pw] temp=[$tmp]"
    sleep "$INT"
done
echo "=== sampling done ==="
