#!/bin/bash
# Random-ids online benchmark for HY3, aligned with the sglang reference table.
# Runs the 1K~16K case matrix (input/output len pairs) x concurrency 1/2/4/8/16
# against an ALREADY-RUNNING server, one of {mtp, base}.
#
# Server lifecycle (separate scripts, like the 8k/16k tail benchmark):
#   deploy/run_tp8_bench_mtp_8k16k.sh            # launch MTP server
#   bash benchmark/run_bench_random_ids.sh mtp   # this script
#   bash deploy/switch_to_baseline.sh            # relaunch as baseline
#   bash benchmark/run_bench_random_ids.sh base  # this script
#
# Cases (1K~16K): 1K/1K 2K/1K 2K/2K 4K/1K 7K/2K 16K/1K
#   (= the first six rows of the sglang table, which extends to 128K/1K)
#
# Usage:
#   bash benchmark/run_bench_random_ids.sh <mtp|base>
set -e
cd "$(dirname "$0")/.."

TAG="$1"
[ -z "$TAG" ] && { echo "usage: $0 <mtp|base>"; exit 1; }
END=${ENDPOINT:-http://localhost:8000}
OUTDIR="benchmark/results_online_rids"
mkdir -p "$OUTDIR"

CASES="1024/1024,2048/1024,2048/2048,4096/1024,7168/2048,16384/1024"
CONCS=${CONCS:-1,2,4,8,16}

echo "=== random-ids bench: TAG=$TAG cases='$CASES' conc='$CONCS' ==="
python3 benchmark/bench_random_ids.py \
    --endpoint "$END" \
    --cases "$CASES" \
    --concurrencies "$CONCS" \
    --runs ${RUNS:-1} --ignore-eos \
    --seed ${SEED:-42} \
    --output "$OUTDIR/${TAG}_cases.json"

echo "=== DONE ${TAG} ==="
ls -la "$OUTDIR/${TAG}_cases.json"
