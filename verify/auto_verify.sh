#!/bin/bash
# Auto-wait for vLLM server to be ready, then run verification
set -e

SERVER_URL="http://localhost:8000"
COMPARE_SCRIPT="/data/mzw/vllm-hy3/compare_80l_full.py"
DUMP_DIR="/data/mzw/vllm-hy3/dumps/pp2_80l"
REPORT="/data/mzw/vllm-hy3/verification_80l_$(date +%m%d_%H%M).txt"

echo "=== Auto-Verify started at $(date) ==="
echo "Waiting for server on $SERVER_URL ..."

# Wait up to 30 minutes
for i in $(seq 1 180); do
    if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "$(date) ✓ Server ready after ${i}0 seconds!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "$(date) ✗ Timeout after 30 minutes"
        exit 1
    fi
    # Sleep 10s but check every iteration
    for j in $(seq 1 10); do
        sleep 1
        # Check for terminal errors
        NODE0_LOG=$(ls -t /data/mzw/vllm-hy3/logs/vllm_node0_pp2_80l_*.log 2>/dev/null | head -1)
        if grep -q "RuntimeError\|ERROR.*WorkerProc failed\|Engine core initialization failed" "$NODE0_LOG" 2>/dev/null; then
            echo "$(date) ✗ Error detected in Node 0 log!"
            grep -E "RuntimeError|ERROR.*WorkerProc|Engine core" "$NODE0_LOG" | tail -5
            exit 1
        fi
    done
    echo -n "."
done
echo ""

# Send test request
echo "$(date) Sending test request..."
RESPONSE=$(curl -s "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"hy3","prompt":"中国的首都是","max_tokens":5,"temperature":0.0}')

GENERATED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "PARSE_FAILED")
echo "Generated: '$GENERATED'"
echo ""

# Wait for dumps
echo "$(date) Waiting for dumps..."
for i in $(seq 1 30); do
    sleep 2
    LAYER_COUNT=$(ls "$DUMP_DIR"/layer_*/ 2>/dev/null | wc -l)
    echo "  [$i] Layer dirs: $LAYER_COUNT"
    if [ -f "$DUMP_DIR/layer_080/logits.pt" ] && [ "$LAYER_COUNT" -ge 80 ]; then
        echo "  ✓ Dumps ready!"
        break
    fi
done
echo ""

# Run comparison
echo "$(date) Running comparison..."
python3 "$COMPARE_SCRIPT" --dump-dir "$DUMP_DIR" --output-report "$REPORT"

echo ""
echo "$(date) === Verification complete ==="
echo "Report: $REPORT"
