#!/bin/bash
# Wait for vLLM server to be ready, then send test request and run comparison
# Usage: bash test_and_compare.sh

set -e

source "$(dirname "$0")/../deploy/env.sh"

SERVER_URL="http://localhost:8000"
DUMP_DIR="$PROJECT_ROOT/dumps/pp2_80l"
COMPARE_SCRIPT="$PROJECT_ROOT/verify/compare_80l_full.py"
REPORT_FILE="$PROJECT_ROOT/verification_report_80l_latest.txt"

echo "=== HY3 80L Verification Test ==="
echo "Started at: $(date)"
echo ""

# ── Wait for server ─────────────────────────────────────
echo "[$(date)] Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "[$(date)] ✓ Server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[$(date)] ✗ Server did not start within 5 minutes."
        echo "Check logs:"
        echo "  Node 0 log: $LOG_DIR/vllm_node0_pp2_80l_*.log"
        echo "  Node 1 log: /tmp/node1_80l.log (in Docker on node 1)"
        exit 1
    fi
    sleep 5
    echo -n "."
done
echo ""

# ── Check models ────────────────────────────────────────
echo "[$(date)] Checking available models..."
curl -s "$SERVER_URL/v1/models" | python3 -m json.tool 2>/dev/null | head -10
echo ""

# ── Send test request ───────────────────────────────────
echo "[$(date)] Sending test completion request..."
RESPONSE=$(curl -s "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy3",
    "prompt": "中国的首都是",
    "max_tokens": 5,
    "temperature": 0.0
  }')

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Extract generated text
GENERATED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "PARSE_FAILED")
echo "Generated text: '$GENERATED'"
echo ""

# ── Wait for dumps ──────────────────────────────────────
echo "[$(date)] Waiting for dumps to be written..."
for i in $(seq 1 10); do
    sleep 2
    LAYER_COUNT=$(ls "$DUMP_DIR"/layer_*/00_input.pt 2>/dev/null | wc -l)
    echo "  Layer dumps found: $LAYER_COUNT"
    if [ -f "$DUMP_DIR/layer_080/logits.pt" ]; then
        echo "  ✓ Logits dump found!"
        break
    fi
done
echo ""

# ── List dumped layers ──────────────────────────────────
echo "[$(date)] Dumped layers:"
ls "$DUMP_DIR/" | sort -V | head -20
echo "  ..."
echo "Total layer dirs: $(ls -d "$DUMP_DIR"/layer_*/ 2>/dev/null | wc -l)"
echo ""

# ── Run comparison ──────────────────────────────────────
echo "[$(date)] Running comparison against golden dumps..."
python3 "$COMPARE_SCRIPT" --dump-dir "$DUMP_DIR" --output-report "$REPORT_FILE" 2>&1

echo ""
echo "[$(date)] === Test complete ==="
echo "Report saved to: $REPORT_FILE"

# ── Clean shutdown ──────────────────────────────────────
echo ""
echo "To stop the server: kill \$(pgrep -f 'api_server')"
