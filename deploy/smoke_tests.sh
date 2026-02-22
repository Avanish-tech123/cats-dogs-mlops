#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test: hits /health and /predict
# Usage: ./deploy/smoke_tests.sh [BASE_URL]

BASE_URL=${1:-http://localhost:8000}
HEALTH_URL="$BASE_URL/health"
PREDICT_URL="$BASE_URL/predict"
METRICS_URL="$BASE_URL/metrics"
PERFORMANCE_URL="$BASE_URL/performance"

command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "jq is required"; exit 2; }
command -v python >/dev/null 2>&1 || { echo "python is required"; exit 2; }

echo "Checking health endpoint: $HEALTH_URL"
health=$(curl -sS --fail "$HEALTH_URL") || { echo "Health check request failed"; exit 3; }
echo "$health" | jq '.' || true
echo "$health" | jq -e '.status=="healthy" and .model_status=="ready"' >/dev/null || { echo "Health check indicates unhealthy"; echo "$health"; exit 4; }
echo "Health OK"

TMP_IMG=$(mktemp --suffix=.jpg)
python - <<PY
from PIL import Image
img = Image.new('RGB',(224,224),(255,0,0))
img.save("$TMP_IMG","JPEG")
print('Wrote', "$TMP_IMG")
PY

echo "Posting test image to $PREDICT_URL"
http_code=$(curl -sS -o response.json -w "%{http_code}" -F "file=@$TMP_IMG" "$PREDICT_URL") || { echo "Prediction request failed"; cat response.json || true; rm -f "$TMP_IMG" response.json; exit 5; }
if [ "$http_code" != "200" ]; then
  echo "Prediction endpoint returned $http_code"
  cat response.json || true
  rm -f "$TMP_IMG" response.json
  exit 6
fi
echo "Prediction response:"
cat response.json | jq '.' || true
cat response.json | jq -e '.predicted_class and .confidence' >/dev/null || { echo "Prediction response missing keys"; rm -f "$TMP_IMG" response.json; exit 7; }
rm -f "$TMP_IMG" response.json

echo "Testing metrics endpoint: $METRICS_URL"
metrics=$(curl -sS --fail "$METRICS_URL") || { echo "Metrics request failed"; exit 8; }
echo "$metrics" | jq '.' || true
echo "$metrics" | jq -e '.status=="active" and .total_predictions' >/dev/null || { echo "Metrics response invalid"; echo "$metrics"; exit 9; }
echo "Metrics OK"

echo "Testing performance endpoint: $PERFORMANCE_URL"
performance=$(curl -sS --fail "$PERFORMANCE_URL") || { echo "Performance request failed"; exit 10; }
echo "$performance" | jq '.' || true
echo "$performance" | jq -e '.total_predictions' >/dev/null || { echo "Performance response invalid"; echo "$performance"; exit 11; }
echo "Performance OK"

echo "Smoke tests passed"
