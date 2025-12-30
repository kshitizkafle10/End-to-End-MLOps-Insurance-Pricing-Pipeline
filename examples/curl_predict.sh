#!/usr/bin/env bash
set -e

curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json | python3 -m json.tool
