#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"

cd "$ROOT_DIR"
exec python3 -m streamlit run app.py \
  --server.port "$PORT" \
  --server.address "$HOST" \
  --browser.gatherUsageStats false \
  "$@"
