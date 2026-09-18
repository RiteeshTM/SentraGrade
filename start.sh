#!/usr/bin/env bash
# Starts the SentraGrade scanner UI and opens it in your browser.
set -euo pipefail

cd "$(dirname "$0")"
PORT=8731
URL="http://127.0.0.1:${PORT}"

if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
  echo "Something is already running on port ${PORT}, stopping it first..."
  pkill -f "uvicorn app.server:app" 2>/dev/null || true
  sleep 1
fi

echo "Starting SentraGrade scanner on ${URL} ..."
uv run uvicorn app.server:app --port "${PORT}" --host 127.0.0.1 &
SERVER_PID=$!

trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

until curl -s -o /dev/null "${URL}/api/meta"; do
  sleep 0.5
done

echo "Ready -> ${URL}"
open "${URL}" 2>/dev/null || true

wait "$SERVER_PID"
