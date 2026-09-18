#!/usr/bin/env bash
# Starts the SentraGrade scanner UI and exposes it via a public Cloudflare
# quick tunnel. The tunnel URL is random and only live while this script runs.
set -euo pipefail

cd "$(dirname "$0")"
PORT=8731
URL="http://127.0.0.1:${PORT}"
TUNNEL_LOG="$(mktemp)"

cleanup() {
  echo ""
  echo "Stopping tunnel and server..."
  [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$TUNNEL_LOG"
}
trap cleanup EXIT

if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
  echo "Something is already running on port ${PORT}, stopping it first..."
  pkill -f "uvicorn app.server:app" 2>/dev/null || true
  sleep 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared isn't installed. Install it with: brew install cloudflared"
  exit 1
fi

echo "Starting SentraGrade scanner on ${URL} ..."
uv run uvicorn app.server:app --port "${PORT}" --host 127.0.0.1 &
SERVER_PID=$!

until curl -s -o /dev/null "${URL}/api/meta"; do
  sleep 0.5
done
echo "Server ready."

echo "Opening a public tunnel..."
cloudflared tunnel --url "${URL}" > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

PUBLIC_URL=""
for _ in $(seq 1 60); do
  PUBLIC_URL="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
  [[ -n "$PUBLIC_URL" ]] && break
  sleep 1
done

if [[ -z "$PUBLIC_URL" ]]; then
  echo "Couldn't get a tunnel URL — check $TUNNEL_LOG"
  exit 1
fi

echo ""
echo "=============================================="
echo " Live at: ${PUBLIC_URL}"
echo " (random URL, only works while this is running)"
echo " Press Ctrl+C to stop."
echo "=============================================="
echo ""
open "${PUBLIC_URL}" 2>/dev/null || true

wait "$SERVER_PID" "$TUNNEL_PID"
