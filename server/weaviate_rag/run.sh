#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dev runner (Linux)
# - Starts Weaviate via Docker Compose
# - Waits for readiness
# - Frees backend/frontend ports
# - Launches FastAPI (uvicorn) + Next.js dev server
# - Cleans up on exit
# NOTE: Does NOT open a browser automatically.
# ============================================================

# --- Config (edit paths if needed) ---
BACKEND_DIR="./"
FRONTEND_DIR="../../client"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

log() { echo -e "[$(date '+%H:%M:%S')] $*"; }

# --- Pick compose command (v2 or legacy) ---
if command -v docker compose >/dev/null 2>&1; then
  COMPOSE() { docker compose "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE() { docker-compose "$@"; }
else
  echo "❌ Docker Compose not found. Install Docker or Docker Desktop." >&2
  exit 1
fi

# --- Helper: find active LAN IPv4 (Linux) ---
get_lan_ip() {
  if command -v ip >/dev/null 2>&1; then
    # Prefer the source IP used to reach the internet
    ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
  fi
}

LAN_IP="${LAN_IP:-$(get_lan_ip || true)}"
if [[ -z "${LAN_IP:-}" ]]; then
  # Fallback to first IPv4 from hostname -I, then loopback
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}') || true
fi
LAN_IP="${LAN_IP:-127.0.0.1}"

# These env vars are read by next.config.ts
export SITE_HOST="$LAN_IP"
export BACKEND_ORIGIN="http://${LAN_IP}:${BACKEND_PORT}"

# --- Start Weaviate (detached) ---
log "Starting Weaviate (detached)…"
COMPOSE up -d weaviate >/dev/null || true

# --- Wait for Weaviate to be READY ---
log "Waiting for Weaviate to report READY on :8080…"
for i in {1..60}; do
  if curl -fsS http://127.0.0.1:8080/v1/.well-known/ready >/dev/null; then
    log "Weaviate is READY ✅"
    break
  fi
  sleep 1
  if [[ $i -eq 60 ]]; then
    log "Weaviate did not become ready in time ❌"
    COMPOSE logs --tail 200 weaviate || true
    exit 1
  fi
done

# --- Free a port if a stale process is holding it ---
free_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -ti :"$port" >/dev/null 2>&1; then
    log "Port $port is in use; trying to free it…"
    # TERM then KILL any processes bound to the port
    kill -TERM $(lsof -ti :"$port") 2>/dev/null || true
    sleep 1
    if lsof -ti :"$port" >/dev/null 2>&1; then
      kill -KILL $(lsof -ti :"$port") 2>/dev/null || true
    fi
  fi
}

free_port "$BACKEND_PORT"
free_port "$FRONTEND_PORT"

# --- Start backend (FastAPI) ---
log "Starting FastAPI backend on :$BACKEND_PORT"
pushd "$BACKEND_DIR" >/dev/null
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!
BACKEND_PGID="$(ps -o pgid= "$BACKEND_PID" | tr -d ' ')"
popd >/dev/null

# --- Start frontend (Next.js) ---
log "Starting Next.js frontend on :$FRONTEND_PORT (SITE_HOST=$SITE_HOST, BACKEND_ORIGIN=$BACKEND_ORIGIN)"
pushd "$FRONTEND_DIR" >/dev/null
npm run dev -- -H 0.0.0.0 -p "$FRONTEND_PORT" &
FRONTEND_PID=$!
FRONTEND_PGID="$(ps -o pgid= "$FRONTEND_PID" | tr -d ' ')"
popd >/dev/null

# --- No automatic browser open (by request) ---
log "Frontend running at: http://${LAN_IP}:${FRONTEND_PORT}  (open manually if desired)"

# --- Cleanup on exit (Ctrl+C, TERM, normal exit) ---
CLEANUP_CALLED=0
cleanup() {
  # Ensure cleanup runs only once
  if (( CLEANUP_CALLED )); then
    return
  fi
  CLEANUP_CALLED=1

  # Prevent re-entry from traps during cleanup
  trap - INT TERM EXIT
  set +e

  log "Stopping frontend/backend…"

  # Use negative PGID to target process groups
  if [[ -n "${FRONTEND_PGID:-}" ]]; then
    kill -TERM -- -"${FRONTEND_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PGID:-}" ]]; then
    kill -TERM -- -"${BACKEND_PGID}" 2>/dev/null || true
  fi

  # Give them a moment to exit gracefully
  sleep 1

  if [[ -n "${FRONTEND_PGID:-}" ]]; then
    kill -KILL -- -"${FRONTEND_PGID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PGID:-}" ]]; then
    kill -KILL -- -"${BACKEND_PGID}" 2>/dev/null || true
  fi

  log "Stopping Weaviate…"
  COMPOSE down || true
}

# Trap signals:
# - On Ctrl+C (INT), cleanup and exit 130.
# - On TERM, cleanup and exit 143.
# - On normal EXIT, cleanup once.
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup' EXIT

# Keep the script alive until interrupted
while :; do sleep 3600; done
