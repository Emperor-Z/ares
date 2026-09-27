#!/usr/bin/env bash
# Ares startup — brings up all services and drops into the REPL.
set -euo pipefail

ARES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ARES_PYTHON:-$ARES_DIR/.venv/bin/python}"

if ! "$PYTHON" -c "import ares" &>/dev/null; then
    echo "[err] Ares is not installed for $PYTHON"
    echo "      Run: uv venv && uv pip install -e ."
    echo "      (or set ARES_PYTHON to an interpreter that has it installed)"
    exit 1
fi

# ── Load secrets from .env (never hardcode keys in this file) ────────────────
if [ -f "$ARES_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ARES_DIR/.env"
    set +a
else
    echo "[warn] No .env found — copy .env.example to .env and fill in your keys."
fi

# ── colours ──────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[0;33m'; R='\033[0;31m'; N='\033[0m'
ok()   { echo -e "${G}[ok]${N}  $*"; }
warn() { echo -e "${Y}[warn]${N} $*"; }
err()  { echo -e "${R}[err]${N} $*"; }

echo -e "\n${G}Ares — starting up${N}\n"

# ── 1. Ollama ─────────────────────────────────────────────────────────────────
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    ok "Ollama already running"
else
    warn "Ollama not detected — starting..."
    ollama serve &>/dev/null &
    sleep 3
    curl -sf http://localhost:11434/api/tags &>/dev/null && ok "Ollama started" || { err "Ollama failed to start"; exit 1; }
fi

# ── 2. Required model check ───────────────────────────────────────────────────
missing_models=$("$PYTHON" -c "
from ares.engine import check_required_models
missing = check_required_models()
print('\n'.join(missing))
" 2>/dev/null || true)

if [ -n "$missing_models" ]; then
    warn "Missing Ollama models (pull them before continuing):"
    while IFS= read -r model; do
        warn "  ollama pull $model"
    done <<< "$missing_models"
    read -rp "  Continue anyway? [y/N] " _yn
    [[ "$_yn" =~ ^[Yy]$ ]] || { err "Aborted."; exit 1; }
fi

# ── 3. Langfuse (Docker) ──────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q ares_langfuse; then
    ok "Langfuse already running (http://localhost:3000)"
else
    warn "Langfuse not running — starting Docker stack..."
    docker compose -f "$ARES_DIR/docker/langfuse/docker-compose.yml" up -d &>/dev/null
    until curl -sf http://localhost:3000/api/public/health &>/dev/null; do sleep 2; done
    ok "Langfuse started (http://localhost:3000)"
fi

# ── 4. A2A agent servers ──────────────────────────────────────────────────────
a2a_up() {
    curl -sf "http://127.0.0.1:$1/health" &>/dev/null
}

declare -A A2A_PORTS=([orchestrator]=8100 [coder]=8101 [thinker]=8102 [runner]=8103 [serena]=8104)
all_up=true
for agent in orchestrator coder thinker runner serena; do
    port="${A2A_PORTS[$agent]}"
    if a2a_up "$port"; then
        ok "A2A $agent already on :$port"
    else
        all_up=false
    fi
done

if [ "$all_up" = false ]; then
    warn "Starting A2A agent servers..."
    "$PYTHON" -c "
from ares.a2a_server import launch_all
procs = launch_all(background=True)
import time; time.sleep(8)
" &

    timeout=120
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        all_ready=true
        for agent in orchestrator coder thinker runner serena; do
            port="${A2A_PORTS[$agent]}"
            a2a_up "$port" || all_ready=false
        done
        [ "$all_ready" = true ] && break
        sleep 3; elapsed=$((elapsed + 3))
        echo -n "."
    done
    echo

    for agent in orchestrator coder thinker runner serena; do
        port="${A2A_PORTS[$agent]}"
        if a2a_up "$port"; then
            ok "A2A $agent :$port"
        else
            warn "A2A $agent :$port did not come up in time (continuing anyway)"
        fi
    done
fi

# ── 5. Drop into REPL ────────────────────────────────────────────────────────
echo
echo -e "${G}All services ready.${N}"
echo -e "  Langfuse UI  : http://localhost:3000"
echo -e "  A2A agents   : :8100 (orch) :8101 (coder) :8102 (thinker) :8103 (runner) :8104 (serena)"
echo
exec "$PYTHON" -m ares
