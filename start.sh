#!/usr/bin/env bash
# Ares startup — brings up all services and drops into the REPL.
#   ./start.sh         start what isn't running, then open the REPL
#   ./start.sh stop    stop the A2A agent servers
set -euo pipefail

ARES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ARES_PYTHON:-$ARES_DIR/.venv/bin/python}"
STATE_DIR="$HOME/.ares"
A2A_PID="$STATE_DIR/a2a.pid"
A2A_LOG="$STATE_DIR/logs/a2a.log"

# ── colours ──────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[0;33m'; R='\033[0;31m'; N='\033[0m'
ok()   { echo -e "${G}[ok]${N}  $*"; }
warn() { echo -e "${Y}[warn]${N} $*"; }
err()  { echo -e "${R}[err]${N} $*"; }

a2a_running() { [ -f "$A2A_PID" ] && kill -0 "$(cat "$A2A_PID")" 2>/dev/null; }

if [ "${1:-}" = "stop" ]; then
    if a2a_running; then
        kill "$(cat "$A2A_PID")" && rm -f "$A2A_PID"
        ok "A2A agent servers stopped"
    else
        rm -f "$A2A_PID"
        warn "A2A agent servers were not running"
    fi
    exit 0
fi

if ! "$PYTHON" -c "import ares" &>/dev/null; then
    err "Ares is not installed for $PYTHON"
    echo "      Run: uv venv && uv pip install -e ."
    echo "      (or set ARES_PYTHON to an interpreter that has it installed)"
    exit 1
fi

# ── Load secrets from .env (never hardcode keys in this file) ────────────────
if [ -f "$ARES_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ARES_DIR/.env"
    set +a
else
    warn "No .env found — copy .env.example to .env and fill in your keys."
fi

OLLAMA_URL="${ARES_OLLAMA_HOST:-http://localhost:11434}"
LANGFUSE_URL="${LANGFUSE_HOST:-http://localhost:3000}"

echo -e "\n${G}Ares — starting up${N}\n"

# ── 1. Ollama ─────────────────────────────────────────────────────────────────
if curl -sf "$OLLAMA_URL/api/tags" &>/dev/null; then
    ok "Ollama running at $OLLAMA_URL"
elif [[ "$OLLAMA_URL" =~ ^http://(localhost|127\.0\.0\.1)(:|/|$) ]]; then
    warn "Ollama not detected — starting..."
    ollama serve &>/dev/null &
    for _ in $(seq 1 15); do
        curl -sf "$OLLAMA_URL/api/tags" &>/dev/null && break
        sleep 1
    done
    curl -sf "$OLLAMA_URL/api/tags" &>/dev/null && ok "Ollama started" || { err "Ollama failed to start"; exit 1; }
else
    err "Ollama not reachable at $OLLAMA_URL"; exit 1
fi

# ── 2. Required model check ───────────────────────────────────────────────────
missing_models=$("$PYTHON" -c "
from ares.engine import check_required_models
print('\n'.join(check_required_models()))
" 2>/dev/null || true)

if [ -n "$missing_models" ]; then
    warn "Missing Ollama models (pull them before continuing):"
    while IFS= read -r model; do
        warn "  ollama pull $model"
    done <<< "$missing_models"
    read -rp "  Continue anyway? [y/N] " _yn
    [[ "$_yn" =~ ^[Yy]$ ]] || { err "Aborted."; exit 1; }
fi

# ── 3. Langfuse (optional, Docker) ────────────────────────────────────────────
if curl -sf "$LANGFUSE_URL/api/public/health" &>/dev/null; then
    ok "Langfuse running at $LANGFUSE_URL"
elif ! command -v docker &>/dev/null; then
    warn "Docker not found — skipping Langfuse (tracing still goes to ~/.ares/traces.db)"
elif docker compose -f "$ARES_DIR/docker/langfuse/docker-compose.yml" up -d &>/dev/null; then
    for _ in $(seq 1 30); do
        curl -sf "$LANGFUSE_URL/api/public/health" &>/dev/null && break
        sleep 2
    done
    if curl -sf "$LANGFUSE_URL/api/public/health" &>/dev/null; then
        ok "Langfuse started ($LANGFUSE_URL)"
    else
        warn "Langfuse did not come up within 60s — continuing without it"
    fi
else
    warn "Could not start the Langfuse stack — continuing without it"
fi

# ── 4. A2A agent servers ──────────────────────────────────────────────────────
declare -A A2A_PORTS=([orchestrator]=8100 [coder]=8101 [thinker]=8102 [runner]=8103 [serena]=8104)
AGENTS=(orchestrator coder thinker runner serena)
a2a_up() { curl -sf "http://127.0.0.1:$1/health" &>/dev/null; }

if a2a_running; then
    ok "A2A agent servers already running (pid $(cat "$A2A_PID"))"
else
    warn "Starting A2A agent servers (log: $A2A_LOG)..."
    mkdir -p "$(dirname "$A2A_LOG")"
    # Outlives this script and the REPL; stop it with ./start.sh stop.
    nohup "$PYTHON" -m ares.a2a_server >>"$A2A_LOG" 2>&1 &
    echo $! > "$A2A_PID"

    for _ in $(seq 1 40); do
        all_ready=true
        for agent in "${AGENTS[@]}"; do
            a2a_up "${A2A_PORTS[$agent]}" || all_ready=false
        done
        [ "$all_ready" = true ] && break
        a2a_running || break
        sleep 3
        echo -n "."
    done
    echo
fi

for agent in "${AGENTS[@]}"; do
    port="${A2A_PORTS[$agent]}"
    if a2a_up "$port"; then
        ok "A2A $agent :$port"
    else
        warn "A2A $agent :$port is not responding (see $A2A_LOG)"
    fi
done

# ── 5. Drop into REPL ────────────────────────────────────────────────────────
echo
echo -e "${G}Services ready.${N}"
echo -e "  Langfuse UI  : $LANGFUSE_URL"
echo -e "  A2A agents   : :8100 (orch) :8101 (coder) :8102 (thinker) :8103 (runner) :8104 (serena)"
echo -e "  Stop agents  : ./start.sh stop"
echo
exec "$PYTHON" -m ares
